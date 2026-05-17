"""Export Sprint 1 text encoder ONNX handoff package.

Outputs:
  mobile_artifacts/text_encoder.onnx
  mobile_artifacts/text_tokenizer.json   (optional, when tokenizer JSON is exposed)
  mobile_artifacts/text_preprocessing.json
  mobile_artifacts/manifest.json (updated with text_encoder section)

Why this exists:
  Sprint 1 originally only exported the image encoder. The Sprint Backlog US-07
  (T18 "경량 ONNX 모델 준비") and US-02/06 (Top-3 results from natural-language
  search) both expect the text path to work end-to-end on the same ONNX runtime.
  Without a text encoder ONNX, the ONNX/INT8 demo modes silently fall back to
  keyword/alias matching only, which makes natural-language search look like a
  category lookup rather than semantic retrieval.

Notes:
  - We export the CLIP text tower (no head). The downstream prototype/text index
    already lives in 512-dim L2-normalized space, so the ONNX path returns a
    L2-normalized 512-d embedding to match.
  - Context length = 77, dtype = int64. Matches open_clip MobileCLIP2 tokenizer.
  - We also dump the tokenizer JSON when accessible. If tokenizer is a callable
    object without an obvious export, we still record the open_clip tokenizer
    name in text_preprocessing.json so the runtime can reconstruct it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landmark_demo.model import load_checkpoint


CONTEXT_LENGTH = 77


class TextEmbeddingModule(nn.Module):
    """Wraps the CLIP text tower to emit an L2-normalized 512-d embedding."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.clip = model.clip

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        features = self.clip.encode_text(tokens).float()
        return F.normalize(features, dim=-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="./best.pt")
    p.add_argument("--output-dir", default="./mobile_artifacts")
    p.add_argument("--opset", type=int, default=18)
    return p.parse_args()


def _try_export_tokenizer(tokenizer, output_dir: Path) -> dict | None:
    """Best-effort dump of tokenizer state. open_clip's HFTokenizer wraps a
    transformers tokenizer; if we can pull a vocab JSON we ship it. Otherwise
    we just record the open_clip tokenizer name so the runtime can call
    `open_clip.get_tokenizer(name)` again.
    """
    try:
        # HFTokenizer path
        if hasattr(tokenizer, "tokenizer"):
            inner = tokenizer.tokenizer
            tok_path = output_dir / "text_tokenizer"
            tok_path.mkdir(parents=True, exist_ok=True)
            try:
                inner.save_pretrained(str(tok_path))
                return {
                    "tokenizer_dir": str(tok_path.name),
                    "tokenizer_type": "huggingface",
                    "context_length": CONTEXT_LENGTH,
                }
            except Exception as exc:
                print(f"[tok] save_pretrained failed: {exc}")
    except Exception as exc:
        print(f"[tok] inspection failed: {exc}")

    return {
        "tokenizer_type": "open_clip_runtime",
        "open_clip_name": "MobileCLIP2-S4",
        "context_length": CONTEXT_LENGTH,
        "note": "Runtime should call open_clip.get_tokenizer(open_clip_name).",
    }


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {checkpoint}")
    model, _classes, train_cfg = load_checkpoint(str(checkpoint), device="cpu")

    import open_clip
    tokenizer_name = train_cfg["model"]["model_name"]
    tokenizer = open_clip.get_tokenizer(tokenizer_name)

    wrapper = TextEmbeddingModule(model).eval()

    # Smoke check: real query through the wrapper
    sample_tokens = tokenizer(["royal palace gate"]).to(torch.int64)
    with torch.no_grad():
        sample_embedding = wrapper(sample_tokens)
    embedding_dim = int(sample_embedding.shape[-1])
    print(f"[smoke] tokens shape={tuple(sample_tokens.shape)}  "
          f"embedding shape={tuple(sample_embedding.shape)}  norm={sample_embedding.norm(dim=-1).item():.4f}")

    onnx_path = output_dir / "text_encoder.onnx"
    dummy = torch.zeros((1, CONTEXT_LENGTH), dtype=torch.int64)
    print(f"[export] {onnx_path} (opset={args.opset})")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            input_names=["input_ids"],
            output_names=["text_embedding"],
            opset_version=args.opset,
            dynamic_axes={"input_ids": {0: "batch"}, "text_embedding": {0: "batch"}},
            do_constant_folding=True,
        )

    # The dynamo-based exporter sometimes spills weights into a sidecar
    # `text_encoder.onnx.data`. That makes the file fragile when copied across
    # drives. Reload and rewrite with everything inline so a single .onnx is
    # self-contained for the runtime.
    import onnx
    sidecar = output_dir / (onnx_path.name + ".data")
    if sidecar.exists():
        print(f"[export] inlining external data sidecar -> {sidecar.name}")
        proto = onnx.load(str(onnx_path), load_external_data=True)
        onnx.save(proto, str(onnx_path), save_as_external_data=False)
        if sidecar.exists():
            sidecar.unlink()

    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"[export] ok  size={onnx_size_mb:.1f} MB")

    tok_meta = _try_export_tokenizer(tokenizer, output_dir)

    text_preprocessing = {
        "version": "sprint1-text-preprocessing-v1",
        "input_name": "input_ids",
        "output_name": "text_embedding",
        "context_length": CONTEXT_LENGTH,
        "input_dtype": "int64",
        "output_dim": embedding_dim,
        "output_l2_normalized": True,
        "tokenizer": tok_meta,
        "open_clip_name": tokenizer_name,
    }
    (output_dir / "text_preprocessing.json").write_text(
        json.dumps(text_preprocessing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Update manifest.json (merge if present)
    manifest_path = output_dir / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    manifest["text_encoder"] = {
        "onnx": str(onnx_path.resolve()),
        "preprocessing": str((output_dir / "text_preprocessing.json").resolve()),
        "size_mb": round(onnx_size_mb, 2),
        "opset": args.opset,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Cross-runtime parity check
    print("[parity] PyTorch vs ONNX cosine on a few queries")
    try:
        import onnxruntime as ort
    except ImportError:
        print("[parity] onnxruntime not installed, skipping cross-check")
        return

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0].name
    out = session.get_outputs()[0].name

    queries = [
        "royal palace gate",
        "art museum in Seoul",
        "한복 입고 사진 찍는 곳",
        "조선 왕조 제사",
        "stream walk in Seoul",
    ]
    import numpy as np
    for q in queries:
        toks = tokenizer([q]).to(torch.int64)
        with torch.no_grad():
            ref = wrapper(toks).cpu().numpy()[0]
        ort_out = session.run([out], {inp: toks.cpu().numpy()})[0][0]
        ort_out = ort_out / (np.linalg.norm(ort_out) + 1e-12)
        cos = float(np.dot(ref / (np.linalg.norm(ref) + 1e-12), ort_out))
        print(f"  cos={cos:.5f}  query='{q}'")


if __name__ == "__main__":
    main()
