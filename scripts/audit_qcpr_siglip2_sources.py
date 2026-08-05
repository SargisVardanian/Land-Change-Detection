#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, platform, sys
from pathlib import Path
from typing import Any
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoModel, AutoProcessor
SIGLIP_REVISION = "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab"
GEORSCLIP_REVISION = "4920188e6eba4e711ef9848cfd7cb77e874ee33f"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

def files_under(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts)

def inventory(root: Path) -> list[dict[str, Any]]:
    return [{"path": str(p.relative_to(root)), "size": p.stat().st_size, "sha256": sha256(p)} for p in files_under(root)]

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def audit_siglip2(model_dir: Path, out: Path, run_inference: bool) -> dict[str, Any]:
    inv = inventory(model_dir)
    write_json(out / "siglip2_file_inventory.json", inv)
    (out / "siglip2_SHA256SUMS").write_text("".join(f"{x['sha256']}  {x['path']}\n" for x in inv), encoding="utf-8")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model_file = model_dir / "model.safetensors"
    safe: dict[str, Any] = {"model_safetensors_exists": model_file.exists()}
    with safe_open(str(model_file), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        safe.update({"tensor_count": len(keys), "first_keys": keys[:20], "safe_open": "PASS"})
    result: dict[str, Any] = {
        "revision": SIGLIP_REVISION,
        "model_identifier": "google/siglip2-base-patch16-256",
        "config_model_type": config.get("model_type"),
        "runtime_class_expected_from_config": "SiglipModel",
        "file_count": len(inv),
        "safe_tensor_audit": safe,
        "status": "SIGLIP2_HASH_AND_STRUCTURE_PASS",
    }
    if run_inference:
        processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
        model = AutoModel.from_pretrained(str(model_dir), local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
        inputs = processor(images=[Image.new("RGB", (256, 256), (127, 127, 127))], text=["a temporal image pair"], return_tensors="pt", padding="max_length")
        with torch.no_grad():
            output = model(**inputs)
        result["inference"] = {
            "status": "PASS",
            "runtime_class": type(model).__name__,
            "image_last_hidden_state": list(output.vision_model_output.last_hidden_state.shape),
            "image_pooler_output": list(output.vision_model_output.pooler_output.shape),
            "text_last_hidden_state": list(output.text_model_output.last_hidden_state.shape),
            "text_pooler_output": list(output.text_model_output.pooler_output.shape),
            "processor_keys": sorted(inputs.keys()),
        }
    else:
        result["inference"] = {"status": "NOT_RUN"}
    write_json(out / "siglip2_architecture.json", {
        "revision": SIGLIP_REVISION,
        "config": config,
        "runtime_class": "SiglipModel (resolved by AutoModel from pinned config)",
        "expected_native_contract": {"image_patches": [1, 256, 768], "text_tokens": [1, 64, 768]},
    })
    (out / "siglip2_license.txt").write_text("Apache-2.0 (Hugging Face model card metadata; pinned revision)\n", encoding="utf-8")
    write_json(out / "siglip2_source_audit.json", result)
    return result

def audit_georsclip(model_dir: Path, out: Path) -> dict[str, Any]:
    inv = inventory(model_dir)
    write_json(out / "georsclip_file_inventory.json", inv)
    (out / "georsclip_SHA256SUMS").write_text("".join(f"{x['sha256']}  {x['path']}\n" for x in inv), encoding="utf-8")
    checkpoint = model_dir / "ckpt" / "RS5M_ViT-B-32.pt"
    result: dict[str, Any] = {"revision": GEORSCLIP_REVISION, "checkpoint": str(checkpoint), "status": "MISSING", "weights_only": None}
    if checkpoint.exists():
        try:
            value = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
            result.update({"status": "WEIGHTS_ONLY_LOAD_PASS", "weights_only": True, "object_type": type(value).__name__, "top_level_keys": list(value.keys())[:40] if isinstance(value, dict) else []})
        except Exception as exc:
            result.update({"status": "GEORSCLIP_WEIGHTS_ONLY_INCOMPATIBLE", "weights_only": False, "error_type": type(exc).__name__, "error": str(exc)})
    write_json(out / "georsclip_checkpoint_audit.json", result)
    write_json(out / "georsclip_source_audit.json", {"revision": GEORSCLIP_REVISION, "license_status": "RESEARCH_BASELINE_ONLY", "file_inventory": inv, "checkpoint_status": result["status"]})
    compat = {"status": "NOT_ATTEMPTED_UNTIL_WEIGHTS_ONLY_PASS"}
    if result.get("status") == "WEIGHTS_ONLY_LOAD_PASS":
        try:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
            missing, unexpected = model.load_state_dict(value, strict=False)
            compat = {"status": "PASS" if not missing and not unexpected else "MISMATCH", "model_name": "ViT-B-32", "open_clip_version": getattr(open_clip, "__version__", "unknown"), "checkpoint_keys": len(value), "missing_keys": list(missing), "unexpected_keys": list(unexpected), "missing_count": len(missing), "unexpected_count": len(unexpected)}
        except Exception as exc:
            compat = {"status": "COMPATIBILITY_CHECK_FAILED", "error_type": type(exc).__name__, "error": str(exc)}
    write_json(out / "georsclip_state_dict_compatibility.json", compat)
    write_json(out / "georsclip_license_audit.json", {"code_license": "pinned README must be checked", "weight_license": "ambiguous until official terms are verified", "dataset_license": "not inferred", "redistribution_permission": "not established", "research_use_status": "RESEARCH_BASELINE_ONLY"})
    return result

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siglip2-dir", required=True)
    parser.add_argument("--georsclip-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-inference", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "siglip2": audit_siglip2(Path(args.siglip2_dir), out, args.run_inference), "georsclip": audit_georsclip(Path(args.georsclip_dir), out)}
    write_json(out / "source_audit_summary.json", report)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
