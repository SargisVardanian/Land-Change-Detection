from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import train_unichange_v2_retrieval as base
from ucv2_cluster_common import build_model, run_metadata, strict_device


def run(data_root: Path, output_dir: Path, universat_source: Path, universat_checkpoint: Path, jina_model: Path) -> int:
    device = strict_device("cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = base.RetrievalConfig(
        data_root=str(data_root), output_dir=str(output_dir),
        universat_source=str(universat_source), universat_checkpoint=str(universat_checkpoint),
        jina_model=str(jina_model), image_size=256, output_grid=32,
        batch_size=2, max_train_samples=64, max_val_samples=2,
        max_steps=2, epochs=1, num_workers=0, device="cuda", use_bf16=True,
    )
    train, _ = base._build_datasets(cfg)
    model = build_model(cfg, device)
    optimizer = base._make_optimizer(model, cfg)
    rows = []
    for batch_size in (2, 4, 8, 16):
        loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=base._collate_temporal, drop_last=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        success = True; error = ""; shapes = {}
        try:
            for index, batch in enumerate(loader):
                if index == 2: break
                batch = base._move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with base._amp_context(device, True):
                    out = model(batch["images"], batch["captions"], batch["caption_to_pair"], batch["temporal_valid_mask"])
                    loss = base.multi_positive_symmetric_info_nce(out.pair_embedding, out.text_embedding, batch["caption_to_pair"])
                loss.backward(); optimizer.step()
                shapes = {"images": list(batch["images"].shape), "captions": len(batch["captions"])}
        except torch.cuda.OutOfMemoryError as exc:
            success = False; error = str(exc); optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        rows.append({"batch_size": batch_size, "success": success, "error": error, "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device), "shapes": shapes})
    ok = [row["batch_size"] for row in rows if row["success"]]
    report = {**run_metadata(), "status": "PASS" if ok else "FAIL", "gpu_name": torch.cuda.get_device_name(device), "results": rows, "recommended_batch_size": max(ok) if ok else None}
    (output_dir / "memory_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if ok else 1
