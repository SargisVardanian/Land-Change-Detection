#!/usr/bin/env python3
"""Create a stratified visual-review packet for automated-verified RSCC captions."""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
from typing import Any
from PIL import Image, ImageDraw, ImageFont

def font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf","/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path,size)
    return ImageFont.load_default()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--per-event",type=int,default=3)
    args=ap.parse_args()
    rows=[json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups=defaultdict(list)
    for row in rows: groups[str(row.get("source_event_id") or "unknown")].append(row)
    selected=[]
    for event in sorted(groups):
        selected.extend(sorted(groups[event],key=lambda r:(-float(r.get("verification_score",0.0)),str(r["canonical_pair_id"])))[:args.per_event])
    selected.sort(key=lambda r:(str(r.get("source_event_id")),str(r["canonical_pair_id"])))
    out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    packet_path = out / "review_packet.jsonl"
    packet_path.write_text("".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n" for r in selected),encoding="utf-8")
    W,H=720,470
    sheets=[]
    for sheet_start in range(0,len(selected),4):
        chunk=selected[sheet_start:sheet_start+4]
        sheet=Image.new("RGB",(W*2,H*2),"white")
        draw=ImageDraw.Draw(sheet)
        for local,row in enumerate(chunk):
            x=(local%2)*W; y=(local//2)*H
            with Image.open(row["t1_path"]) as a: first=a.convert("RGB").resize((350,300),Image.Resampling.BICUBIC)
            with Image.open(row["t2_path"]) as b: second=b.convert("RGB").resize((350,300),Image.Resampling.BICUBIC)
            sheet.paste(first,(x,y)); sheet.paste(second,(x+355,y))
            caption=" ".join(str((row.get("captions") or [""])[0]).split())
            label=f'{row.get("source_event_id")} | score={float(row.get("verification_score",0.0)):.4f}\n{caption}'
            draw.text((x,y+305),label,fill="black",font=font(13),spacing=2)
        path=out/f"review_sheet_{sheet_start//4:02d}.png"
        sheet.save(path)
        sheets.append(str(path))
    report={
        "schema_version":"qcpr-stage2-rscc-visual-review-packet-v2",
        "rows":len(selected),
        "events":len(groups),
        "per_event":args.per_event,
        "sheets":sheets,
        "reviewer_status":"pending_independent_human_review",
        "automated_source":"siglip2-base-patch16-256",
        "human_audit_passed":False,
        "source_candidates_sha256":sha256(args.input),
        "review_packet_sha256":sha256(packet_path),
        "event_ids_provenance_only":True,
        "semantic_positive_sets_materialized":False,
    }
    (out/"review_packet_audit.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    (out/"README.md").write_text(
        "# RSCC Stage-2 visual review packet\n\n"
        "This is a visual convenience packet for two independent human reviewers. "
        "It is not a completed audit and does not create semantic positives. Inspect "
        "both T1 and T2 in every sheet, then record decisions only in the formal "
        "reviewer_a_decisions.jsonl and reviewer_b_decisions.jsonl files. Event IDs "
        "are provenance/split fields only. Codex and automated model inspection do "
        "not count as human review.\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows":len(selected),"events":len(groups),"sheets":len(sheets),"output_dir":str(out)},sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
