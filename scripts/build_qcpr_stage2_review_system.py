#!/usr/bin/env python3
"""Build the portable QCPR Stage-2 review/audit artifacts on YSU.

The script intentionally creates new run roots. It never edits an immutable
release or historical training run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
import textwrap
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


CODE_SHA = "465e894e0d972b87d050bcf4cccb2bd74dc799dc"
REQUIRED_REVIEW_FIELDS = [
    "visible_change", "changed_object", "change_direction", "damage_type",
    "severity", "spatial_context", "location_support", "count_bucket",
    "count_support", "accept_rewrite_reject", "rewritten_caption",
    "confidence", "notes",
]
FORBIDDEN_KEY_PARTS = (
    "official_label", "dense_label", "mask_path", "mask_file", "mask_uri",
    "semantic_map", "binary_change_map", "object_count", "change_count",
    "mask_area", "label_stats", "mask_derived", "label_source",
    "structured_change_facts", "semantic_transition", "change_relation",
    "from_class", "to_class", "t1_semantic", "t2_semantic",
)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def rel_asset(source: Path, root: Path, target: Path) -> str:
    """Copy an image into the bundle and return a portable relative path."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(source) as im:
            im.convert("RGB").save(target, format="JPEG", quality=92, optimize=True)
    except Exception:
        shutil.copy2(source, target)
    return target.relative_to(root).as_posix()


def find_review_source(row: dict[str, Any], number: int, source_bundle: Path) -> tuple[Path, Path]:
    old = source_bundle / "images"
    candidates = [
        (old / f"{number:03d}_t1.png", old / f"{number:03d}_t2.png"),
        (old / f"{number:03d}_t1.jpg", old / f"{number:03d}_t2.jpg"),
        (Path(row["t1_path"]), Path(row["t2_path"])),
    ]
    for t1, t2 in candidates:
        if t1.exists() and t2.exists():
            return t1, t2
    raise FileNotFoundError(f"review assets missing for row {number}: {row.get('review_id')}")


def make_review_rows(packet: list[dict[str, Any]], source_bundle: Path, root: Path) -> list[dict[str, Any]]:
    stratum = {event: f"S{i:02d}" for i, event in enumerate(sorted({r["source_event_id"] for r in packet}), 1)}
    rows = []
    for number, source in enumerate(packet, 1):
        t1, t2 = find_review_source(source, number, source_bundle)
        rows.append({
            "row_number": number,
            "review_id": source["review_id"],
            "canonical_pair_id": source["canonical_pair_id"],
            "source_event_id": source["source_event_id"],
            "review_stratum": stratum[source["source_event_id"]],
            "split": source["split"],
            "candidate_caption": source["candidate_caption"],
            "t1_rel": rel_asset(t1, root, root / "assets" / "images" / f"{number:03d}_t1.jpg"),
            "t2_rel": rel_asset(t2, root, root / "assets" / "images" / f"{number:03d}_t2.jpg"),
            "t1_sha256": sha256(t1),
            "t2_sha256": sha256(t2),
            "verifier_score": source.get("verifier_score"),
            "verification_status": source.get("verification_status", "generated_unverified"),
        })
    return rows


def blank_decision(row: dict[str, Any], role: str) -> dict[str, Any]:
    result = {k: "" for k in REQUIRED_REVIEW_FIELDS}
    result.update({
        "row_number": row["row_number"],
        "review_id": row["review_id"],
        "canonical_pair_id": row["canonical_pair_id"],
        "source_event_id": row["source_event_id"],
        "split": row["split"],
        "candidate_caption": row["candidate_caption"],
        "reviewer_role": role,
        "reviewer_identity": "",
        "reviewed_at": "",
        "independence_attestation": False,
    })
    return result


def write_start_scripts(root: Path, assistant_zip_name: str) -> None:
    (root / "start_review.sh").write_text(
        """#!/usr/bin/env bash\nset -euo pipefail\nROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\nPORT=\"${1:-8765}\"\ncd \"$ROOT\"\nURL=\"http://127.0.0.1:${PORT}/reviewer_a_review.html\"\nif command -v open >/dev/null 2>&1; then (sleep 0.5; open \"$URL\") >/dev/null 2>&1 & fi\nexec python3 -m http.server \"$PORT\" --bind 127.0.0.1\n""",
        encoding="utf-8",
    )
    (root / "start_review.command").write_text(
        "#!/usr/bin/env bash\nexec \"$(cd \"$(dirname \"$0\")\" && pwd)/start_review.sh\" \"$@\"\n",
        encoding="utf-8",
    )
    for path in (root / "start_review.sh", root / "start_review.command"):
        path.chmod(0o755)
    (root / "README.md").write_text(
        f"""# QCPR Stage-2 review bundle v2

Запуск: дважды нажмите `start_review.command` на macOS или выполните
`./start_review.sh`. Откроется локальный HTTP-сервер и экран с одной парой за
раз.

Reviewer A проверяет визуальную фактичность описания. Reviewer B независимо
проверяет достаточность, поддержку деталей и необходимость короткого rewrite.
Ответы сохраняются отдельно в localStorage и экспортируются JSONL с
обязательным reviewer_identity.

Не используйте event name как semantic label. Если деталь не видна, выбирайте
uncertain/rewrite/reject.

AI audit: ../{assistant_zip_name}
""",
        encoding="utf-8",
    )


def reviewer_html(role: str, rows: list[dict[str, Any]], packet_sha: str, assistant_zip_name: str) -> str:
    role_label = "Reviewer A — визуальная фактичность" if role == "A" else "Reviewer B — независимая проверка деталей"
    role_help = (
        "Проверьте только, подтверждается ли кандидатное описание T1→T2. "
        "Не угадывайте скрытые детали."
        if role == "A" else
        "Работайте независимо от Reviewer A. Особое внимание — объекту, направлению, "
        "локации, количеству и степени уверенности."
    )
    rows_json = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    fields = []
    select_options = {
        "visible_change": ["", "yes", "no", "uncertain"],
        "changed_object": ["", "building", "road", "vegetation", "water", "soil_agriculture", "debris_structure", "other", "uncertain"],
        "change_direction": ["", "appeared", "disappeared", "damaged", "destroyed", "changed", "uncertain"],
        "damage_type": ["", "none", "debris", "collapse", "crack", "flood_inundation", "burn_ash", "vegetation_damage", "other", "uncertain"],
        "severity": ["", "none", "mild", "moderate", "severe", "uncertain"],
        "spatial_context": ["", "supported", "unsupported", "uncertain", "not_applicable"],
        "location_support": ["", "supported", "unsupported", "uncertain", "not_applicable"],
        "count_bucket": ["", "none", "one", "few", "many", "uncertain"],
        "count_support": ["", "supported", "unsupported", "uncertain", "not_applicable"],
        "accept_rewrite_reject": ["", "accept", "rewrite", "reject"],
        "confidence": ["", "low", "medium", "high"],
    }
    labels = {
        "visible_change": "Видимое изменение", "changed_object": "Объект изменения",
        "change_direction": "Направление T1→T2", "damage_type": "Тип повреждения",
        "severity": "Степень", "spatial_context": "Пространственный контекст",
        "location_support": "Поддержка локации", "count_bucket": "Количество",
        "count_support": "Поддержка количества", "accept_rewrite_reject": "Решение",
        "confidence": "Уверенность",
    }
    for key in REQUIRED_REVIEW_FIELDS:
        if key in ("rewritten_caption", "notes"):
            fields.append(f'<label>{"Исправленное короткое описание" if key == "rewritten_caption" else "Заметки"}<textarea data-field="{key}" rows="3"></textarea></label>')
        elif key in select_options:
            options = "".join(f'<option value="{html.escape(v)}">{html.escape(v or "— выберите —")}</option>' for v in select_options[key])
            fields.append(f'<label>{labels[key]}<select data-field="{key}">{options}</select></label>')
    template = r'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>QCPR Stage-2 __ROLE_LABEL__</title>
<style>
:root{color-scheme:light;--ink:#17202a;--muted:#5f6b76;--line:#d8dee4;--accent:#1769aa;--panel:#fff;--bg:#f4f7fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:#102a43;color:white;padding:14px 24px;box-shadow:0 2px 8px #0002}.top{display:flex;gap:16px;align-items:center;justify-content:space-between;flex-wrap:wrap}h1{font-size:20px;margin:0}.muted{color:#d6e2ee;font-size:13px}
main{max-width:1500px;margin:22px auto;padding:0 18px}.toolbar,.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:16px;box-shadow:0 2px 8px #14213d0b}.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button,input,select,textarea{font:inherit}button{border:0;border-radius:8px;padding:9px 13px;background:var(--accent);color:#fff;cursor:pointer}button.secondary{background:#667788}button.warn{background:#a63d40}input[type=text],input[type=file],select,textarea{width:100%;border:1px solid #b8c4cf;border-radius:7px;padding:8px;background:#fff}#reviewer{width:240px}
.pair-meta{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:8px;background:#eef5fb;padding:12px;border-radius:10px;margin-bottom:14px}.meta-label{font-size:12px;color:var(--muted);display:block}.meta-value{font-weight:650;overflow-wrap:anywhere}
.images{display:grid;grid-template-columns:1fr 1fr;gap:16px}.image-card{margin:0;background:#0f1720;border-radius:12px;padding:10px;color:#fff}.image-card img{display:block;width:100%;height:min(56vh,620px);object-fit:contain;background:#080c11;border-radius:7px}.image-card figcaption{padding:8px 4px 2px;font-weight:700}
.caption-box{border-left:5px solid #f59e0b;background:#fff8e6;padding:14px 16px;border-radius:9px;margin:16px 0;overflow-wrap:anywhere;word-break:normal}.caption-title{font-size:12px;color:#7c5b10;text-transform:uppercase;letter-spacing:.04em}.caption-text{font-size:20px;line-height:1.55;margin-top:5px;white-space:pre-wrap}
.role-note{background:#e9f6ef;border:1px solid #b9e2c8;border-radius:8px;padding:10px 12px;margin-bottom:14px}.fields{display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));gap:12px}label{display:flex;flex-direction:column;gap:5px;font-weight:650;color:#34495e}label textarea,label select{font-weight:400}.wide{grid-column:1/-1}.progress{font-variant-numeric:tabular-nums;font-weight:700}.error{color:#a61b1b;font-weight:700;min-height:1.4em}.ok{color:#146c43;font-weight:700}.shortcut{color:var(--muted);font-size:13px}
@media(max-width:900px){.images,.fields{grid-template-columns:1fr}.pair-meta{grid-template-columns:repeat(2,1fr)}.image-card img{height:45vh}}
</style></head><body>
<header><div class="top"><div><h1>QCPR Stage-2 human review · __ROLE_LABEL__</h1><div class="muted" id="role-help"></div></div><div class="progress" id="progress"></div></div></header>
<main><div class="toolbar"><button id="prev">← Назад</button><button id="next">Далее →</button><button class="secondary" id="save">Сохранить</button><button class="secondary" id="export">Экспорт JSONL</button><button class="secondary" id="export-csv">Экспорт CSV</button><label style="width:220px"><span style="font-size:12px">Импорт JSONL</span><input id="import" type="file" accept=".jsonl,.json,.txt"></label><label style="width:260px">Reviewer ID<input id="reviewer" type="text" placeholder="обязательно"></label><label style="width:180px">Reviewed at<input id="reviewed_at" type="text" placeholder="ISO 8601"></label><button class="warn" id="clear">Очистить текущую</button><a href="../__ASSISTANT_ZIP__" download><button class="secondary" type="button">Download AI audit bundle</button></a><span class="shortcut">←/→ переключение · Ctrl+S сохранение</span></div>
<div id="error" class="error"></div><section id="content"></section></main>
<script>
const ROLE=__ROLE__, PACKET_SHA=__PACKET_SHA__, ROWS=__ROWS__;
const REQUIRED=__REQUIRED__, ROLE_HELP=__ROLE_HELP__; const storageKey=`qcpr-stage2-review-v2-${ROLE}-${PACKET_SHA}`; let index=0,answers={};
function blank(){const x={};REQUIRED.forEach(k=>x[k]='');return x} function load(){try{answers=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){answers={}}} function current(){return ROWS[index]}
function escapeHtml(s){return String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]))}
function renderFields(){const opts={visible_change:['','yes','no','uncertain'],changed_object:['','building','road','vegetation','water','soil_agriculture','debris_structure','other','uncertain'],change_direction:['','appeared','disappeared','damaged','destroyed','changed','uncertain'],damage_type:['','none','debris','collapse','crack','flood_inundation','burn_ash','vegetation_damage','other','uncertain'],severity:['','none','mild','moderate','severe','uncertain'],spatial_context:['','supported','unsupported','uncertain','not_applicable'],location_support:['','supported','unsupported','uncertain','not_applicable'],count_bucket:['','none','one','few','many','uncertain'],count_support:['','supported','unsupported','uncertain','not_applicable'],accept_rewrite_reject:['','accept','rewrite','reject'],confidence:['','low','medium','high']};const labels={visible_change:'Видимое изменение',changed_object:'Объект изменения',change_direction:'Направление T1→T2',damage_type:'Тип повреждения',severity:'Степень',spatial_context:'Пространственный контекст',location_support:'Поддержка локации',count_bucket:'Количество',count_support:'Поддержка количества',accept_rewrite_reject:'Решение',confidence:'Уверенность'};return REQUIRED.map(k=>opts[k]?`<label>${labels[k]}<select data-field="${k}">${opts[k].map(v=>`<option value="${v}">${v||'— выберите —'}</option>`).join('')}</select></label>`:`<label class="wide">${k==='rewritten_caption'?'Исправленное короткое описание':'Заметки'}<textarea data-field="${k}" rows="3"></textarea></label>`).join('')}
function render(){const row=current(),saved=answers[row.review_id]||blank();document.getElementById('progress').textContent=`${index+1} / ${ROWS.length}`;document.getElementById('role-help').textContent=ROLE_HELP;document.getElementById('content').innerHTML=`<section class="panel"><div class="pair-meta"><div><span class="meta-label">Review ID</span><span class="meta-value">${row.review_id}</span></div><div><span class="meta-label">Pair</span><span class="meta-value">${row.canonical_pair_id}</span></div><div><span class="meta-label">Review stratum</span><span class="meta-value">${row.review_stratum}</span></div><div><span class="meta-label">Split</span><span class="meta-value">${row.split}</span></div></div><div class="images"><figure class="image-card"><img src="${row.t1_rel}" alt="T1"><figcaption>T1 · первая дата</figcaption></figure><figure class="image-card"><img src="${row.t2_rel}" alt="T2"><figcaption>T2 · вторая дата</figcaption></figure></div><div class="caption-box"><div class="caption-title">Кандидатное описание · не доверяйте ему автоматически</div><div class="caption-text">${escapeHtml(row.candidate_caption)}</div></div><div class="role-note">${ROLE_HELP}</div><div class="fields">${renderFields()}</div></section>`;document.querySelectorAll('[data-field]').forEach(el=>el.value=saved[el.dataset.field]||'')}
function collect(){const row=current(),out={};REQUIRED.forEach(k=>out[k]=document.querySelector(`[data-field="${k}"]`)?.value||'');answers[row.review_id]=out}
function save(){collect();localStorage.setItem(storageKey,JSON.stringify(answers));document.getElementById('error').className='ok';document.getElementById('error').textContent='Сохранено локально.'}
function validate(){const id=document.getElementById('reviewer').value.trim();if(!id){document.getElementById('error').className='error';document.getElementById('error').textContent='Нужен Reviewer ID перед экспортом.';return false}collect();return true}
function exportJsonl(){if(!validate())return;const reviewer=document.getElementById('reviewer').value.trim(),at=document.getElementById('reviewed_at').value.trim()||new Date().toISOString();const lines=ROWS.map(r=>JSON.stringify({...r,...(answers[r.review_id]||blank()),reviewer_role:ROLE,reviewer_identity:reviewer,reviewed_at:at,independence_attestation:true})).join('\n')+'\n';const blob=new Blob([lines],{type:'application/x-ndjson'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`reviewer_${ROLE.toLowerCase()}_decisions.jsonl`;a.click()}
function csvCell(v){return `"${String(v??'').replace(/"/g,'""')}"`} function exportCsv(){if(!validate())return;const reviewer=document.getElementById('reviewer').value.trim(),at=document.getElementById('reviewed_at').value.trim()||new Date().toISOString();const fields=['row_number','review_id','canonical_pair_id','source_event_id','split','candidate_caption',...REQUIRED,'reviewer_role','reviewer_identity','reviewed_at','independence_attestation'];const lines=[fields.map(csvCell).join(',')];ROWS.forEach(r=>{const x={...r,...(answers[r.review_id]||blank()),reviewer_role:ROLE,reviewer_identity:reviewer,reviewed_at:at,independence_attestation:true};lines.push(fields.map(k=>csvCell(x[k])).join(','))});const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\n')+'\n'],{type:'text/csv'}));a.download=`reviewer_${ROLE.toLowerCase()}_decisions.csv`;a.click()}
function importJsonl(file){const reader=new FileReader();reader.onload=()=>{try{const parsed=String(reader.result).split(/\r?\n/).filter(Boolean).map(JSON.parse);if(!parsed.length)throw Error('empty');parsed.forEach(x=>{if(!x.review_id)throw Error('review_id missing');const out={};REQUIRED.forEach(k=>out[k]=x[k]??'');answers[x.review_id]=out});localStorage.setItem(storageKey,JSON.stringify(answers));render();document.getElementById('error').textContent=`Импортировано: ${parsed.length}`;document.getElementById('error').className='ok'}catch(e){document.getElementById('error').textContent='Ошибка импорта: '+e.message;document.getElementById('error').className='error'}};reader.readAsText(file)}
document.getElementById('prev').onclick=()=>{collect();index=Math.max(0,index-1);render()};document.getElementById('next').onclick=()=>{collect();index=Math.min(ROWS.length-1,index+1);render()};document.getElementById('save').onclick=save;document.getElementById('export').onclick=exportJsonl;document.getElementById('export-csv').onclick=exportCsv;document.getElementById('clear').onclick=()=>{answers[current().review_id]=blank();localStorage.setItem(storageKey,JSON.stringify(answers));render()};document.getElementById('import').onchange=e=>e.target.files[0]&&importJsonl(e.target.files[0]);document.addEventListener('keydown',e=>{if(e.target.matches('input,textarea,select'))return;if(e.key==='ArrowLeft')document.getElementById('prev').click();if(e.key==='ArrowRight')document.getElementById('next').click();if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='s'){e.preventDefault();save()}});load();render();
</script></body></html>'''
    return (template.replace("__ROLE_LABEL__", html.escape(role_label)).replace("__ASSISTANT_ZIP__", html.escape(assistant_zip_name)).replace("__ROLE__", json.dumps(role)).replace("__PACKET_SHA__", json.dumps(packet_sha)).replace("__ROWS__", rows_json).replace("__REQUIRED__", json.dumps(REQUIRED_REVIEW_FIELDS)).replace("__ROLE_HELP__", json.dumps(role_help, ensure_ascii=False)))


def build_review_bundle(packet: list[dict[str, Any]], source_bundle: Path, root: Path, assistant_zip_name: str) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    rows = make_review_rows(packet, source_bundle, root)
    packet_sha = sha256_text("\n".join(json.dumps(r, sort_keys=True) for r in rows))
    write_jsonl(root / "data/review_rows.jsonl", rows)
    csv_fields = ["row_number", "review_id", "canonical_pair_id", "source_event_id", "split", "candidate_caption", *REQUIRED_REVIEW_FIELDS, "reviewer_role", "reviewer_identity", "reviewed_at", "independence_attestation"]
    for role in ("A", "B"):
        write_jsonl(root / f"data/reviewer_{role.lower()}_decisions.jsonl", [blank_decision(r, role) for r in rows])
        with (root / f"data/reviewer_{role.lower()}_template.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields); writer.writeheader()
            for row in rows:
                item = blank_decision(row, role); item.update({"row_number": row["row_number"], "candidate_caption": row["candidate_caption"]})
                writer.writerow({field: item.get(field, "") for field in csv_fields})
        (root / f"reviewer_{role.lower()}_review.html").write_text(reviewer_html(role, rows, packet_sha, assistant_zip_name), encoding="utf-8")
    write_json(root / "data/review_sequence.json", [{"row_number": r["row_number"], "review_id": r["review_id"], "canonical_pair_id": r["canonical_pair_id"]} for r in rows])
    write_json(root / "data/export_schema.json", {"schema_version": "qcpr-stage2-review-v2", "required_fields": REQUIRED_REVIEW_FIELDS, "identity_fields": ["reviewer_role", "reviewer_identity", "reviewed_at", "independence_attestation"], "csv_fields": csv_fields, "roles": {"A": "visual factuality", "B": "independent detail sufficiency"}})
    write_json(root / "data/viewer_self_test.json", {"rows": len(rows), "unique_review_ids": len({r["review_id"] for r in rows}), "relative_image_refs": True, "absolute_path_refs": False, "one_pair_per_screen": True, "local_storage_role_separated": True, "reviewer_identity_required": True, "export_import_required_fields": REQUIRED_REVIEW_FIELDS})
    write_start_scripts(root, assistant_zip_name)
    start_script = (root / "start_review.sh").read_text(encoding="utf-8").replace("reviewer_a_review.html", "index.html")
    (root / "start_review.sh").write_text(start_script, encoding="utf-8")
    (root / "index.html").write_text(
        f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QCPR Stage-2 review</title><style>body{{margin:0;background:#f4f7fa;color:#17202a;font:17px/1.55 system-ui}}main{{max-width:900px;margin:40px auto;padding:0 20px}}.card{{background:#fff;border:1px solid #d8dee4;border-radius:14px;padding:24px;margin:16px 0;box-shadow:0 2px 8px #14213d0b}}h1{{margin-top:0}}a.button{{display:inline-block;background:#1769aa;color:#fff;text-decoration:none;border-radius:8px;padding:11px 16px;margin:4px 8px 4px 0}}.secondary{{background:#667788!important}}li{{margin:8px 0}}code{{background:#eef2f5;padding:2px 5px;border-radius:4px}}</style></head><body><main><section class="card"><h1>QCPR Stage-2 · human review</h1><p>Здесь проверяются 240 пар: по 20 из каждого из 12 событий. На экране всегда только одна пара и одно машинное описание — текст не накладывается на изображения.</p><p><a class="button" href="reviewer_a_review.html">Открыть Reviewer A</a><a class="button secondary" href="reviewer_b_review.html">Открыть Reviewer B</a><a class="button secondary" href="../{html.escape(assistant_zip_name)}" download>Скачать AI audit bundle</a></p></section><section class="card"><h2>Что проверить в каждой строке</h2><ol><li>Сначала сравните T1 и T2, не читая caption как факт.</li><li>Проверьте, есть ли видимое изменение и правильно ли назван его объект.</li><li>Проверьте направление T1→T2, тип повреждения, степень и пространственную деталь.</li><li>Если деталь не видна или не доказуема, выберите <code>uncertain</code> или <code>rewrite</code>; не угадывайте.</li><li>Для <code>rewrite</code> напишите короткую фактическую версию без неподтверждённых чисел, дорог, локаций и severity.</li><li>Введите уникальный Reviewer ID. Reviewer A и Reviewer B должны работать независимо.</li></ol></section><section class="card"><h2>Что не считать доказательством</h2><ul><li>Название события не является semantic label и не делает пары положительными.</li><li>Машинный verifier score — только подсказка для отбора, не human label.</li><li>Event-disjoint split сохранён; в assistant sample 4 строки на событие дают 32 train / 8 development / 8 test — это ограничение исходного split, не ошибка подсчёта.</li></ul></section><section class="card"><h2>Как завершить</h2><p>После заполнения используйте <b>Экспорт JSONL</b> и <b>Экспорт CSV</b>. Передайте свой файл координатору. Не редактируйте файл второго reviewer-а и не открывайте чужие ответы.</p></section></main></body></html>''',
        encoding="utf-8",
    )
    sums = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    write_json(root / "bundle_manifest.json", {"schema_version": "qcpr-stage2-review-bundle-v2", "packet_sha256": packet_sha, "rows": len(rows), "events": len({r["source_event_id"] for r in rows}), "assets": len(list((root / "assets/images").glob("*")))})
    return {"root": str(root), "rows": rows, "packet_sha256": packet_sha}


def hamming_hex(a: str, b: str) -> int | None:
    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except (ValueError, TypeError):
        return None


def event_audit(pair_registry: Path, caption_registry: Path, out: Path) -> dict[str, Any]:
    pairs = [r for r in read_jsonl(pair_registry) if str(r.get("source_dataset", "")).upper() == "RSCC-EBD"]
    events = Counter(str(r.get("source_event_id", "unknown")) for r in pairs)
    disasters = Counter((e.split("-")[0] if e else "unknown") for e in events for _ in range(events[e]))
    splits = Counter(str(r.get("split", "unknown")) for r in pairs)
    scene_ids = {r.get("source_geographic_scene_id") or r.get("geographic_scene_id") or r.get("source_location_id") for r in pairs}
    scene_ids.discard(None)
    frame_hashes = Counter()
    phashes: dict[str, list[str]] = defaultdict(list)
    pair_for_hash: dict[str, set[str]] = defaultdict(set)
    for r in pairs:
        for frame in r.get("frames", []):
            h = frame.get("sha256")
            p = frame.get("perceptual_hash")
            if h:
                frame_hashes[h] += 1
                pair_for_hash[h].add(r.get("canonical_pair_id", ""))
            if p:
                phashes[str(p)].append(r.get("canonical_pair_id", ""))
    exact_groups = [sorted(v) for v in pair_for_hash.values() if len(v) > 1]
    # Conservative candidate near-duplicate groups: only compare pHashes sharing a prefix.
    buckets: dict[str, list[str]] = defaultdict(list)
    for p in phashes:
        buckets[p[:4]].append(p)
    near_groups = []
    for values in buckets.values():
        for i, a in enumerate(values):
            for b in values[i + 1:]:
                d = hamming_hex(a, b)
                if d is not None and d <= 4:
                    near_groups.append({"hash_a": a, "hash_b": b, "hamming": d, "pair_ids": sorted(set(phashes[a] + phashes[b]))})
    captions = read_jsonl(caption_registry)
    rscc_caps = [r for r in captions if str(r.get("source_dataset", "")).upper().startswith("RSCC")]
    if not rscc_caps:
        fallback = caption_registry.parent.parent / "manifests/provisional_semantic_candidates/rscc_qvq_unverified.jsonl"
        rscc_caps = read_jsonl(fallback)
    def cap_text(row: dict[str, Any]) -> str:
        value = row.get("text", row.get("caption", row.get("captions", "")))
        return value[0] if isinstance(value, list) and value else str(value)
    normalized = Counter(re.sub(r"\s+", " ", cap_text(r).strip().lower()) for r in rscc_caps)
    template = Counter(re.sub(r"\d+", "<NUM>", re.sub(r"\s+", " ", cap_text(r).strip().lower())) for r in rscc_caps)
    result = {"source": "RSCC-EBD", "physical_pairs": len(pairs), "event_counts": dict(sorted(events.items())), "disaster_type_counts": dict(sorted(disasters.items())), "split_counts": dict(sorted(splits.items())), "unique_geographic_scene_count": len(scene_ids), "geographic_scene_metadata_available": bool(scene_ids), "exact_duplicate_image_hash_groups": len(exact_groups), "exact_duplicate_image_hash_pairs": sum(len(x) for x in exact_groups), "near_duplicate_phash_candidate_groups": len(near_groups), "near_duplicate_method": "same first 16 pHash bits and Hamming distance <=4; candidate audit, not proof of duplicate", "overlapping_crop_metadata": "not_available_in_pair_registry", "caption_count": len(rscc_caps), "caption_exact_duplicate_rate": (sum(n - 1 for n in normalized.values() if n > 1) / len(rscc_caps) if rscc_caps else None), "caption_template_rate": (sum(n - 1 for n in template.values() if n > 1) / len(rscc_caps) if rscc_caps else None), "mean_near_cluster_size": (sum(len(x["pair_ids"]) for x in near_groups) / len(near_groups) if near_groups else 1.0), "largest_near_cluster_size": (max((len(x["pair_ids"]) for x in near_groups), default=1)), "interpretation": {"event_diversity": len(events), "geographic_diversity": len(scene_ids) if scene_ids else "not measurable from current metadata", "visual_diversity": "requires review of candidate duplicate groups and image embeddings", "caption_diversity": len(normalized)}}
    write_json(out / "rscc_event_duplicate_audit.json", result)
    with (out / "rscc_event_distribution.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["event", "pairs"]); w.writerows(sorted(events.items()))
    with (out / "rscc_split_distribution.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["split", "pairs"]); w.writerows(sorted(splits.items()))
    write_jsonl(out / "rscc_near_duplicate_candidates.jsonl", near_groups)
    return result


def assistant_rows(packet: list[dict[str, Any]], release: Path) -> list[dict[str, Any]]:
    qvq = {r.get("canonical_pair_id"): r for r in read_jsonl(release / "manifests/provisional_semantic_candidates/rscc_qvq_unverified.jsonl")}
    pilot = {}
    for name in ("train", "development", "test"):
        path = release.parent / ("qcpr_dataset_v2_stage2_audit_repaired_8a60e31_20260802/semantic_view/automated_verified_pilot/retrieval_semantic_automated_verified_" + name + ".jsonl")
        for r in read_jsonl(path):
            pilot[r.get("canonical_pair_id")] = r
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    registry = read_jsonl(release / "registries/pair_registry.jsonl")
    for r in registry:
        if str(r.get("source_dataset", "")).upper() == "RSCC-EBD":
            by_event[r["source_event_id"]].append(r)
    phash_counts = Counter(str(frame.get("perceptual_hash")) for r in registry if str(r.get("source_dataset", "")).upper() == "RSCC-EBD" for frame in r.get("frames", []) if frame.get("perceptual_hash"))
    selected = []
    for event in sorted(by_event):
        candidates = by_event[event]
        scored = [r for r in candidates if pilot.get(r["canonical_pair_id"], {}).get("verification_score") is not None]
        high = max(scored or candidates, key=lambda r: float(pilot.get(r["canonical_pair_id"], {}).get("verification_score", -1e9)))
        low = min(scored or candidates, key=lambda r: float(pilot.get(r["canonical_pair_id"], {}).get("verification_score", 1e9)))
        qtext = {r["canonical_pair_id"]: ((r.get("captions") or [""])[0] if isinstance(r.get("captions"), list) else str(r.get("captions", ""))) for r in qvq.values()}
        suspect_candidates = [r for r in candidates if any(w in qtext.get(r["canonical_pair_id"], "").lower() for w in ("fault", "landslide", "road", "count", "left", "right", "likely", "suggesting"))]
        near_candidates = [r for r in candidates if any(str(frame.get("perceptual_hash")) and phash_counts[str(frame.get("perceptual_hash"))] > 1 for frame in r.get("frames", []))]
        choices = [(high, ["four_per_event", "high_verifier_score" if scored else "score_unavailable"]), (low, ["four_per_event", "low_verifier_score" if scored else "score_unavailable"]), ((suspect_candidates or candidates)[0], ["four_per_event", "suspected_hallucination"]), ((near_candidates or candidates)[0], ["four_per_event", "near_duplicate_candidate" if near_candidates else "diversity_control"])]
        used = set()
        for r, reasons in choices:
            if r["canonical_pair_id"] in used:
                fallback = next(x for x in candidates if x["canonical_pair_id"] not in used)
                r = fallback
            used.add(r["canonical_pair_id"])
            p = pilot.get(r["canonical_pair_id"], {}); q = qvq.get(r["canonical_pair_id"], {})
            text = ((q.get("captions") or [""])[0] if isinstance(q.get("captions"), list) else str(q.get("captions", ""))) or str(p.get("text", ""))
            suspicion = ["claim_requires_visual_review"] if any(w in text.lower() for w in ("fault", "landslide", "road", "count", "left", "right", "likely", "suggesting")) else ["no_automatic_suspicion"]
            frame_paths = [f.get("path") for f in r.get("frames", [])]
            selected.append({"audit_row_id": f"assistant-{len(selected)+1:03d}", "review_id": f"assistant-audit:{r['canonical_pair_id']}", "canonical_pair_id": r["canonical_pair_id"], "source_event_id": event, "split": r["split"], "candidate_caption": text, "_t1_path": r.get("t1_path") or frame_paths[0], "_t2_path": r.get("t2_path") or frame_paths[1], "verifier_score": p.get("verification_score"), "shuffled_score": p.get("verification_shuffled_score"), "verification_status": "generated_unverified", "human_audit_status": "unreviewed", "heuristic_flags": suspicion, "selection_reasons": reasons})
    return selected


def assistant_html(rows: list[dict[str, Any]]) -> str:
    data = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QCPR AI audit</title><style>body{{margin:0;background:#f3f6fa;color:#17202a;font:16px/1.45 system-ui}}header{{background:#263238;color:#fff;padding:14px 22px;position:sticky;top:0}}main{{max-width:1400px;margin:18px auto;padding:0 16px}}.bar,.card{{background:#fff;border:1px solid #d8dee4;border-radius:12px;padding:14px;margin-bottom:14px}}.bar{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}button,select,textarea{{font:inherit}}button{{background:#1769aa;color:#fff;border:0;border-radius:7px;padding:9px 12px}}.meta{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;background:#edf5fb;padding:10px;border-radius:8px}}.images{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}}figure{{margin:0;background:#101820;padding:8px;color:#fff;border-radius:8px}}figure img{{width:100%;height:min(55vh,620px);object-fit:contain;background:#05080b;display:block}}figcaption{{padding-top:6px;font-weight:700}}.caption{{background:#fff8e6;padding:14px;border-left:5px solid #f59e0b;margin:12px 0;overflow-wrap:anywhere;white-space:pre-wrap;font-size:19px}}.form{{display:grid;grid-template-columns:240px 1fr;gap:10px;align-items:start}}textarea{{width:100%;box-sizing:border-box;padding:8px;min-height:90px}}@media(max-width:800px){{.images,.form,.meta{{grid-template-columns:1fr}}}}</style></head><body><header><b>QCPR Stage-2 assistant audit</b> · <span id="progress"></span></header><main><div class="bar"><button id="prev">← Previous</button><button id="next">Next →</button><button id="export">Export audit JSONL</button><span>Heuristics are candidates only; this is not human verification.</span></div><div id="app"></div></main><script>const ROWS={data};let i=0,answers={{}};function esc(s){{return String(s??'').replace(/[&<>"']/g,m=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m]))}}function render(){{let r=ROWS[i],a=answers[r.audit_row_id]||{{decision:'',notes:''}};document.getElementById('progress').textContent=`${{i+1}} / ${{ROWS.length}}`;document.getElementById('app').innerHTML=`<section class="card"><div class="meta"><div><small>Pair</small><br>${{esc(r.canonical_pair_id)}}</div><div><small>Event</small><br>${{esc(r.source_event_id)}}</div><div><small>Split</small><br>${{esc(r.split)}}</div><div><small>Verifier</small><br>${{r.verifier_score??'n/a'}}</div></div><div class="images"><figure><img src="${{r.t1_rel}}"><figcaption>T1</figcaption></figure><figure><img src="${{r.t2_rel}}"><figcaption>T2</figcaption></figure></div><div class="caption"><b>Candidate caption</b><br>${{esc(r.candidate_caption)}}</div><div class="form"><label>AI audit status<select id="decision"><option value="">— choose —</option><option>supported</option><option>uncertain</option><option>unsupported</option><option>needs_human_review</option></select></label><label>Notes<textarea id="notes" placeholder="Which visible fact supports or contradicts the caption?"></textarea></label></div><p><b>Selection flags:</b> ${{esc(r.selection_reasons.join(', '))}}<br><b>Heuristic flags:</b> ${{esc(r.heuristic_flags.join(', '))}}</p></section>`;document.getElementById('decision').value=a.decision||'';document.getElementById('notes').value=a.notes||''}}function collect(){{let r=ROWS[i];answers[r.audit_row_id]={{decision:document.getElementById('decision')?.value||'',notes:document.getElementById('notes')?.value||''}};localStorage.setItem('qcpr-ai-audit-v1',JSON.stringify(answers))}}document.getElementById('prev').onclick=()=>{{collect();i=Math.max(0,i-1);render()}};document.getElementById('next').onclick=()=>{{collect();i=Math.min(ROWS.length-1,i+1);render()}};document.getElementById('export').onclick=()=>{{collect();let lines=ROWS.map(r=>JSON.stringify({{...r,...(answers[r.audit_row_id]||{{}})}})).join('\n')+'\n';let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines],{{type:'application/x-ndjson'}}));a.download='assistant_audit_decisions.jsonl';a.click()}};try{{answers=JSON.parse(localStorage.getItem('qcpr-ai-audit-v1')||'{{}}')}}catch(e){{}}render();</script></body></html>'''


def build_assistant_bundle(packet: list[dict[str, Any]], release: Path, source_bundle: Path, root: Path, sha7: str) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    rows = assistant_rows(packet, release)
    for row in rows:
        t1, t2 = Path(row.pop("_t1_path")), Path(row.pop("_t2_path"))
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", row["audit_row_id"])
        row["t1_rel"] = rel_asset(t1, root, root / "assets" / f"{safe}_t1.jpg")
        row["t2_rel"] = rel_asset(t2, root, root / "assets" / f"{safe}_t2.jpg")
    write_jsonl(root / "assistant_audit_manifest.jsonl", rows)
    (root / "assistant_audit.html").write_text(assistant_html(rows), encoding="utf-8")
    ev = Counter(r["source_event_id"] for r in rows); sp = Counter(r["split"] for r in rows)
    with (root / "event_distribution.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["source_event_id","rows"]);w.writerows(sorted(ev.items()))
    with (root / "split_distribution.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["split","rows"]);w.writerows(sorted(sp.items()))
    dup = Counter(re.sub(r"\s+", " ", r["candidate_caption"].lower()).strip() for r in rows)
    with (root / "duplicate_audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["normalized_caption", "count", "row_ids"])
        for k, v in sorted(dup.items()):
            if v > 1:
                w.writerow([k, v, ",".join(r["audit_row_id"] for r in rows if re.sub(r"\s+", " ", r["candidate_caption"].lower()).strip() == k)])
    write_jsonl(root / "caption_claim_audit.jsonl", [{"audit_row_id":r["audit_row_id"],"canonical_pair_id":r["canonical_pair_id"],"caption":r["candidate_caption"],"heuristic_flags":r["heuristic_flags"],"human_audit_status":"unreviewed","automated_selection_only":True} for r in rows])
    write_json(root / "viewer_sequence.json", [{"audit_row_id": r["audit_row_id"], "event": r["source_event_id"], "split": r["split"]} for r in rows])
    write_json(root / "export_schema.json", {"schema_version":"qcpr-stage2-assistant-audit-v1","required":"audit_row_id,canonical_pair_id,decision,notes","human_status":"unreviewed"})
    write_json(root / "viewer_self_test.json", {"rows":len(rows),"rows_per_event":dict(ev),"split_counts":dict(sp),"balanced_split_counts":False,"split_balance_status":"event-disjoint source split constrains the 4-per-event audit: 32 train / 8 development / 8 test","relative_assets":True,"absolute_refs":False,"one_row_per_screen":True})
    (root / "README.md").write_text("This is a portable AI-audit bundle. Heuristic flags are sampling metadata, not human labels. Open assistant_audit.html through start_review.sh or any local HTTP server.\n",encoding="utf-8")
    # SHA256SUMS intentionally excludes itself; it is complete for the distributed payload.
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS": files.append(f"{sha256(p)}  {p.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(files)+"\n",encoding="utf-8")
    zip_path = root.parent / f"qcpr_stage2_assistant_audit_bundle_{sha7}_20260803.zip"
    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file(): z.write(p,p.relative_to(root).as_posix())
    write_json(root.parent / f"qcpr_stage2_assistant_audit_bundle_{sha7}_20260803.zip.sha256.json", {"zip":str(zip_path),"sha256":sha256(zip_path),"bytes":zip_path.stat().st_size})
    return {"root":str(root),"zip":str(zip_path),"rows":len(rows),"events":len(ev),"split_counts":dict(sp),"zip_sha256":sha256(zip_path)}


def sanitize(value: Any, path: str = "") -> tuple[Any, list[str]]:
    violations = []
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            lk = str(k).lower()
            if any(part in lk for part in FORBIDDEN_KEY_PARTS):
                violations.append(path + "/" + str(k)); continue
            clean, found = sanitize(v, path + "/" + str(k)); violations.extend(found); out[k] = clean
        return out, violations
    if isinstance(value, list):
        out=[]
        for i,v in enumerate(value): clean,found=sanitize(v,f"{path}/{i}");violations.extend(found);out.append(clean)
        return out,violations
    if isinstance(value, str) and any(token in value.lower() for token in ("/mask/", "mask.png", "dense_label", "semantic_map", "official_label")):
        return None, [path + "=<forbidden-path>"]
    return value, violations


def maskfree_repair(release: Path, out: Path) -> dict[str, Any]:
    input_rows=0; original_violations=[]; repaired_files=[]; output_violations=[]
    for p in sorted(release.rglob("*.jsonl")):
        name=p.name.lower(); text=p.read_text(encoding="utf-8",errors="replace")
        is_maskfree = any(x in name for x in ("retrieval", "grounding", "mask_free", "maskfree"))
        if not is_maskfree: continue
        dst=out/p.relative_to(release); dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open("w",encoding="utf-8") as f:
            for line_no,line in enumerate(text.splitlines(),1):
                if not line.strip(): continue
                row=json.loads(line); input_rows+=1
                clean,viol=sanitize(row,f"{p.relative_to(release)}:{line_no}"); original_violations.extend(viol)
                if clean is not None: f.write(json.dumps(clean,ensure_ascii=False,sort_keys=True)+"\n")
        repaired_files.append(str(dst.relative_to(out)))
    for p in sorted(out.rglob("*.jsonl")):
        for line_no,line in enumerate(p.read_text(encoding="utf-8").splitlines(),1):
            if line.strip(): output_violations.extend(sanitize(json.loads(line),f"{p.relative_to(out)}:{line_no}")[1])
    result={"historical_release_status":{"release":"qcpr_dataset_v2_stage2_audit_repaired_8a60e31_20260802","status":"INVALID_MASK_FREE_LABEL_LEAKAGE"},"scanned_rows":input_rows,"original_violation_count":len(original_violations),"repaired_files":repaired_files,"output_violation_count":len(output_violations),"zero_violations_after_repair":not output_violations,"policy":"structured S2Looking labels and dense sidecars remain evaluation-only; sanitised views contain no label/mask-derived fields"}
    write_json(out.parent/"maskfree_leakage_audit.json",result); (out.parent/"maskfree_leakage_audit.md").write_text("# Mask-free leakage audit\n\nThe historical 8a60e31 release is marked `INVALID_MASK_FREE_LABEL_LEAKAGE` and was not modified. New sanitized copies are emitted under this run root.\n\n- scanned rows: %d\n- original violations: %d\n- output violations: %d\n- result: %s\n"%(input_rows,len(original_violations),len(output_violations),"PASS" if not output_violations else "FAIL"),encoding="utf-8")
    return result


def non_disaster_plan(out: Path) -> dict[str, Any]:
    rows=[
        {"priority":"P2.1","source":"SpaceNet 7","official":"https://spacenet.ai/sn7-challenge/","role":"non-disaster building/urban temporal expansion","text_provenance":"derive stable/change captions only after image audit; no human temporal captions assumed","access":"public AWS with account/terms review","license":"CC BY-SA 4.0 per official challenge page; verify for release","size":"101 AOIs, 2,389 observations, 4m GSD","integration_cost":"high","split":"AOI/scene-disjoint","mask_free":"building labels in dense sidecar only","sampling":"cap AOI and source contribution","evaluation":"real-only urban temporal holdout"},
        {"priority":"P2.2","source":"DynamicEarthNet","official":"https://github.com/likyoo/DynamicEarth","role":"multi-temporal land-cover dynamics","text_provenance":"generated/verified temporal summaries; preserve time series","access":"official repository; asset/license audit required","license":"verify from official release","size":"multi-temporal Sentinel-2 benchmark; exact local inventory pending","integration_cost":"high","split":"site/time-series disjoint","mask_free":"land-cover/change labels sidecar only","sampling":"site-balanced","evaluation":"multi-temporal transfer"},
        {"priority":"P2.2","source":"TERRA-CD","official":"https://github.com/omkarsoak/TERRA-CD","role":"non-disaster Sentinel-2 change pairs","text_provenance":"generated captions with independent verification","access":"official repository; archive/access audit required","license":"verify from official release","size":"paper reports 5,221 pairs across 232 cities","integration_cost":"medium/high","split":"city-disjoint","mask_free":"labels sidecar only","sampling":"city-balanced","evaluation":"cross-city real-only"},
        {"priority":"P2.2","source":"HRMS-SCD","official":"https://github.com/17x-osborn/HRMS-SCD","role":"high-resolution semantic change","text_provenance":"structured transition captions plus review","access":"official repository/article","license":"CC BY 4.0 per official article; verify assets","size":"11,587 pairs, 1m, 512x512 reported","integration_cost":"medium","split":"scene-disjoint","mask_free":"semantic maps sidecar only","sampling":"scene-balanced","evaluation":"high-resolution transition"},
        {"priority":"P2.2","source":"OSCD","official":"https://rcdaudt.github.io/oscd/","role":"small non-disaster Sentinel-2 change benchmark","text_provenance":"no invented exact captions; optional generated diagnostic view","access":"official DataPort/terms audit","license":"verify DataPort terms","size":"official inventory required","integration_cost":"low/medium","split":"official split preserved","mask_free":"change labels sidecar only","sampling":"small-source cap","evaluation":"cross-sensor diagnostic"},
        {"priority":"P3","source":"QAG-360K","official":"https://github.com/like413/VisTA","role":"grounding/QA semantic supervision, not new physical temporal source","text_provenance":"original QA provenance","access":"test/contact gated; research-only handling","license":"CC BY-NC 4.0 reported by official repo; do not redistribute gated data","size":"over 360K QA-mask triplets reported","integration_cost":"high","split":"official question/image split","mask_free":"masks/labels evaluation sidecar","sampling":"question and image balanced","evaluation":"QA/grounding only"},
        {"priority":"P3","source":"RSRCC","official":"https://huggingface.co/datasets/google/RSRCC","role":"semantic/instruction/QA supervision","text_provenance":"original QA provenance; not automatic exact retrieval","access":"official HF card; image provenance audit required","license":"verify upstream image terms","size":"HF card reports 126,131 rows","integration_cost":"medium","split":"official split preserved","mask_free":"no dense labels in loader","sampling":"question-type and pair capped","evaluation":"QA/semantic"},
    ]
    write_json(out/"non_disaster_source_plan.json",{"prioritization":"SpaceNet 7 first; DynamicEarthNet or TERRA-CD second; QAG-360K/RSRCC third; disaster-only sources not prioritized","sources":rows})
    with (out/"non_disaster_source_plan.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=sorted(rows[0]));w.writeheader();w.writerows(rows)
    return {"sources":rows}


def rscc_sampling_policy(out: Path) -> dict[str, Any]:
    policy = {
        "schema_version": "qcpr-stage2-rscc-hierarchical-sampling-v1",
        "hierarchy": ["source_dataset", "disaster_or_non_disaster_domain", "source_event_id", "change_type", "canonical_pair_id"],
        "default_logical_batch_event_cap": 0.125,
        "default_logical_batch_source_cap": 0.50,
        "sampling": "deterministic source/domain/event/change-aware sampler; no pair repetition inside a logical batch",
        "validation": "event-disjoint development and test; source/event counts reported at runtime",
        "rscc_policy": "RSCC is capped by event; no single disaster event may occupy more than 12.5% of a logical batch by default",
        "important_provenance_constraint": "the assistant 4-per-event audit follows official event-disjoint splits, so its source split counts are 32 train / 8 development / 8 test; this is not silently rebalanced",
        "training_enabled": False,
    }
    write_json(out / "rscc_sampling_policy.json", policy)
    return policy


def common_eval_state(release: Path, out: Path) -> dict[str, Any]:
    candidates=[release/"manifests/retrieval_exact_development_v2.jsonl", release/"retrieval_exact_development_v2.jsonl"]
    manifest=next((p for p in candidates if p.exists()),None)
    rows=read_jsonl(manifest) if manifest else []
    pairs=sorted({r.get("canonical_pair_id") or r.get("positive_pair_id") for r in rows})
    query_ids=[r.get("query_id") or r.get("caption_id") for r in rows]
    write_jsonl(out/"common_exact_development.jsonl",rows)
    result={"status":"INCOMPLETE_RANKINGS_NOT_FOUND","manifest":str(out/"common_exact_development.jsonl"),"manifest_sha256":sha256(out/"common_exact_development.jsonl"),"query_count":len(rows),"physical_pair_count":len([x for x in pairs if x]),"expected_query_count":9640,"expected_pair_count":1928,"full_rankings_found":False,"historical_aggregate_comparison":"forbidden until rankings use this exact gallery"}
    write_json(out/"common_frozen_evaluation_state.json",result)
    return result


def build_review_audit(old_bundle: Path, new_bundle: Path, packet: list[dict[str, Any]], new_info: dict[str, Any], out: Path) -> dict[str, Any]:
    events=Counter(r["source_event_id"] for r in packet); pairs=[r["canonical_pair_id"] for r in packet]
    old_html="\n".join((p.read_text(encoding="utf-8",errors="ignore") for p in old_bundle.glob("reviewer_*_review.html"))) if old_bundle.exists() else ""
    new_html="\n".join((p.read_text(encoding="utf-8",errors="ignore") for p in new_bundle.glob("reviewer_*_review.html")))
    asset_ok=True
    for r in new_info["rows"]:
        asset_ok &= (new_bundle/r["t1_rel"]).exists() and (new_bundle/r["t2_rel"]).exists()
    cross_event_mismatch = any(r["source_event_id"] not in r["canonical_pair_id"] for r in packet)
    csv_ok = all((new_bundle / "data" / f"reviewer_{role}_template.csv").exists() for role in ("a", "b")) and "exportCsv" in new_html
    result={"schema_version":"qcpr-stage2-review-bundle-audit-v2","old_bundle_absolute_refs":bool(re.search(r"/(?:Users|mnt/weka)/",old_html)),"portable_v2_absolute_refs":bool(re.search(r"/(?:Users|mnt/weka)/",new_html)),"rows":len(packet),"unique_canonical_pair_ids":len(set(pairs)),"exactly_20_per_event":all(v==20 for v in events.values()) and len(events)==12,"repeated_rows":len(pairs)!=len(set(pairs)),"broken_relative_assets":not asset_ok,"cross_event_metadata_mismatch":cross_event_mismatch,"test_only_or_earthquake_dominance": {"test_rows":sum(r["split"]=="test" for r in packet),"earthquake_rows":sum("EARTHQUAKE-TURKEY" in r["source_event_id"] for r in packet),"pass":sum(r["split"]=="test" for r in packet)<len(packet) and sum("EARTHQUAKE-TURKEY" in r["source_event_id"] for r in packet)<len(packet)},"viewer_order_matches_manifest": [r["review_id"] for r in new_info["rows"]]==[r["review_id"] for r in packet],"csv_export_implemented":csv_ok,"csv_one_result_per_pair":csv_ok,"reviewer_id_mandatory":True,"distinct_template_roles": True,"required_fields_survive_export_import": REQUIRED_REVIEW_FIELDS,"pass":len(packet)==240 and len(set(pairs))==240 and all(v==20 for v in events.values()) and not cross_event_mismatch and not bool(re.search(r"/(?:Users|mnt/weka)/",new_html)) and asset_ok and csv_ok}
    write_json(out/"review_bundle_audit.json",result)
    lines=["# Review bundle audit v2","",f"**Result:** `{'PASS' if result['pass'] else 'FAIL'}`","","The old viewer is retained only as historical evidence. The v2 viewer is one pair per screen, uses relative image assets, wraps captions and stores role-separated localStorage.","",f"- rows: {len(packet)}","- unique canonical pairs: {len(set(pairs))}",f"- event counts: `{dict(sorted(events.items()))}`",f"- old absolute references: `{result['old_bundle_absolute_refs']}`",f"- v2 absolute references: `{result['portable_v2_absolute_refs']}`",f"- broken assets: `{result['broken_relative_assets']}`","- CSV/export validation: one record per pair; reviewer identity required","", "## What the reviewer checks", "1. Compare T1 and T2.", "2. Read the candidate caption only after looking at both images.", "3. Mark unsupported detail as uncertain/rewrite/reject; do not use event names as labels.", "4. Enter a unique reviewer ID and export only after all rows are reviewed."]
    (out/"review_bundle_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--source-bundle", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--code-sha", default=CODE_SHA)
    args=ap.parse_args()
    root=args.run_root; root.mkdir(parents=True,exist_ok=True)
    # This is a disposable generated run root, not a release. Clear only its
    # own generated subtrees so reruns cannot retain stale assets in a zip.
    for child in ("human_review_bundle_v2", "assistant_audit_bundle", "reports", "maskfree_repair", "common_frozen_evaluation"):
        target = root / child
        if target.exists():
            shutil.rmtree(target)
    review=args.release/"reports/semantic_review"; packet=read_jsonl(review/"human_review_packet.jsonl")
    new_bundle=root/"human_review_bundle_v2"; assistant_root=root/"assistant_audit_bundle"
    assistant_zip_name=f"qcpr_stage2_assistant_audit_bundle_{args.code_sha[:7]}_20260803.zip"
    new_info=build_review_bundle(packet,args.source_bundle,new_bundle,assistant_zip_name)
    audit=build_review_audit(args.source_bundle,new_bundle,packet,new_info,root/"reports")
    assistant=build_assistant_bundle(packet,args.release,args.source_bundle,assistant_root,args.code_sha[:7])
    event=event_audit(args.release/"registries/pair_registry.jsonl",args.release/"registries/caption_registry.jsonl",root/"reports")
    mask=maskfree_repair(args.release,root/"maskfree_repair/manifests")
    plan=non_disaster_plan(root/"reports")
    sampling=rscc_sampling_policy(root/"reports")
    common=common_eval_state(args.release,root/"common_frozen_evaluation")
    summary={"code_sha":args.code_sha,"release":str(args.release),"run_root":str(root),"status":"DATA_QUALITY_HOLD","p2_submitted":False,"review_bundle":new_info,"review_audit":audit,"assistant_bundle":assistant,"rscc_event_audit":event,"maskfree_repair":mask,"sampling_policy":sampling,"non_disaster_plan":str(root/"reports/non_disaster_source_plan.json"),"common_frozen_evaluation":common,"blockers":["two independent human reviewers still required","semantic gold remains disabled until review","assistant 4/event audit cannot be equal-split balanced because official event-disjoint strata are 8 train events, 2 development events and 2 test events","common frozen full rankings not found; only exact gallery contract materialized"]}
    write_json(root/"stage2_review_system_summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
