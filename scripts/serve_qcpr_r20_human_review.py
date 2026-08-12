#!/usr/bin/env python3
"""Serve the blind QCPR r20 review packet as a local, fail-closed web form.

Usage on the cluster (the default bind is localhost only)::

    python3 scripts/serve_qcpr_r20_human_review.py \
      --review-package /path/to/human_review_package \
      --reviewer reviewer_a --port 8765

Then forward the port with SSH and open http://127.0.0.1:8765.  The runner
serves only the selected blind packet and ``blind_assets``.  It never serves
the producer-only join ledger, source lineage, or model artifacts.  Saving a
row updates only that reviewer's decision template atomically; it does not
create adjudication or training labels.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import tempfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


DECISIONS = (
    "EXACT",
    "SEMANTIC_ONLY",
    "AMBIGUOUS_IGNORE",
    "REWRITE_REQUIRED",
    "REJECT",
)
QUESTION_VALUES = {"yes", "no", "uncertain"}
REVIEWERS = {"reviewer_a", "reviewer_b"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(value)
    return rows


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def identity_key(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def timestamp_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: Any) -> bool:
    if not nonempty(value):
        return False
    try:
        datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


class ReviewStore:
    """Load and atomically update one blind reviewer template."""

    def __init__(self, package: Path, reviewer: str) -> None:
        if reviewer not in REVIEWERS:
            raise ValueError(f"reviewer must be one of {sorted(REVIEWERS)}")
        self.package = package.resolve()
        self.reviewer = reviewer
        self.packet_path = self.package / f"{reviewer}_packet.jsonl"
        self.template_path = self.package / f"{reviewer}_decision_template.jsonl"
        self.asset_root = (self.package / "blind_assets").resolve()
        self.packet_rows = read_jsonl(self.packet_path)
        self.decision_rows = read_jsonl(self.template_path)
        if len(self.packet_rows) != 1500 or len(self.decision_rows) != 1500:
            raise ValueError("review packet and template must each contain 1,500 rows")
        self.packet_by_id = self._index(self.packet_rows, "packet")
        self.decision_by_id = self._index(self.decision_rows, "decision template")
        if set(self.packet_by_id) != set(self.decision_by_id):
            raise ValueError("packet and decision template sample IDs do not match")
        for sample_id, row in self.packet_by_id.items():
            self._validate_visible_row(sample_id, row)
        self._validate_templates()

    @staticmethod
    def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for row in rows:
            sample_id = str(row.get("review_sample_id") or "")
            if not sample_id or sample_id in indexed:
                raise ValueError(f"duplicate/missing review_sample_id in {label}")
            indexed[sample_id] = row
        return indexed

    @staticmethod
    def _validate_visible_row(sample_id: str, row: dict[str, Any]) -> None:
        forbidden = {"query_id", "source_pair_id", "source_dataset", "source_event_id", "event_id"}
        if forbidden.intersection(row):
            raise ValueError(f"blind packet exposes canonical/source fields: {sample_id}")
        if row.get("paths_are_blind_aliases") is not True or row.get("visible_internal_ids") is not False:
            raise ValueError(f"blind packet is not marked blind: {sample_id}")
        for path in [
            (row.get("true_pair") or {}).get("t1_path"),
            (row.get("true_pair") or {}).get("t2_path"),
        ] + [
            value
            for neighbour in row.get("candidate_neighbours") or []
            for value in (neighbour.get("t1_path"), neighbour.get("t2_path"))
        ]:
            if not nonempty(path) or not str(path).startswith("blind_assets/"):
                raise ValueError(f"blind packet has unsafe image path: {sample_id}")

    def _validate_templates(self) -> None:
        for sample_id, row in self.decision_by_id.items():
            if row.get("reviewer_id") != self.reviewer:
                raise ValueError(f"wrong reviewer_id in template: {sample_id}")
            decision = row.get("final_decision")
            if decision not in (None, "") and decision not in DECISIONS:
                raise ValueError(f"invalid existing decision: {sample_id}")

    def rows_for_browser(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for sample_id in self.packet_by_id:
            row = json.loads(json.dumps(self.packet_by_id[sample_id]))
            row["reviewer_fields"] = self.decision_by_id[sample_id]
            result.append(row)
        return result

    def state(self) -> dict[str, Any]:
        completed = sum(1 for row in self.decision_by_id.values() if row.get("final_decision") not in (None, ""))
        identities = sorted({identity_key(row.get("reviewer_identity")) for row in self.decision_by_id.values() if nonempty(row.get("reviewer_identity"))})
        return {
            "reviewer": self.reviewer,
            "rows": len(self.packet_by_id),
            "completed": completed,
            "remaining": len(self.packet_by_id) - completed,
            "identities": identities,
            "training_enabled": False,
            "adjudication_created": False,
        }

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        sample_id = str(payload.get("review_sample_id") or "")
        if sample_id not in self.packet_by_id:
            raise ValueError("unknown review_sample_id")
        identity = identity_key(payload.get("reviewer_identity"))
        if not identity:
            raise ValueError("reviewer_identity is required")
        existing_identities = {
            identity_key(row.get("reviewer_identity"))
            for row in self.decision_by_id.values()
            if nonempty(row.get("reviewer_identity"))
        }
        if existing_identities and identity not in existing_identities:
            raise ValueError("all rows in one reviewer template must use the same reviewer identity")
        if payload.get("independence_attestation") is not True:
            raise ValueError("independence_attestation must be explicitly checked")
        for field in ("caption_accurate", "identifiable_against_neighbours", "nonexact_candidate_consistent"):
            if payload.get(field) not in QUESTION_VALUES:
                raise ValueError(f"{field} must be yes, no, or uncertain")
        decision = payload.get("final_decision")
        if decision not in DECISIONS:
            raise ValueError("final_decision is invalid")
        rewritten = payload.get("rewritten_caption")
        if decision == "REWRITE_REQUIRED" and not nonempty(rewritten):
            raise ValueError("REWRITE_REQUIRED requires rewritten_caption")
        confidence = payload.get("confidence")
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence must be numeric") from exc
        if not 0.0 <= confidence_value <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        reviewed_at = str(payload.get("reviewed_at") or timestamp_now())
        if not parse_timestamp(reviewed_at):
            raise ValueError("reviewed_at must be an ISO timestamp")
        row = self.decision_by_id[sample_id]
        row.update(
            {
                "reviewer_identity": identity,
                "reviewed_at": reviewed_at,
                "independence_attestation": "I independently reviewed this row without the other reviewer's decisions.",
                "caption_accurate": payload["caption_accurate"],
                "identifiable_against_neighbours": payload["identifiable_against_neighbours"],
                "nonexact_candidate_consistent": payload["nonexact_candidate_consistent"],
                "final_decision": decision,
                "rewritten_caption": rewritten if nonempty(rewritten) else None,
                "confidence": confidence_value,
                "notes": str(payload.get("notes") or "").strip() or None,
            }
        )
        rows = [self.decision_by_id[sample_id] for sample_id in self.packet_by_id]
        self._atomic_write(rows)
        return {"ok": True, "sample_id": sample_id, "state": self.state()}

    def _atomic_write(self, rows: list[dict[str, Any]]) -> None:
        self.template_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.template_path.name}.", dir=self.template_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.template_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QCPR r20 blind human review</title>
<style>
body{font:14px system-ui,sans-serif;margin:0;background:#f5f6f8;color:#18202a}
header{position:sticky;top:0;background:#17202b;color:white;padding:10px 18px;z-index:2}
main{max-width:1500px;margin:18px auto;padding:0 18px}.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button{padding:7px 12px;border:1px solid #8793a1;border-radius:5px;background:white;cursor:pointer}
button.primary{background:#146c43;color:white;border-color:#146c43}button:disabled{opacity:.45}
.card{background:white;border:1px solid #d8dde4;border-radius:8px;padding:14px;margin:12px 0}
.images{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.images img{width:100%;max-height:420px;object-fit:contain;background:#111;border-radius:4px}
.candidate-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.candidate{border:1px solid #d8dde4;padding:8px;border-radius:6px}
label{display:block;margin:8px 0}input[type=text],input[type=number],textarea,select{width:100%;box-sizing:border-box;padding:7px;border:1px solid #adb7c2;border-radius:4px}
textarea{min-height:60px}.question{display:grid;grid-template-columns:1fr auto;align-items:center;gap:10px}.status{margin-left:auto}.ok{color:#126b3e}.err{color:#a51d2d}.small{font-size:12px;color:#586575}
@media(max-width:800px){.candidate-grid{grid-template-columns:1fr}.images{grid-template-columns:1fr}.question{grid-template-columns:1fr}}
</style></head>
<body><header><div class="bar"><strong>QCPR r20 blind human review</strong><span id="progress"></span><span class="status" id="status"></span></div></header>
<main><div class="card"><div class="bar"><label style="flex:1">Reviewer identity<input id="identity" type="text" autocomplete="off"></label><label style="width:180px">Confidence<input id="confidence" type="number" min="0" max="1" step="0.01" value="0.8"></label></div><p class="small">Review the visible T1→T2 pair and candidates independently. No source IDs, model scores, masks, or rankings are available in this interface.</p></div>
<section id="content"></section>
<div class="card"><h3>Decision</h3><div class="question"><span>Caption accurately describes intended pair</span><select id="caption_accurate"><option value="">select</option><option>yes</option><option>no</option><option>uncertain</option></select></div><div class="question"><span>Caption identifies this pair against neighbours</span><select id="identifiable_against_neighbours"><option value="">select</option><option>yes</option><option>no</option><option>uncertain</option></select></div><div class="question"><span>Any nonexact neighbour fully consistent with caption</span><select id="nonexact_candidate_consistent"><option value="">select</option><option>yes</option><option>no</option><option>uncertain</option></select></div><label>Final decision<select id="final_decision"><option value="">select</option><option>EXACT</option><option>SEMANTIC_ONLY</option><option>AMBIGUOUS_IGNORE</option><option>REWRITE_REQUIRED</option><option>REJECT</option></select></label><label id="rewrite_wrap" hidden>Verified rewrite from visible T1/T2 only<textarea id="rewritten_caption"></textarea></label><label>Notes<textarea id="notes"></textarea></label><label><input id="independent" type="checkbox"> I independently reviewed this row without the other reviewer's decisions.</label></div>
<div class="bar"><button id="prev">Previous</button><button id="next">Next</button><button class="primary" id="save">Save decision</button><button id="save_next" class="primary">Save and next</button></div></main>
<script>
let rows=[],index=0,state={};
const $=id=>document.getElementById(id);
function setStatus(text,good=false){$('status').textContent=text;$('status').className=good?'status ok':'status err'}
function path(p){return '/'+p.split('/').map(encodeURIComponent).join('/')}
function img(src,alt){const e=document.createElement('img');e.src=path(src);e.alt=alt;return e}
function render(){const r=rows[index];$('progress').textContent=`${index+1}/${rows.length} · ${r.stratum}`;$('content').replaceChildren();const card=document.createElement('div');card.className='card';const h=document.createElement('h2');h.textContent=r.caption;card.append(h);const title=document.createElement('h3');title.textContent='Intended pair';card.append(title);const im=document.createElement('div');im.className='images';im.append(img(r.true_pair.t1_path,'intended T1'),img(r.true_pair.t2_path,'intended T2'));card.append(im);const ch=document.createElement('h3');ch.textContent='Candidate neighbours (not labels)';card.append(ch);const grid=document.createElement('div');grid.className='candidate-grid';(r.candidate_neighbours||[]).forEach(n=>{const c=document.createElement('div');c.className='candidate';const t=document.createElement('strong');t.textContent=n.candidate_slot;c.append(t);const ci=document.createElement('div');ci.className='images';ci.append(img(n.t1_path,n.candidate_slot+' T1'),img(n.t2_path,n.candidate_slot+' T2'));c.append(ci);grid.append(c)});card.append(grid);$('content').append(card);fill(r.reviewer_fields||{});$('prev').disabled=index===0;$('next').disabled=index===rows.length-1}
function fill(d){['caption_accurate','identifiable_against_neighbours','nonexact_candidate_consistent','final_decision','rewritten_caption','notes','confidence'].forEach(k=>{if($(k)&&d[k]!==null&&d[k]!==undefined)$(k).value=d[k]});$('identity').value=d.reviewer_identity||localStorage.qcprReviewerIdentity||'';$('independent').checked=!!d.independence_attestation;$('rewrite_wrap').hidden=$('final_decision').value!=='REWRITE_REQUIRED'}
function payload(){return {review_sample_id:rows[index].review_sample_id,reviewer_identity:$('identity').value,independence_attestation:$('independent').checked,caption_accurate:$('caption_accurate').value,identifiable_against_neighbours:$('identifiable_against_neighbours').value,nonexact_candidate_consistent:$('nonexact_candidate_consistent').value,final_decision:$('final_decision').value,rewritten_caption:$('rewritten_caption').value,confidence:$('confidence').value,notes:$('notes').value}}
async function save(goNext){const p=payload();if(p.reviewer_identity)localStorage.qcprReviewerIdentity=p.reviewer_identity;setStatus('saving…');const res=await fetch('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});const x=await res.json();if(!res.ok){setStatus(x.error||'save failed');return}rows[index].reviewer_fields={...rows[index].reviewer_fields,...p,independence_attestation:'I independently reviewed this row without the other reviewer's decisions.',reviewed_at:new Date().toISOString()};state=x.state;setStatus(`saved · ${state.completed}/${state.rows}`,true);if(goNext&&index<rows.length-1){index++;render()}}
$('save').onclick=()=>save(false);$('save_next').onclick=()=>save(true);$('prev').onclick=()=>{if(index){index--;render()}};$('next').onclick=()=>{if(index<rows.length-1){index++;render()}};$('final_decision').onchange=()=>{$('rewrite_wrap').hidden=$('final_decision').value!=='REWRITE_REQUIRED'};
(async()=>{const [a,b]=await Promise.all([fetch('/api/rows'),fetch('/api/state')]);rows=await a.json();state=await b.json();if(!rows.length){setStatus('no rows');return}render();setStatus(`${state.completed}/${state.rows} completed`,true)})().catch(e=>setStatus(String(e)));
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    store: ReviewStore

    def _send(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/rows":
            self._send(200, json_bytes(self.store.rows_for_browser()))
            return
        if parsed.path == "/api/state":
            self._send(200, json_bytes(self.store.state()))
            return
        if parsed.path.startswith("/blind_assets/"):
            relative = unquote(parsed.path.lstrip("/"))
            target = (self.store.package / relative).resolve()
            if not str(target).startswith(str(self.store.asset_root)) or not target.is_file():
                self._send(404, b'{"error":"not found"}')
                return
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self._send(200, target.read_bytes(), content_type)
            return
        self._send(404, b'{"error":"not found"}')

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/decision":
            self._send(404, b'{"error":"not found"}')
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 100_000:
                raise ValueError("request too large")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            self._send(200, json_bytes(self.store.save(payload)))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, json_bytes({"error": str(exc)}))

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[review-runner] {format % args}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-package", type=Path, required=True)
    parser.add_argument("--reviewer", choices=sorted(REVIEWERS), required=True)
    parser.add_argument("--bind", default="127.0.0.1", help="Bind address; localhost is the safe default.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    store = ReviewStore(args.review_package, args.reviewer)
    Handler.store = store
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"Serving {args.reviewer} blind review on http://{args.bind}:{args.port}")
    print("Only the selected packet and blind_assets are served; Ctrl-C stops the runner.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
