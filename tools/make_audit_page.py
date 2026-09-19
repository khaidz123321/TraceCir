#!/usr/bin/env python
"""Build a self-contained HTML page for the P0 manual compiler audit
(protocol Sec. 5.3): random queries, each showing the reference image, the
modification text and the compiler output, with Correct / Partially correct /
Incorrect buttons and a running tally against the ~85% threshold.

The page embeds resized reference images, keeps labels in the browser's
localStorage (a refresh loses nothing) and can export them as JSON.

Example:
    python tools/make_audit_page.py --dataset circo \
        --data-root data/circo/circo \
        --compiled circo_compiled.jsonl --n 100 --seed 0 \
        --out audit/audit_circo_val.html
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets import CIRCODataset, CIRRDataset  # noqa: E402


def image_data_uri(path: Path, max_side: int = 520) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_items(args) -> list[dict]:
    noop = lambda image: image  # noqa: E731
    if args.dataset == "circo":
        ds = CIRCODataset(args.data_root, args.split, "relative", noop)
        query_id = lambda item: str(item["reference_img_id"])  # noqa: E731
        image_path = lambda item: ds._image_path(item["reference_img_id"])  # noqa: E731
    else:
        ds = CIRRDataset(args.data_root, args.split, "relative", noop)
        query_id = lambda item: item["reference_name"] + "|" + str(item["pair_id"])  # noqa: E731
        image_path = lambda item: ds._image_path(item["reference_name"])  # noqa: E731

    compiled = {}
    with open(args.compiled, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            compiled[record["query_id"]] = record

    items = []
    for index in range(len(ds)):
        item = ds[index]
        qid = query_id(item)
        record = compiled.get(qid)
        if record is None:
            continue
        items.append({
            "query_id": qid,
            "modification": item["relative_caption"],
            "target": record["target"],
            "atoms": record["atoms"],
            "parse_ok": record["parse_ok"],
            "image": image_data_uri(image_path(item)),
        })
    return items


PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audit compiler P0</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --ink:#1b1f24; --mute:#66707c; --line:#d9dee4;
          --ok:#1a7f4b; --part:#b7791f; --bad:#c0392b; --accent:#2b5fd9; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14171b; --card:#1d2126; --ink:#e8ebef; --mute:#98a2ae; --line:#333a42;
            --ok:#4cc38a; --part:#e0a94a; --bad:#ef7b6e; --accent:#7ea3ff; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif; }
  header { position:sticky; top:0; z-index:5; background:var(--card); border-bottom:1px solid var(--line);
           padding:10px 18px; display:flex; flex-wrap:wrap; gap:14px; align-items:center; }
  header b { font-size:16px; }
  .tally span { margin-right:12px; white-space:nowrap; }
  .ok{color:var(--ok)} .part{color:var(--part)} .bad{color:var(--bad)}
  main { max-width:980px; margin:18px auto; padding:0 14px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px;
          display:grid; grid-template-columns:minmax(0,340px) minmax(0,1fr); gap:18px; }
  @media (max-width:760px) { .card { grid-template-columns:1fr; } }
  .card img { width:100%; border-radius:8px; border:1px solid var(--line); background:#0002; }
  .lab { font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:var(--mute); margin:12px 0 2px; }
  .mod { font-size:17px; font-weight:600; }
  table { border-collapse:collapse; width:100%; font-size:14px; }
  th,td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
  th { color:var(--mute); font-weight:500; font-size:12px; }
  .op { font-weight:700; font-family:ui-monospace,Consolas,monospace; }
  .nul { color:var(--mute); font-style:italic; }
  .warn { color:var(--bad); font-weight:600; }
  .btns { display:flex; gap:8px; flex-wrap:wrap; margin-top:14px; }
  button { font:inherit; padding:8px 14px; border-radius:8px; border:1px solid var(--line); background:var(--card);
           color:var(--ink); cursor:pointer; }
  button:hover { border-color:var(--accent); }
  button.sel-ok{background:var(--ok);color:#fff;border-color:var(--ok)}
  button.sel-part{background:var(--part);color:#fff;border-color:var(--part)}
  button.sel-bad{background:var(--bad);color:#fff;border-color:var(--bad)}
  textarea { width:100%; margin-top:10px; padding:8px; border-radius:8px; border:1px solid var(--line);
             background:var(--bg); color:var(--ink); font:inherit; min-height:44px; resize:vertical; }
  .nav { display:flex; gap:8px; align-items:center; margin:14px 0; }
  .hint { color:var(--mute); font-size:13px; }
  #out { width:100%; min-height:120px; margin-top:8px; font:12px ui-monospace,Consolas,monospace; }
  details { margin-top:22px; }
</style></head><body>
<header>
  <b>Audit compiler P0</b>
  <span id="pos"></span>
  <span class="tally"><span class="ok" id="tOk"></span><span class="part" id="tPart"></span><span class="bad" id="tBad"></span><span id="tDone"></span></span>
  <span id="verdict"></span>
</header>
<main>
  <div class="nav">
    <button id="prev">&larr; Trước</button><button id="next">Tiếp &rarr;</button>
    <button id="nextTodo">Câu chưa chấm kế tiếp</button>
    <span class="hint">Phím: 1 = Correct, 2 = Partially, 3 = Incorrect, ←/→ chuyển câu</span>
  </div>
  <div id="card"></div>
  <details><summary>Xuất kết quả</summary>
    <div class="nav"><button id="dl">Tải file JSON</button><button id="cp">Sao chép JSON</button>
      <button id="clear">Xóa hết nhãn</button></div>
    <textarea id="out" readonly></textarea>
  </details>
</main>
<script>
const ITEMS = __ITEMS__;
const KEY = "audit-__NAME__";
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let i = 0;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const val = v => v === null || v === "" ? '<span class="nul">null</span>' : esc(v);
function violation(a) {
  if (a.operation === "ADD" && (a.target_state == null || a.source_state != null)) return "ADD phải chỉ có target_state";
  if ((a.operation === "REMOVE" || a.operation === "PRESERVE") && (a.source_state == null || a.target_state != null)) return a.operation + " phải chỉ có source_state";
  if (a.operation === "REPLACE" && (a.source_state == null || a.target_state == null)) return "REPLACE cần cả hai";
  return "";
}
function save() { try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {} }
function counts() {
  let c = {correct:0, partial:0, incorrect:0};
  ITEMS.forEach(it => { const l = labels[it.query_id]; if (l && c[l.label] !== undefined) c[l.label]++; });
  return c;
}
function tally() {
  const c = counts(), n = c.correct + c.partial + c.incorrect;
  $("tOk").textContent = "Correct " + c.correct;
  $("tPart").textContent = "Partially " + c.partial;
  $("tBad").textContent = "Incorrect " + c.incorrect;
  $("tDone").textContent = "Đã chấm " + n + "/" + ITEMS.length;
  if (n) {
    const p = 100 * c.correct / n, q = 100 * (c.correct + c.partial) / n;
    $("verdict").innerHTML = "Correct = <b>" + p.toFixed(1) + "%</b> (Correct+Partially = " + q.toFixed(1) + "%) — ngưỡng ~85%: " +
      (p >= 85 ? '<b class="ok">ĐẠT</b>' : '<b class="bad">CHƯA ĐẠT</b>');
  } else $("verdict").textContent = "";
  $("out").value = JSON.stringify({counts: c, labeled: n, total: ITEMS.length, labels}, null, 1);
}
function render() {
  const it = ITEMS[i], l = labels[it.query_id] || {};
  $("pos").textContent = "Câu " + (i + 1) + "/" + ITEMS.length + "  (id " + it.query_id + ")";
  const rows = it.atoms.map(a => {
    const v = violation(a);
    return "<tr><td class='op'>" + esc(a.operation) + "</td><td>" + val(a.source_state) + "</td><td>" + val(a.target_state) +
      "</td><td class='warn'>" + esc(v) + "</td></tr>";
  }).join("") || "<tr><td colspan='4' class='nul'>(không có atom)</td></tr>";
  $("card").innerHTML =
    "<div class='card'><div><img src='" + it.image + "' alt='reference'></div><div>" +
    "<div class='lab'>Câu lệnh chỉnh sửa (mô tả ảnh ĐÍCH)</div><div class='mod'>" + esc(it.modification) + "</div>" +
    "<div class='lab'>Target do compiler sinh</div><div>" + (it.target ? esc(it.target) : "<span class='warn'>(rỗng)</span>") + "</div>" +
    (it.parse_ok ? "" : "<div class='warn'>parse_ok = False (đếm là Incorrect)</div>") +
    "<div class='lab'>Atoms</div><table><tr><th>Phép</th><th>source_state (ảnh tham chiếu)</th><th>target_state (ảnh đích)</th><th></th></tr>" + rows + "</table>" +
    "<div class='btns'>" +
    "<button data-l='correct' class='" + (l.label === "correct" ? "sel-ok" : "") + "'>1 · Correct</button>" +
    "<button data-l='partial' class='" + (l.label === "partial" ? "sel-part" : "") + "'>2 · Partially correct</button>" +
    "<button data-l='incorrect' class='" + (l.label === "incorrect" ? "sel-bad" : "") + "'>3 · Incorrect</button></div>" +
    "<textarea id='note' placeholder='Ghi chú (không bắt buộc): thiếu gì, sai gì...'>" + esc(l.note || "") + "</textarea>" +
    "</div></div>";
  document.querySelectorAll(".btns button").forEach(b => b.onclick = () => setLabel(b.dataset.l));
  $("note").oninput = e => { const cur = labels[it.query_id] || {}; cur.note = e.target.value; labels[it.query_id] = cur; save(); tally(); };
  tally();
}
function setLabel(lab) {
  const it = ITEMS[i], cur = labels[it.query_id] || {};
  cur.label = lab; labels[it.query_id] = cur; save(); render();
}
function go(d) { i = Math.min(ITEMS.length - 1, Math.max(0, i + d)); render(); window.scrollTo(0, 0); }
$("prev").onclick = () => go(-1); $("next").onclick = () => go(1);
$("nextTodo").onclick = () => {
  for (let k = 1; k <= ITEMS.length; k++) { const j = (i + k) % ITEMS.length;
    if (!(labels[ITEMS[j].query_id] || {}).label) { i = j; render(); window.scrollTo(0, 0); return; } }
};
document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA") return;
  if (e.key === "1") setLabel("correct"); else if (e.key === "2") setLabel("partial"); else if (e.key === "3") setLabel("incorrect");
  else if (e.key === "ArrowRight") go(1); else if (e.key === "ArrowLeft") go(-1);
});
$("dl").onclick = () => { const b = new Blob([$("out").value], {type:"application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = "audit___NAME__.json"; a.click(); };
$("cp").onclick = () => { $("out").select(); document.execCommand("copy"); };
$("clear").onclick = () => { if (confirm("Xóa toàn bộ nhãn đã chấm?")) { labels = {}; save(); render(); } };
render();
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["circo", "cirr"], required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--compiled", required=True, help="JSONL from compile_cli / run_compiler_batch")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    items = build_items(args)
    random.Random(args.seed).shuffle(items)
    items = items[: args.n]

    name = f"{args.dataset}-{args.split}-seed{args.seed}"
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    page = PAGE.replace("__ITEMS__", payload).replace("__NAME__", html.escape(name))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"{len(items)} cau -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
