from __future__ import annotations

import argparse
import json
import mimetypes
import sqlite3
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
VALID_DECISIONS = {"good", "bad", "unsure"}

HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Civitai Validation</title>
<style>
:root{
  color-scheme:dark;
  --bg:#090a0d;--panel:#101217;--line:#242936;--text:#f3f4f6;--muted:#8e95a5;
  --good:#2bb673;--bad:#db4b4b;--unsure:#707789
}
*{box-sizing:border-box}html,body{width:100%;height:100%;margin:0;background:var(--bg);color:var(--text);
font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;overflow:hidden}
.app{height:100vh;display:grid;grid-template-rows:58px minmax(0,1fr) 104px 34px}
header{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:18px;padding:0 22px;background:#101217;
border-bottom:1px solid var(--line)}
.brand{font-size:14px;font-weight:800;letter-spacing:.08em}
.progress-wrap{min-width:min(540px,48vw);display:grid;grid-template-columns:auto minmax(120px,1fr);align-items:center;gap:14px}
.progress-label{font-size:13px;font-variant-numeric:tabular-nums}.progress-track{height:5px;border-radius:999px;background:#242936;overflow:hidden}
.progress-bar{width:0%;height:100%;background:#a3a9ff;transition:width .15s ease}.counts{justify-self:end;display:flex;gap:13px;font-size:12px;color:var(--muted)}
.counts strong{color:#e9ebf1}main{min-height:0;display:grid;grid-template-columns:minmax(0,68fr) minmax(320px,32fr)}
.image-panel{min-width:0;min-height:0;display:flex;align-items:center;justify-content:center;padding:18px;background:#07080a;border-right:1px solid var(--line)}
.image-shell{width:100%;height:100%;display:flex;align-items:center;justify-content:center}
#image{max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;display:block;user-select:none}
.image-error{display:none;color:#e7a3a3}.prompt-panel{min-width:0;min-height:0;background:var(--panel);display:grid;grid-template-rows:auto minmax(0,1fr) auto}
.prompt-head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:17px 18px 13px;border-bottom:1px solid var(--line)}
.prompt-title{font-size:11px;font-weight:800;letter-spacing:.12em;color:var(--muted)}
.source-link{font-size:12px;color:#aeb4ff;text-decoration:none}.source-link:hover{text-decoration:underline}
#prompt{min-height:0;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;padding:18px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
font-size:14px;line-height:1.58;color:#e7e9ef}
.decision-chip{margin:0 18px 16px;padding:9px 12px;border:1px solid var(--line);border-radius:9px;color:var(--muted);font-size:12px;background:#0d0f14;display:none}
.decision-chip.visible{display:block}.decision-chip.good{color:#8be0b2}.decision-chip.bad{color:#f1a0a0}.decision-chip.unsure{color:#c7cad3}
.actions{background:var(--panel);border-top:1px solid var(--line);display:grid;grid-template-columns:repeat(3,minmax(140px,1fr));align-items:stretch;gap:12px;padding:13px 18px}
.action{border:0;border-radius:12px;color:#fff;cursor:pointer;font-size:18px;font-weight:800;position:relative}
.action small{position:absolute;right:13px;top:11px;font-size:11px;color:rgba(255,255,255,.6)}
.action:disabled{opacity:.45;cursor:not-allowed}.bad{background:var(--bad)}.unsure{background:var(--unsure)}.good{background:var(--good)}
footer{background:#0d0f13;border-top:1px solid #171a21;display:grid;grid-template-columns:1fr auto 1fr;align-items:center;padding:0 18px;color:var(--muted);font-size:11px}
.nav{display:flex;gap:10px}.nav button{border:0;background:transparent;color:#adb2bf;cursor:pointer}.nav button:disabled{opacity:.25;cursor:default}
#imageId{font-variant-numeric:tabular-nums}.save-state{justify-self:end}.save-state.error{color:#f08b8b}.save-state.saving{color:#d9c985}.save-state.saved{color:#76cca0}
.complete{display:none;grid-column:1/-1;place-items:center;background:var(--bg)}.complete.visible{display:grid}
.complete-card{width:min(560px,calc(100vw - 40px));background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:34px;text-align:center}
.complete-counts{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.complete-counts div{background:#151821;border-radius:10px;padding:16px 8px;color:var(--muted)}
.complete-counts strong{display:block;color:#fff;font-size:26px;margin-top:5px}
@media(max-width:900px){
  .app{grid-template-rows:58px minmax(0,1fr) 88px 34px}header{grid-template-columns:1fr auto}.counts{display:none}
  main{grid-template-columns:1fr;grid-template-rows:minmax(0,64fr) minmax(210px,36fr)}.image-panel{border-right:0;border-bottom:1px solid var(--line);padding:10px}
  .actions{padding:10px;gap:8px}.action{font-size:15px}
}
</style>
</head>
<body>
<div class="app">
<header>
  <div class="brand">VALIDATION</div>
  <div class="progress-wrap"><div class="progress-label" id="progressLabel">0 / 0</div><div class="progress-track"><div class="progress-bar" id="progressBar"></div></div></div>
  <div class="counts"><span>GOOD <strong id="goodCount">0</strong></span><span>BAD <strong id="badCount">0</strong></span><span>ХЗ <strong id="unsureCount">0</strong></span></div>
</header>

<main id="reviewMain">
  <section class="image-panel"><div class="image-shell"><img id="image" alt="review image"><div class="image-error" id="imageError">Не удалось показать изображение.</div></div></section>
  <aside class="prompt-panel">
    <div class="prompt-head"><div class="prompt-title">PROMPT</div><a id="sourceLink" class="source-link" href="#" target="_blank" rel="noopener noreferrer">Open source ↗</a></div>
    <div id="prompt"></div>
    <div id="decisionChip" class="decision-chip"></div>
  </aside>
</main>

<section id="complete" class="complete">
  <div class="complete-card"><h1>Review complete</h1><p>Все изображения размечены.</p>
    <div class="complete-counts"><div>GOOD<strong id="completeGood">0</strong></div><div>BAD<strong id="completeBad">0</strong></div><div>ХЗ<strong id="completeUnsure">0</strong></div></div>
  </div>
</section>

<section class="actions" id="actions">
  <button class="action bad" data-decision="bad">BAD <small>B</small></button>
  <button class="action unsure" data-decision="unsure">ХЗ <small>U</small></button>
  <button class="action good" data-decision="good">GOOD <small>G</small></button>
</section>

<footer>
  <div class="nav"><button id="previousButton">← назад</button><button id="nextButton">вперёд →</button></div>
  <div id="imageId">—</div>
  <div class="save-state" id="saveState"></div>
</footer>
</div>

<script>
(() => {
"use strict";
const e={
  reviewMain:document.getElementById("reviewMain"),complete:document.getElementById("complete"),actions:document.getElementById("actions"),
  image:document.getElementById("image"),imageError:document.getElementById("imageError"),prompt:document.getElementById("prompt"),
  sourceLink:document.getElementById("sourceLink"),progressLabel:document.getElementById("progressLabel"),progressBar:document.getElementById("progressBar"),
  goodCount:document.getElementById("goodCount"),badCount:document.getElementById("badCount"),unsureCount:document.getElementById("unsureCount"),
  completeGood:document.getElementById("completeGood"),completeBad:document.getElementById("completeBad"),completeUnsure:document.getElementById("completeUnsure"),
  imageId:document.getElementById("imageId"),decisionChip:document.getElementById("decisionChip"),previousButton:document.getElementById("previousButton"),
  nextButton:document.getElementById("nextButton"),saveState:document.getElementById("saveState"),buttons:[...document.querySelectorAll("[data-decision]")]
};
let state=null,busy=false;
function msg(text="",kind=""){e.saveState.textContent=text;e.saveState.className="save-state"+(kind?` ${kind}`:"")}
function lock(v){busy=v;e.buttons.forEach(b=>b.disabled=v);e.previousButton.disabled=v||!state||state.index<=0;e.nextButton.disabled=v||!state||state.index>=state.total-1}
function progress(){
  if(!state)return;const reviewed=state.counts.good+state.counts.bad+state.counts.unsure,pct=state.total?(reviewed/state.total)*100:0;
  e.progressLabel.textContent=`${reviewed} / ${state.total}`;e.progressBar.style.width=`${pct.toFixed(2)}%`;
  e.goodCount.textContent=state.counts.good;e.badCount.textContent=state.counts.bad;e.unsureCount.textContent=state.counts.unsure;
  e.completeGood.textContent=state.counts.good;e.completeBad.textContent=state.counts.bad;e.completeUnsure.textContent=state.counts.unsure
}
function chip(d){e.decisionChip.className="decision-chip";e.decisionChip.textContent="";if(!d)return;
  const labels={good:"Текущая оценка: GOOD",bad:"Текущая оценка: BAD",unsure:"Текущая оценка: ХЗ"};e.decisionChip.textContent=labels[d]||d;e.decisionChip.classList.add("visible",d)}
function render(){
  if(!state)return;progress();
  if(state.complete&&!state.item){e.reviewMain.style.display="none";e.actions.style.display="none";e.complete.classList.add("visible");e.imageId.textContent="готово";return}
  e.complete.classList.remove("visible");e.reviewMain.style.display="";e.actions.style.display="";
  const i=state.item;if(!i)return;e.imageError.style.display="none";e.image.style.display="block";e.image.src=`/image/${encodeURIComponent(i.image_id)}?v=${Date.now()}`;
  e.prompt.textContent=i.prompt||"";e.prompt.scrollTop=0;e.imageId.textContent=`Image #${i.image_id}`;
  if(i.source_url){e.sourceLink.href=i.source_url;e.sourceLink.style.visibility="visible"}else{e.sourceLink.href="#";e.sourceLink.style.visibility="hidden"}
  chip(i.decision);e.previousButton.disabled=busy||state.index<=0;e.nextButton.disabled=busy||state.index>=state.total-1
}
async function load(q=""){lock(true);try{const r=await fetch(`/api/state${q}`,{cache:"no-store"});if(!r.ok)throw new Error(`HTTP ${r.status}`);state=await r.json();render();msg("")}catch(err){msg(`Ошибка загрузки: ${err.message}`,"error")}finally{lock(false)}}
async function save(decision){
  if(busy||!state?.item)return;const image_id=state.item.image_id;lock(true);msg("сохраняю…","saving");
  try{const r=await fetch("/api/review",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({image_id,decision})});
    const p=await r.json().catch(()=>({}));if(!r.ok)throw new Error(p.error||`HTTP ${r.status}`);state=p;msg("сохранено","saved");render();setTimeout(()=>{if(e.saveState.classList.contains("saved"))msg("")},650)
  }catch(err){msg(`Не сохранено: ${err.message}`,"error")}finally{lock(false)}
}
async function nav(d){if(busy||!state)return;const n=state.index+d;if(n<0||n>=state.total)return;await load(`?index=${n}`)}
e.image.addEventListener("error",()=>{e.image.style.display="none";e.imageError.style.display="block"});
e.buttons.forEach(b=>b.addEventListener("click",()=>save(b.dataset.decision)));e.previousButton.addEventListener("click",()=>nav(-1));e.nextButton.addEventListener("click",()=>nav(1));
document.addEventListener("keydown",ev=>{if(ev.ctrlKey||ev.metaKey||ev.altKey||ev.repeat)return;const k=ev.key.toLowerCase();
  if(k==="g"){ev.preventDefault();save("good")}else if(k==="b"){ev.preventDefault();save("bad")}else if(k==="u"){ev.preventDefault();save("unsure")}
  else if(ev.key==="ArrowLeft"){ev.preventDefault();nav(-1)}else if(ev.key==="ArrowRight"){ev.preventDefault();nav(1)}
});
load();
})();
</script>
</body>
</html>"""

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    items, seen = [], set()
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            try:
                image_id = int(item["image_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Missing/invalid image_id in {path}:{line_no}") from exc
            if image_id in seen:
                continue
            seen.add(image_id)
            local_image_path = item.get("local_image_path") or item.get("image_path") or f"images/{image_id}.webp"
            items.append({**item,"image_id":image_id,"prompt":str(item.get("prompt") or ""),
                          "source_url":str(item.get("source_url") or item.get("url") or ""),
                          "local_image_path":str(local_image_path)})
    return items

class ReviewStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self):
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS reviews(
                image_id INTEGER PRIMARY KEY,
                decision TEXT NOT NULL,
                reasons_json TEXT NOT NULL DEFAULT '[]',
                note TEXT,
                reviewed_at TEXT NOT NULL
            )""")
            columns={r["name"] for r in conn.execute("PRAGMA table_info(reviews)").fetchall()}
            if "reasons_json" not in columns:
                conn.execute("ALTER TABLE reviews ADD COLUMN reasons_json TEXT NOT NULL DEFAULT '[]'")
            if "note" not in columns:
                conn.execute("ALTER TABLE reviews ADD COLUMN note TEXT")
            if "reviewed_at" not in columns:
                conn.execute("ALTER TABLE reviews ADD COLUMN reviewed_at TEXT")
            conn.commit()

    def all_decisions(self):
        with self._connect() as conn:
            rows=conn.execute("SELECT image_id,decision FROM reviews").fetchall()
        return {int(r["image_id"]):str(r["decision"]) for r in rows}

    def save(self,image_id:int,decision:str):
        if decision not in VALID_DECISIONS:
            raise ValueError(f"Invalid decision: {decision}")
        now=utc_now_iso()
        with self._connect() as conn:
            conn.execute("""INSERT INTO reviews(image_id,decision,reasons_json,note,reviewed_at)
                VALUES(?,?,'[]',NULL,?)
                ON CONFLICT(image_id) DO UPDATE SET
                  decision=excluded.decision,reasons_json='[]',note=NULL,reviewed_at=excluded.reviewed_at""",
                (image_id,decision,now))
            conn.commit()

class ReviewApp:
    def __init__(self, validation_dir:Path):
        self.validation_dir=validation_dir.resolve()
        self.items=read_manifest(self.validation_dir/"review_manifest.jsonl")
        self.by_id={int(i["image_id"]):i for i in self.items}
        self.index_by_id={int(i["image_id"]):n for n,i in enumerate(self.items)}
        self.store=ReviewStore(self.validation_dir/"review.sqlite3")

    @staticmethod
    def counts(decisions):
        return {"good":sum(v=="good" for v in decisions.values()),
                "bad":sum(v=="bad" for v in decisions.values()),
                "unsure":sum(v=="unsure" for v in decisions.values())}

    def first_unreviewed_index(self,decisions,after_index=None):
        total=len(self.items)
        if not total:return None
        start=0 if after_index is None else min(after_index+1,total)
        for idx in range(start,total):
            if int(self.items[idx]["image_id"]) not in decisions:return idx
        for idx in range(0,start):
            if int(self.items[idx]["image_id"]) not in decisions:return idx
        return None

    def state(self,index=None):
        decisions=self.store.all_decisions();total=len(self.items)
        if total==0:
            return {"total":0,"index":-1,"counts":self.counts(decisions),"complete":True,"item":None}
        if index is None:
            index=self.first_unreviewed_index(decisions)
            if index is None:
                return {"total":total,"index":total-1,"counts":self.counts(decisions),"complete":True,"item":None}
        index=max(0,min(int(index),total-1));item=self.items[index];image_id=int(item["image_id"])
        return {"total":total,"index":index,"counts":self.counts(decisions),"complete":False,
                "item":{"image_id":image_id,"prompt":item.get("prompt") or "",
                        "source_url":item.get("source_url") or "","decision":decisions.get(image_id)}}

    def save_and_next(self,image_id,decision):
        if image_id not in self.by_id:raise KeyError(f"Unknown image_id: {image_id}")
        self.store.save(image_id,decision)
        decisions=self.store.all_decisions();current=self.index_by_id[image_id]
        nxt=self.first_unreviewed_index(decisions,after_index=current)
        if nxt is None:
            return {"total":len(self.items),"index":current,"counts":self.counts(decisions),"complete":True,"item":None}
        return self.state(index=nxt)

    def image(self,image_id):
        item=self.by_id.get(image_id)
        if item is None:raise FileNotFoundError(f"Unknown image_id: {image_id}")
        raw=Path(str(item["local_image_path"]))
        path=(raw if raw.is_absolute() else self.validation_dir/raw).resolve()
        try:path.relative_to(self.validation_dir)
        except ValueError as exc:raise FileNotFoundError("Image path escapes validation directory") from exc
        if not path.exists():
            matches=list((self.validation_dir/"images").glob(f"{image_id}.*"))
            if not matches:raise FileNotFoundError(f"Image not found for image_id={image_id}")
            path=matches[0].resolve()
        content_type,_=mimetypes.guess_type(path.name)
        return path,content_type if content_type and content_type.startswith("image/") else "application/octet-stream"

def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version="CivitaiValidation/1.0"
        def log_message(self,*args):return
        def send_bytes(self,data,content_type,status=HTTPStatus.OK,cache_control="no-store"):
            self.send_response(status);self.send_header("Content-Type",content_type);self.send_header("Content-Length",str(len(data)));self.send_header("Cache-Control",cache_control);self.end_headers();self.wfile.write(data)
        def send_json(self,payload,status=HTTPStatus.OK):
            self.send_bytes(json.dumps(payload,ensure_ascii=False,separators=(",",":")).encode(),"application/json; charset=utf-8",status)
        def do_GET(self):
            p=urlparse(self.path)
            if p.path=="/":
                return self.send_bytes(HTML.encode(),"text/html; charset=utf-8")
            if p.path=="/api/state":
                try:
                    v=parse_qs(p.query).get("index",[None])[0];return self.send_json(app.state(index=None if v is None else int(v)))
                except (TypeError,ValueError) as exc:return self.send_json({"error":str(exc)},HTTPStatus.BAD_REQUEST)
            if p.path.startswith("/image/"):
                try:
                    image_id=int(unquote(p.path.removeprefix("/image/")));path,ctype=app.image(image_id);return self.send_bytes(path.read_bytes(),ctype,cache_control="private, max-age=3600")
                except (ValueError,FileNotFoundError,OSError) as exc:return self.send_json({"error":str(exc)},HTTPStatus.NOT_FOUND)
            self.send_json({"error":"Not found"},HTTPStatus.NOT_FOUND)
        def do_POST(self):
            if urlparse(self.path).path!="/api/review":
                return self.send_json({"error":"Not found"},HTTPStatus.NOT_FOUND)
            try:
                n=int(self.headers.get("Content-Length","0"));payload=json.loads(self.rfile.read(n).decode())
                image_id=int(payload["image_id"]);decision=str(payload["decision"]).strip().lower()
                if decision not in VALID_DECISIONS:raise ValueError("decision must be good, bad or unsure")
                return self.send_json(app.save_and_next(image_id,decision))
            except KeyError as exc:return self.send_json({"error":str(exc)},HTTPStatus.NOT_FOUND)
            except (ValueError,TypeError,json.JSONDecodeError) as exc:return self.send_json({"error":str(exc)},HTTPStatus.BAD_REQUEST)
            except sqlite3.Error as exc:return self.send_json({"error":f"Database error: {exc}"},HTTPStatus.INTERNAL_SERVER_ERROR)
    return Handler

def main(argv=None):
    p=argparse.ArgumentParser(description="Minimal GOOD/BAD/ХЗ review UI")
    p.add_argument("--validation-dir",type=Path,default=Path("dataset_miner/validation"))
    p.add_argument("--host",default=DEFAULT_HOST);p.add_argument("--port",type=int,default=DEFAULT_PORT)
    p.add_argument("--no-open",action="store_true")
    args=p.parse_args(argv)
    try:app=ReviewApp(args.validation_dir)
    except (OSError,ValueError) as exc:
        print(f"ERROR: {exc}",file=sys.stderr);return 2
    server=ThreadingHTTPServer((args.host,args.port),make_handler(app));url=f"http://{args.host}:{args.port}/"
    print("Civitai validation UI");print(f"validation_dir: {app.validation_dir}");print(f"manifest_items: {len(app.items)}")
    print(f"review_db: {app.store.db_path}");print(f"url: {url}");print("shortcuts: B=BAD, U=ХЗ, G=GOOD, ←/→ navigation");print("Ctrl+C to stop")
    if not args.no_open:threading.Timer(.35,lambda:webbrowser.open(url)).start()
    try:server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:print("\nStopped.")
    finally:server.server_close()
    return 0

if __name__=="__main__":
    raise SystemExit(main())
