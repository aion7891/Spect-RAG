"""The HTML page used by graph_view.py. Self-contained: inline CSS and JavaScript, no downloads.
Colours: the six edge signals use a palette checked for colour-vision deficiency (light and dark);
every colour also has a text label in the legend, and the selected chunks are listed as text."""

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SpectRAG graph view</title>
<style>
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;
--s0:#2a78d6;--s1:#eb6834;--s2:#1baf7a;--s3:#eda100;--s4:#e87ba4;--s5:#008300;
--seed:#6250d6;--hop:#c3c2b7;--selected:#e34948;--plain:#a9a8a1}
@media (prefers-color-scheme: dark){:root{--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#898781;--line:#2c2c2a;--s0:#3987e5;--s1:#d95926;--s2:#199e70;--s3:#c98500;--s4:#d55181;--s5:#008300;
--seed:#9085e9;--hop:#52514e;--selected:#e66767;--plain:#6b6a65}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:14px -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line)}h1{font-size:16px;margin:0 0 4px}
.sub{color:var(--ink2);font-size:13px}main{display:flex;gap:12px;padding:12px 16px;flex-wrap:wrap}
#view{flex:1 1 560px;min-width:0;height:72vh;background:var(--surface);border:1px solid var(--line);border-radius:8px;touch-action:none}
aside{flex:0 1 320px;min-width:260px;display:flex;flex-direction:column;gap:12px}
.box{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.box h2{font-size:13px;margin:0 0 8px;color:var(--ink2);font-weight:600}label{display:flex;gap:6px;align-items:center;margin:3px 0;cursor:pointer}
.sw{width:12px;height:12px;border-radius:3px;display:inline-block}select{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:3px}
#tip{position:fixed;pointer-events:none;max-width:340px;background:var(--surface);color:var(--ink);border:1px solid var(--line);
border-radius:8px;padding:8px 10px;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.15);display:none;z-index:9}
#tip b{display:block;margin-bottom:2px}#tip .t{color:var(--ink2);white-space:pre-wrap}
ol{margin:0;padding-left:20px;max-height:28vh;overflow:auto}li{margin:2px 0;cursor:pointer}li:hover{text-decoration:underline}
.n{stroke:var(--surface);stroke-width:1.5}.e{stroke-linecap:round}.dim{opacity:.12}
@media (max-width:700px){#view{height:60vh}}
</style></head><body>
<header><h1>SpectRAG chunk graph</h1><div class="sub" id="sub"></div></header>
<main><svg id="view" role="img" aria-label="Chunk graph"><g id="zoom"><g id="edges"></g><g id="nodes"></g></g></svg>
<aside><div class="box"><h2>Colour nodes by</h2><select id="mode"><option value="state">retrieval path</option>
<option value="doc">document</option><option value="cluster">spectral cluster</option></select><div id="nodeLegend" style="margin-top:8px"></div></div>
<div class="box"><h2>Edges (strongest signal)</h2><div id="edgeLegend"></div></div>
<div class="box" id="pathBox"><h2>Selected context (in pick order)</h2><ol id="picked"></ol></div></aside></main>
<div id="tip"></div>
<script>
const D = __DATA__;
const NS = "http://www.w3.org/2000/svg", css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const byId = {}; D.nodes.forEach(n => byId[n.id] = n);
const neighbours = {}; D.edges.forEach(([i, j]) => { (neighbours[i] ||= new Set()).add(j); (neighbours[j] ||= new Set()).add(i); });
const shown = new Set(D.signals);
document.getElementById("sub").textContent = `${D.info.documents} documents · ${D.info.chunks} chunks · ${D.info.graph_edges} edges` +
  (D.nodes.length < D.info.chunks ? ` · showing ${D.nodes.length} chunks` : "") + (D.question ? ` · question: "${D.question}"` : "");
const commonest = (key, n) => Object.entries(D.nodes.reduce((c, x) => (c[x[key]] = (c[x[key]] || 0) + 1, c), {}))
  .sort((a, b) => b[1] - a[1]).slice(0, n).map(e => e[0]);
const groups = {doc: commonest("doc", 7), cluster: commonest("cluster", 7)};   // 7 colours + "other"
const STATE = {selected: ["--selected", "selected context"], seed: ["--seed", "seed (stage 1)"], hop: ["--hop", "reached by expansion"], "": ["--plain", "not retrieved"]};
function colour(n) {
  const mode = document.getElementById("mode").value;
  if (mode === "state") return css(STATE[n.state][0]);
  const k = groups[mode].indexOf(String(n[mode])); return k < 0 ? css("--plain") : css("--s" + (k % 6));
}
function nodeLegend() {
  const mode = document.getElementById("mode").value, el = document.getElementById("nodeLegend");
  const rows = mode === "state" ? Object.values(STATE).map(([v, t]) => [css(v), t])
    : groups[mode].slice(0, 6).map((g, k) => [css("--s" + k), (mode === "cluster" ? "cluster " : "") + g]).concat([[css("--plain"), "other"]]);
  el.innerHTML = rows.map(([c, t]) => `<label><span class="sw" style="background:${c}"></span>${t}</label>`).join("");
}
function edgeLegend() {
  document.getElementById("edgeLegend").innerHTML = D.signals.map((s, k) =>
    `<label><input type="checkbox" checked data-s="${s}"><span class="sw" style="background:${css("--s" + k)}"></span>${s}` +
    ` <span style="color:var(--muted)">(${D.edges.filter(e => e[3] === s).length})</span></label>`).join("");
  document.querySelectorAll("#edgeLegend input").forEach(b => b.onchange = () => { b.checked ? shown.add(b.dataset.s) : shown.delete(b.dataset.s); draw(); });
}
let focus = null;
function draw() {
  const E = document.getElementById("edges"), N = document.getElementById("nodes"); E.innerHTML = N.innerHTML = "";
  const maxDeg = Math.max(...D.nodes.map(n => n.degree), 1e-9), r = D.nodes.length > 400 ? 4 : 7;
  D.edges.forEach(([i, j, w, s]) => { if (!shown.has(s)) return; const a = byId[i], b = byId[j], l = document.createElementNS(NS, "line");
    l.setAttribute("x1", a.x); l.setAttribute("y1", a.y); l.setAttribute("x2", b.x); l.setAttribute("y2", b.y);
    l.setAttribute("class", "e" + (focus !== null && i !== focus && j !== focus ? " dim" : ""));
    l.setAttribute("stroke", css("--s" + D.signals.indexOf(s))); l.setAttribute("stroke-width", 0.6 + 2.5 * w); l.setAttribute("stroke-opacity", .35 + .5 * w); E.appendChild(l); });
  D.nodes.forEach(n => { const c = document.createElementNS(NS, "circle"); c.setAttribute("cx", n.x); c.setAttribute("cy", n.y);
    c.setAttribute("r", r * (0.7 + 0.8 * Math.sqrt(n.degree / maxDeg)) * (n.state === "selected" ? 1.35 : 1));
    c.setAttribute("fill", colour(n)); c.setAttribute("class", "n" + (focus !== null && n.id !== focus && !(neighbours[focus] || new Set()).has(n.id) ? " dim" : ""));
    c.onmousemove = ev => tip(ev, n); c.onmouseleave = () => tipEl.style.display = "none";
    c.onclick = ev => { ev.stopPropagation(); focus = focus === n.id ? null : n.id; draw(); }; N.appendChild(c); });
}
const tipEl = document.getElementById("tip");
function tip(ev, n) {
  const state = STATE[n.state][1];
  tipEl.innerHTML = `<b>${n.doc} · chunk ${n.id}</b>${n.section ? n.section + "<br>" : ""}cluster ${n.cluster} · ` +
    `weighted degree ${n.degree} · ${state}<div class="t">${n.text.replace(/</g, "&lt;")}…</div>`;
  tipEl.style.display = "block"; tipEl.style.left = Math.min(ev.clientX + 14, innerWidth - 360) + "px"; tipEl.style.top = (ev.clientY + 14) + "px";
}
const svg = document.getElementById("view"), Z = document.getElementById("zoom"); let view = {x: -40, y: -40, w: 1080, h: 1080}, drag = null;
const setView = () => svg.setAttribute("viewBox", `${view.x} ${view.y} ${view.w} ${view.h}`); setView();
svg.onwheel = ev => { ev.preventDefault(); const f = ev.deltaY > 0 ? 1.15 : 1 / 1.15, b = svg.getBoundingClientRect();
  const px = view.x + (ev.clientX - b.left) / b.width * view.w, py = view.y + (ev.clientY - b.top) / b.height * view.h;
  view = {x: px - (px - view.x) * f, y: py - (py - view.y) * f, w: view.w * f, h: view.h * f}; setView(); };
svg.onpointerdown = ev => drag = {x: ev.clientX, y: ev.clientY, v: {...view}};
svg.onpointermove = ev => { if (!drag) return; const b = svg.getBoundingClientRect();
  view.x = drag.v.x - (ev.clientX - drag.x) / b.width * view.w; view.y = drag.v.y - (ev.clientY - drag.y) / b.height * view.h; setView(); };
svg.onpointerup = svg.onpointerleave = () => drag = null; svg.onclick = () => { focus = null; draw(); };
document.getElementById("mode").onchange = () => { nodeLegend(); draw(); };
const list = document.getElementById("picked");
if (!D.picked.length) document.getElementById("pathBox").innerHTML = '<h2>Selected context</h2><div class="sub">Give a question to see the retrieval path.</div>';
D.picked.forEach(p => { const li = document.createElement("li"); li.textContent = `${p.doc} (chunk ${p.id}) · ${p.reason} · score ${p.score}`;
  li.onclick = () => { focus = p.id; draw(); }; list.appendChild(li); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { nodeLegend(); edgeLegend(); draw(); });
nodeLegend(); edgeLegend(); draw();
</script></body></html>
"""
