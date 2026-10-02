import { useEffect, useState } from "react";
import Heart from "./Heart";
import Eval from "./Eval";

// Reference ranges: [low, high, unit]. Features not listed here just show no range.
const NORMAL = { BP: [90, 140, "mmHg"], PR: [60, 100, "bpm"], FBS: [70, 100, "mg/dL"], LDL: [0, 130, "mg/dL"],
  HDL: [40, 200, "mg/dL"], TG: [0, 150, "mg/dL"], "EF-TTE": [50, 70, "%"], BUN: [7, 20, "mg/dL"],
  ESR: [0, 20, "mm/h"], Cr: [0.6, 1.3, "mg/dL"], HB: [12, 17, "g/dL"], PLT: [150, 450, "x10^3/uL"],
  K: [3.5, 5.1, "mEq/L"], Na: [135, 145, "mEq/L"] };

const API = "http://localhost:8000";
// Quick-entry fields (only those present in /schema are shown)
const KEY = ["Age", "Sex", "BP", "PR", "HTN", "DM", "FBS", "LDL", "HDL", "TG", "Typical Chest Pain",
  "Atypical", "Tinversion", "St Depression", "EF-TTE", "Region RWMA"];
const j = (p, o) => fetch(API + p, o).then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.statusText))));

export default function App() {
  const [schema, setSchema] = useState(null);
  const [f, setF] = useState({});
  const [res, setRes] = useState(null);
  const [sel, setSel] = useState("CAD");
  const [idx, setIdx] = useState(0);
  const [metrics, setMetrics] = useState(null);
  const [err, setErr] = useState("");
  const [tab, setTab] = useState("dash");

  useEffect(() => {
    j("/schema").then((s) => setSchema(s.features)).catch((e) => setErr("API unreachable: " + e.message));
    j("/metrics").then((m) => setMetrics(m.best_models)).catch(() => {});
  }, []);

  // live prediction (debounced) whenever inputs change
  useEffect(() => {
    if (!schema) return;
    const t = setTimeout(() => {
      j("/predict", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: f, top_k: 8 }) })
        .then((r) => { setRes(r); setErr(""); }).catch((e) => setErr(e.message));
    }, 350);
    return () => clearTimeout(t);
  }, [f, schema]);

  const items = res ? { CAD: res.cad, ...res.vessels } : {};
  const cur = items[sel];
  const colors = res ? Object.fromEntries(Object.entries(res.vessels).map(([k, v]) => [k, v.color])) : {};
  const maxShare = cur ? Math.max(...cur.factors.map((x) => x.share_pct)) : 1;

  return (
    <>
      <div className="warn">⚠ Decision support / educational use only — not a substitute for formal diagnostic imaging or clinical judgement.</div>
      <div className="tabs">
        <button className={tab === "dash" ? "on" : ""} onClick={() => setTab("dash")}>Patient dashboard</button>
        <button className={tab === "eval" ? "on" : ""} onClick={() => setTab("eval")}>Model evaluation</button>
      </div>
      {tab === "eval" ? <Eval /> : <div className="grid">
        <aside className="card">
          <h3>Patient inputs</h3>
          <div className="row">
            <input type="number" min="0" value={idx} onChange={(e) => setIdx(e.target.value)} />
            <button onClick={() => j("/patients/" + idx).then((p) => setF(p.features))}>Load dataset patient</button>
          </div>
          {schema && KEY.filter((k) => schema[k]).map((k) => {
            const s = schema[k], v = f[k] ?? s.default;
            const set = (x) => setF({ ...f, [k]: x });
            return (
              <label key={k}>{k}
                {s.type === "categorical" || (s.min === 0 && s.max === 1)
                  ? <select value={v} onChange={(e) => set(s.options ? e.target.value : Number(e.target.value))}>{(s.options ?? [0, 1]).map((o) => <option key={o} value={o}>{s.options ? o : o ? "Yes" : "No"}</option>)}</select>
                  : <input type="number" step={s.integer ? 1 : "any"} min={s.min} max={s.max} value={v}
                      onChange={(e) => set(e.target.value === "" ? s.default : Number(e.target.value))} />}
              </label>
            );
          })}
          <p className="mut">Unset fields use dataset median/mode.</p>
        </aside>

        <main className="card canvas">
          <Heart colors={colors} pcts={res ? Object.fromEntries(Object.entries(res.vessels).map(([k, v]) => [k, Math.round(v.percent)])) : null} selected={sel} onSelect={setSel} />
          <div className="hint">Drag to rotate · scroll to zoom · click an artery</div>
        </main>

        <section className="card">
          {err && <p className="err">{err}</p>}
          <div className="cards">
            {Object.entries(items).map(([k, v]) => (
              <button key={k} className={"risk" + (sel === k ? " on" : "")} onClick={() => setSel(k)}
                style={{ borderColor: k === "CAD" ? undefined : v.color }}>
                <b>{k}</b><span className="pct">{v.percent}%</span>
                <small>{k === "CAD" ? res.cad.prediction : v.level}
                  {metrics?.[k] && ` · AUC ${metrics[k].roc_auc}`}{metrics?.[k]?.roc_auc < 0.75 && " · lower confidence"}</small>
              </button>
            ))}
          </div>
          {cur && (<>
            <h3>Why {sel}? <small className="mut">({cur.model})</small></h3>
            {cur.factors.map((x) => (
              <div key={x.feature} className="fac">
                {(() => { const n = NORMAL[x.feature]; const bad = n && (x.value < n[0] || x.value > n[1]);
                  return <div className="fl"><span>{x.feature} = {String(x.value)} {n && <small className={bad ? "bad" : "ok"}>
                    {bad ? "outside" : "within"} ref {n[0]}-{n[1]} {n[2]}</small>}</span><span>{x.share_pct}%</span></div>; })()}
                <div className="bar"><i style={{ width: (100 * x.share_pct) / maxShare + "%",
                  background: x.shap > 0 ? "#ef4444" : "#22c55e" }} /></div>
                <small className="mut">{x.direction}</small>
              </div>
            ))}
          </>)}
          {res?.warnings?.map((w) => <p key={w} className="err">{w}</p>)}
        </section>
      </div>}
    </>
  );
}