import { useEffect, useRef, useState } from "react";
import Heart from "./Heart";
import Eval from "./Eval";

const NORMAL = { BP: [90, 140, "mmHg"], PR: [60, 100, "bpm"], FBS: [70, 100, "mg/dL"], LDL: [0, 130, "mg/dL"],
  HDL: [40, 200, "mg/dL"], TG: [0, 150, "mg/dL"], "EF-TTE": [50, 70, "%"], BUN: [7, 20, "mg/dL"],
  ESR: [0, 20, "mm/h"], Cr: [0.6, 1.3, "mg/dL"], HB: [12, 17, "g/dL"], PLT: [150, 450, "x10^3/uL"],
  K: [3.5, 5.1, "mEq/L"], Na: [135, 145, "mEq/L"] };

const API = "http://localhost:8000";
// Input groups (only fields present in /schema are shown)
const GROUPS = [
  { title: "Demographics", keys: ["Age", "Sex"] },
  { title: "Vitals & history", keys: ["BP", "PR", "HTN", "DM"] },
  { title: "Lipids & glucose", keys: ["FBS", "LDL", "HDL", "TG"] },
  { title: "Symptoms & ECG", keys: ["Typical Chest Pain", "Atypical", "Tinversion", "St Depression"] },
  { title: "Echocardiography", keys: ["EF-TTE", "Region RWMA"] },
];
const NAV = [
  { id: "dash", label: "Patient dashboard", icon: "M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z" },
  { id: "eval", label: "Model evaluation", icon: "M5 9.2h3V19H5V9.2zM10.6 5h2.8v14h-2.8V5zm5.6 8H19v6h-2.8v-6z" },
  { id: "about", label: "About & safety", icon: "M12 2a10 10 0 100 20 10 10 0 000-20zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" },
];
const j = (p, o) => fetch(API + p, o).then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.statusText))));

const Icon = ({ d }) => <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d={d} /></svg>;

/* Generic dropdown menu: items = [{ label, value, hint?, onSelect }] */
function Menu({ label, value, items, align = "left" }) {
  const [open, setOpen] = useState(false);
  const ref = useRef();
  useEffect(() => {
    const h = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    const k = (e) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", h); document.addEventListener("keydown", k);
    return () => { document.removeEventListener("mousedown", h); document.removeEventListener("keydown", k); };
  }, []);
  return (
    <div className="menu" ref={ref}>
      <button className="btn" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(!open)}>
        {label && <span className="mut">{label}</span>}{value && <b>{value}</b>}
        <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M7 10l5 5 5-5z" /></svg>
      </button>
      {open && (
        <div className={"pop " + align} role="menu">
          {items.map((it) => (
            <button key={it.label} role="menuitem" className={it.value !== undefined && it.value === value ? "sel" : ""}
              onClick={() => { it.onSelect(); setOpen(false); }}>
              <span>{it.label}</span>{it.hint && <small>{it.hint}</small>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

const levelClass = (p) => (p >= 66 ? "hi" : p >= 33 ? "mid" : "lo");

export default function App() {
  const [schema, setSchema] = useState(null);
  const [f, setF] = useState({});
  const [res, setRes] = useState(null);
  const [sel, setSel] = useState("CAD");
  const [idx, setIdx] = useState(0);
  const [metrics, setMetrics] = useState(null);
  const [err, setErr] = useState("");
  const [tab, setTab] = useState("dash");
  const [topK, setTopK] = useState(8);
  const [navOpen, setNavOpen] = useState(false);
  const [loaded, setLoaded] = useState(null);
  const [auto, setAuto] = useState(false);     // live update on every edit
  const [dirty, setDirty] = useState(false);   // inputs changed since last prediction
  const [busy, setBusy] = useState(false);
  const ctlRef = useRef(null);

  useEffect(() => {
    j("/schema").then((s) => setSchema(s.features)).catch((e) => setErr("API unreachable: " + e.message));
    j("/metrics").then((m) => setMetrics(m.best_models)).catch(() => {});
  }, []);

  const runPredict = (features = f) => {
    ctlRef.current?.abort();                    // drop any in-flight request (no out-of-order results)
    const ctl = new AbortController();
    ctlRef.current = ctl;
    setBusy(true);
    j("/predict", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features, top_k: topK }), signal: ctl.signal })
      .then((r) => { setRes(r); setErr(""); setDirty(false); })
      .catch((e) => e.name !== "AbortError" && setErr(e.message))
      .finally(() => { if (ctlRef.current === ctl) setBusy(false); });
  };

  // first prediction once the schema is ready, and again when the factor count changes
  useEffect(() => { if (schema) runPredict(); }, [schema, topK]); // eslint-disable-line react-hooks/exhaustive-deps
  // live mode: predict 350 ms after the last edit
  useEffect(() => {
    if (!schema || !auto || !dirty) return;
    const t = setTimeout(() => runPredict(), 350);
    return () => clearTimeout(t);
  }, [f, auto, dirty, schema]);                                   // eslint-disable-line react-hooks/exhaustive-deps

  const edit = (next) => { setF(next); setDirty(true); };
  const loadPatient = (i) => {
    const n = Math.max(0, Number(i) || 0);
    setIdx(n);
    j("/patients/" + n)
      .then((p) => { setF(p.features); setLoaded(n); setErr(""); runPredict(p.features); })
      .catch((e) => setErr("Could not load patient " + n + ": " + e.message));
  };
  const reset = () => { setF({}); setLoaded(null); runPredict({}); };

  const items = res ? { CAD: res.cad, ...res.vessels } : {};
  const cur = items[sel];
  const colors = res ? Object.fromEntries(Object.entries(res.vessels).map(([k, v]) => [k, v.color])) : {};
  const maxShare = cur ? Math.max(...cur.factors.map((x) => x.share_pct)) : 1;
  const current = NAV.find((n) => n.id === tab);

  const renderField = (k) => {
    const s = schema[k], v = f[k] ?? s.default;
    const set = (x) => edit({ ...f, [k]: x });
    const n = NORMAL[k];
    const isSel = s.type === "categorical" || (s.min === 0 && s.max === 1);
    return (
      <label key={k}>
        <span className="lbl">{k}{n && <small>{n[2]}</small>}</span>
        {isSel
          ? <select value={v} onChange={(e) => set(s.options ? e.target.value : Number(e.target.value))}>
              {(s.options ?? [0, 1]).map((o) => <option key={o} value={o}>{s.options ? o : o ? "Yes" : "No"}</option>)}
            </select>
          : <input type="number" step={s.integer ? 1 : "any"} min={s.min} max={s.max} value={v}
              onChange={(e) => set(e.target.value === "" ? s.default : Number(e.target.value))} />}
      </label>
    );
  };

  return (
    <div className={"shell" + (navOpen ? " nav-open" : "")}>
      <aside className="side">
        <div className="brand">
          <svg width="26" height="26" viewBox="0 0 24 24" fill="#38bdf8"><path d="M12 21s-7-4.6-9.3-9.1C1 8.4 2.9 5 6.2 5c1.9 0 3.2 1 3.8 2.2h.0C10.6 6 11.9 5 13.8 5c3.3 0 5.2 3.4 3.5 6.9C19 16.4 12 21 12 21z" /></svg>
          <div><b>CardioRisk 3D</b><small>Coronary risk support</small></div>
        </div>
        <nav>
          <small className="sec">Workspace</small>
          {NAV.map((n) => (
            <button key={n.id} className={"nav" + (tab === n.id ? " on" : "")} onClick={() => { setTab(n.id); setNavOpen(false); }}>
              <Icon d={n.icon} />{n.label}
            </button>
          ))}
        </nav>
        <div className="side-foot">
          <span className={"dot" + (err.startsWith("API") ? " off" : "")} />
          {err.startsWith("API") ? "API offline" : "API connected"}
          <small>{API}</small>
        </div>
      </aside>

      <div className="main">
        <header className="top">
          <button className="btn burger" aria-label="Toggle menu" onClick={() => setNavOpen(!navOpen)}>☰</button>
          <div className="crumb"><small>Workspace</small><h1>{current.label}</h1></div>
          {tab === "dash" && (
            <div className="tools">
              <div className="pick">
                <button className="btn" aria-label="Previous patient" onClick={() => loadPatient(idx - 1)}>‹</button>
                <input type="number" min="0" value={idx} aria-label="Dataset patient index"
                  onChange={(e) => setIdx(e.target.value)} onKeyDown={(e) => e.key === "Enter" && loadPatient(idx)} />
                <button className="btn" aria-label="Next patient" onClick={() => loadPatient(Number(idx) + 1)}>›</button>
                <button className="btn primary" onClick={() => loadPatient(idx)}>Load patient</button>
              </div>
              <Menu label="Focus" value={sel} align="right"
                items={Object.keys(items).length ? Object.keys(items).map((k) => ({ label: k, value: k, onSelect: () => setSel(k),
                  hint: items[k].percent + "%" })) : [{ label: "CAD", value: "CAD", onSelect: () => setSel("CAD") }]} />
              <Menu label="Factors" value={String(topK)} align="right"
                items={[5, 8, 12].map((n) => ({ label: "Top " + n, value: String(n), onSelect: () => setTopK(n) }))} />
              <Menu label="Actions" align="right" items={[
                { label: "Predict now", onSelect: () => runPredict() },
                { label: "Reset to defaults", onSelect: reset },
                { label: "Load random patient", onSelect: () => loadPatient(Math.floor(Math.random() * 300)) },
                { label: "Print report", onSelect: () => window.print() },
              ]} />
            </div>
          )}
        </header>

        <div className="warn">⚠ Decision support / educational use only. Not a substitute for formal diagnostic imaging or clinical judgement.</div>

        {tab === "eval" && <Eval />}

        {tab === "about" && (
          <div className="page"><section className="card prose">
            <h3>About this tool</h3>
            <p>CardioRisk 3D estimates the probability of coronary artery disease and per-vessel stenosis (LAD, LCX, RCA) from routine clinical features, and explains each prediction with SHAP-based factor contributions.</p>
            <h3>Intended use</h3>
            <p>Educational and decision-support only. Predictions must not replace angiography, CT or the judgement of a qualified clinician.</p>
            <h3>Reading the results</h3>
            <p>Vessel colours run from green (low) to red (high predicted stenosis probability). Vessels whose model AUC is below 0.75 are flagged as lower confidence. Unset inputs fall back to dataset median or mode.</p>
          </section></div>
        )}

        {tab === "dash" && (
          <div className="grid">
            <aside className="card inputs">
              <div className="card-h">
                <h3>Patient inputs</h3>
                <span className={"chip" + (loaded !== null ? " on" : "")}>{loaded !== null ? "Dataset #" + loaded : "Manual entry"}</span>
              </div>
              {!schema && <p className="mut">Waiting for the API…</p>}
              {schema && GROUPS.map((g, i) => {
                const keys = g.keys.filter((k) => schema[k]);
                return keys.length ? (
                  <details key={g.title} open={i < 2}>
                    <summary>{g.title}</summary>
                    <div className="fields">{keys.map(renderField)}</div>
                  </details>
                ) : null;
              })}
              <p className="mut note">Unset fields use dataset median/mode.</p>
              <div className="actions">
                <button className="btn primary wide" disabled={!schema || busy} onClick={() => runPredict()}>
                  {busy ? "Predicting…" : dirty ? "Predict ●" : "Predict"}
                </button>
                <button className="btn" disabled={!schema} onClick={reset}>Reset</button>
                <label className="switch">
                  <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} />
                  <span>Live update</span>
                </label>
              </div>
            </aside>

            <main className="card canvas">
              <Heart colors={colors} pcts={res ? Object.fromEntries(Object.entries(res.vessels).map(([k, v]) => [k, Math.round(v.percent)])) : null}
                selected={sel} onSelect={setSel} />
              <div className="hint">Drag to rotate · scroll to zoom · click an artery</div>
            </main>

            <section className={"card results" + (dirty && !auto ? " stale-on" : "")}>
              {err && <p className="err">{err}</p>}
              {dirty && !auto && <p className="stale">Inputs changed. Click <b>Predict</b> to update the results.</p>}
              <div className="cards">
                {Object.entries(items).map(([k, v]) => (
                  <button key={k} className={"risk " + levelClass(v.percent) + (sel === k ? " on" : "")} onClick={() => setSel(k)}
                    style={k === "CAD" ? undefined : { "--vc": v.color }}>
                    <span className="rk">{k}</span>
                    <span className="pct">{v.percent}<em>%</em></span>
                    <small>{k === "CAD" ? res.cad.prediction : v.level}
                      {metrics?.[k] && ` · AUC ${metrics[k].roc_auc}`}</small>
                    {metrics?.[k]?.roc_auc < 0.75 && <span className="chip warnchip">Lower confidence</span>}
                  </button>
                ))}
              </div>
              {cur && (<>
                <div className="card-h"><h3>Why {sel}?</h3><small className="mut">{cur.model}</small></div>
                {cur.factors.map((x) => {
                  const n = NORMAL[x.feature];
                  const bad = n && (x.value < n[0] || x.value > n[1]);
                  return (
                    <div key={x.feature} className="fac">
                      <div className="fl">
                        <span>{x.feature} = {String(x.value)} {n && <small className={bad ? "bad" : "ok"}>
                          {bad ? "outside" : "within"} ref {n[0]}-{n[1]} {n[2]}</small>}</span>
                        <span>{x.share_pct}%</span>
                      </div>
                      <div className="bar"><i style={{ width: (100 * x.share_pct) / maxShare + "%", background: x.shap > 0 ? "#ef4444" : "#22c55e" }} /></div>
                      <small className="mut">{x.direction}</small>
                    </div>
                  );
                })}
              </>)}
              {res?.warnings?.map((w) => <p key={w} className="err">{w}</p>)}
            </section>
          </div>
        )}
      </div>
    </div>
  );
}