import { useEffect, useState } from "react";

const API = "http://127.0.0.1:8000";
const get = (p) => fetch(API + p).then((r) => r.json());
const COLS = [["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"], ["roc_auc", "ROC-AUC"]];

export default function Eval() {
  const [m, setM] = useState(null);
  const [t, setT] = useState("CAD");
  const [imp, setImp] = useState([]);
  const [ev, setEv] = useState(null);
  const [op, setOp] = useState("default");
  useEffect(() => { get("/metrics").then(setM).catch(() => {}); get("/evaluation").then((d) => d.nested && setEv(d)).catch(() => {}); }, []);
  useEffect(() => { get(`/importance/${t}?top_k=12`).then((r) => setImp(r.features)).catch(() => {}); }, [t]);
  const max = imp[0]?.mean_abs_shap || 1;
  return (
    <div className="eval">
      <section className="card">
        <h3>Cross-validated performance <small className="mut">({m?.cv})</small></h3>
        <table>
          <thead><tr><th>Target</th><th>Model</th>{COLS.map(([, l]) => <th key={l}>{l}</th>)}</tr></thead>
          <tbody>
            {m?.cv_results.map((r) => {
              const best = m.best_models[r.target]?.model === r.model;
              return (
                <tr key={r.target + r.model} className={best ? "best" : ""}>
                  <td>{r.target}</td><td>{r.model}{best && " *"}</td>
                  {COLS.map(([c]) => <td key={c}>{r[c].toFixed(3)} <small className="mut">±{r[c + "_sd"].toFixed(2)}</small></td>)}
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="mut">* = model selected and deployed for that target (highest ROC-AUC). Mean ± SD over 15 folds.</p>
      </section>
      {ev ? <Extra ev={ev} m={m} t={t} setT={setT} op={op} setOp={setOp} /> :
        <section className="card"><h3>Stricter evaluation</h3><p className="mut">Run <code>python backend/evaluate.py</code>, then refresh.</p></section>}
      <section className="card">
        <h3>Global feature importance (mean |SHAP|)</h3>
        <div className="row">
          {["CAD", "LAD", "LCX", "RCA"].map((x) => <button key={x} className={"pill" + (t === x ? " on" : "")} onClick={() => setT(x)}>{x}</button>)}
        </div>
        {imp.map((x) => (
          <div key={x.feature} className="fac">
            <div className="fl"><span>{x.feature}</span><span>{x.mean_abs_shap}</span></div>
            <div className="bar"><i style={{ width: (100 * x.mean_abs_shap) / max + "%", background: "#3b82f6" }} /></div>
          </div>
        ))}
      </section>
    </div>
  );
}

const COLORS = { CAD: "#60a5fa", LAD: "#f97316", LCX: "#facc15", RCA: "#c084fc" };
const OPS = { default: "Default 0.5", sens90: "Screening (recall >= 90%)", youden: "Balanced (Youden J)" };

function Extra({ ev, m, t, setT, op, setOp }) {
  const W = 260, pad = 30, xy = (f, v) => [pad + f * (W - 2 * pad), W - pad - v * (W - 2 * pad)];
  const pt = ev.operating_points[t][op];
  return (<>
    <section className="card">
      <h3>Honest estimate: nested cross-validation</h3>
      <table><thead><tr><th>Target</th><th>Model-selection CV AUC</th><th>Nested AUC</th><th>Optimism</th><th>Models chosen in folds</th></tr></thead>
        <tbody>{Object.entries(ev.nested).map(([k, n]) => {
          const sel = m?.best_models[k]?.roc_auc;
          return (<tr key={k}><td>{k}</td><td>{sel}</td><td>{n.auc_mean} <small className="mut">±{n.auc_sd}</small></td>
            <td>{sel != null ? (sel - n.auc_mean).toFixed(3) : "-"}</td>
            <td><small>{Object.entries(n.selected).map(([a, c]) => `${a} x${c}`).join(", ")}</small></td></tr>);
        })}</tbody></table>
      <p className="mut">The model is chosen inside each training fold and scored on unseen data, so the nested AUC is the less optimistic figure ({ev.repeats}x repeated 5-fold).</p>
    </section>
    <section className="card">
      <h3>ROC curves <small className="mut">(out-of-fold, calibrated)</small></h3>
      <svg viewBox={`0 0 ${W} ${W}`} width="100%" style={{ maxWidth: 340 }}>
        <rect x={pad} y={pad} width={W - 2 * pad} height={W - 2 * pad} fill="none" stroke="#22304a" />
        <line x1={pad} y1={W - pad} x2={W - pad} y2={pad} stroke="#22304a" strokeDasharray="4" />
        {Object.entries(ev.roc).map(([k, r]) => <polyline key={k} fill="none" stroke={COLORS[k]} strokeWidth={t === k ? 2.5 : 1.3}
          points={r.fpr.map((f, i) => xy(f, r.tpr[i]).join(",")).join(" ")} />)}
        <text x={W / 2} y={W - 6} fill="#8b98b0" fontSize="9" textAnchor="middle">False positive rate</text>
        <text x="8" y={W / 2} fill="#8b98b0" fontSize="9" transform={`rotate(-90 8 ${W / 2})`} textAnchor="middle">True positive rate</text>
      </svg>
      <div>{Object.entries(ev.roc).map(([k, r]) => <span key={k} style={{ color: COLORS[k], marginRight: 10 }}>{k} AUC {r.auc}</span>)}</div>
    </section>
    <section className="card">
      <h3>Confusion matrix &amp; operating point</h3>
      <div className="row">{["CAD", "LAD", "LCX", "RCA"].map((x) => <button key={x} className={"pill" + (t === x ? " on" : "")} onClick={() => setT(x)}>{x}</button>)}</div>
      <div className="row" style={{ marginTop: 6 }}>{Object.entries(OPS).map(([k, l]) => <button key={k} className={"pill" + (op === k ? " on" : "")} onClick={() => setOp(k)}>{l}</button>)}</div>
      <p>Threshold <b>{pt.threshold}</b></p>
      <div className="cm"><span /><b>Pred. no</b><b>Pred. yes</b>
        <b>Actual no</b><i className="ok">{pt.tn}</i><i className="bad">{pt.fp}</i>
        <b>Actual yes</b><i className="bad">{pt.fn}</i><i className="ok">{pt.tp}</i></div>
      <p className="mut">Accuracy {pt.accuracy} · Precision {pt.precision} · Recall {pt.recall} · Specificity {pt.specificity} · F1 {pt.f1}</p>
    </section>
  </>);
}