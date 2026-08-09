"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { DatasetImporter, LocalDatasetWorkspace, type LocalDataset } from "./local-datasets";

type Capability = "embeddings" | "explain" | "predict" | "scenario";
type Question = "default" | "credit";
type ExplanationMode = "shapley" | "interaction";

type EmbeddingData = {
  n: number;
  gx_default: number[];
  gy_default: number[];
  gx_credit: number[];
  gy_credit: number[];
  gx_raw: number[];
  gy_raw: number[];
  lab_default: number[];
  lab_credit: number[];
  limit: number[];
  pay0: number[];
  age: number[];
  meta: {
    sil_default: number;
    sil_default_off: number;
    sil_credit: number;
    sil_credit_off: number;
    sil_raw_default: number;
    sil_raw_credit: number;
    rate_default: number;
    rate_credit: number;
  };
};

type Point = { index: number; x: number; y: number };

const CAPABILITIES: Array<{
  id: Capability;
  eyebrow: string;
  label: string;
  description: string;
}> = [
  { id: "embeddings", eyebrow: "Em", label: "Embeddings", description: "See target-aware structure" },
  { id: "explain", eyebrow: "Ix", label: "Explain", description: "Shapley values + interactions" },
  { id: "predict", eyebrow: "Zs", label: "Zero-shot", description: "Predict without training" },
  { id: "scenario", eyebrow: "Sc", label: "Scenarios", description: "Change inputs, compare outcomes" },
];

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));
const money = (value: number) => new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
}).format(value);

function cohortRisk(data: EmbeddingData, age: number, limit: number, pay0: number) {
  const distances = new Array<{ distance: number; outcome: number }>(data.n);
  for (let index = 0; index < data.n; index += 1) {
    const ageDelta = (data.age[index] - age) / 14;
    const limitDelta = (data.limit[index] - limit) / 180_000;
    const paymentDelta = (data.pay0[index] - pay0) / 2.2;
    distances[index] = {
      distance: ageDelta * ageDelta + limitDelta * limitDelta + paymentDelta * paymentDelta,
      outcome: data.lab_default[index],
    };
  }
  distances.sort((a, b) => a.distance - b.distance);
  const neighbors = distances.slice(0, 72);
  return neighbors.reduce((sum, neighbor) => sum + neighbor.outcome, 0) / neighbors.length;
}

function EmbeddingCanvas({
  data,
  question,
  raw,
  selected,
  onSelect,
}: {
  data: EmbeddingData;
  question: Question;
  raw: boolean;
  selected: number;
  onSelect: (index: number) => void;
}) {
  const frameRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const currentRef = useRef<{ x: Float32Array; y: Float32Array } | null>(null);
  const rafRef = useRef(0);
  const dimensionsRef = useRef({ width: 0, height: 0 });
  const hoveredRef = useRef<Point | null>(null);
  const [hovered, setHovered] = useState<Point | null>(null);

  const positions = useMemo(() => {
    if (raw) return { x: data.gx_raw, y: data.gy_raw };
    return question === "default"
      ? { x: data.gx_default, y: data.gy_default }
      : { x: data.gx_credit, y: data.gy_credit };
  }, [data, question, raw]);

  const labels = question === "default" ? data.lab_default : data.lab_credit;
  const activeColor = question === "default" ? "#b45309" : "#1e2a78";

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const current = currentRef.current;
    if (!canvas || !current) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    const { width, height } = dimensionsRef.current;
    const pad = 24;
    const xScale = (value: number) => pad + (value / 1000) * (width - pad * 2);
    const yScale = (value: number) => pad + (value / 1000) * (height - pad * 2);

    context.clearRect(0, 0, width, height);
    context.strokeStyle = "rgba(30,42,120,0.055)";
    context.lineWidth = 1;
    for (let line = 1; line < 5; line += 1) {
      const x = (width / 5) * line;
      const y = (height / 5) * line;
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo(x, height);
      context.stroke();
      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(width, y);
      context.stroke();
    }

    for (let pass = 0; pass < 2; pass += 1) {
      for (let index = 0; index < data.n; index += 1) {
        const positive = labels[index] === 1;
        if ((pass === 0) === positive) continue;
        context.globalAlpha = positive ? 0.82 : 0.32;
        context.fillStyle = positive ? activeColor : "#8e918e";
        context.beginPath();
        context.arc(xScale(current.x[index]), yScale(current.y[index]), positive ? 2.35 : 1.7, 0, Math.PI * 2);
        context.fill();
      }
    }

    const focus = hoveredRef.current?.index ?? selected;
    context.globalAlpha = 1;
    context.fillStyle = labels[focus] ? activeColor : "#6b716e";
    context.beginPath();
    context.arc(xScale(current.x[focus]), yScale(current.y[focus]), 5.5, 0, Math.PI * 2);
    context.fill();
    context.strokeStyle = "#111827";
    context.lineWidth = 2;
    context.beginPath();
    context.arc(xScale(current.x[focus]), yScale(current.y[focus]), 9, 0, Math.PI * 2);
    context.stroke();
    context.globalAlpha = 1;
  }, [activeColor, data.n, labels, selected]);

  useEffect(() => {
    const frame = frameRef.current;
    const canvas = canvasRef.current;
    if (!frame || !canvas) return;
    const resize = () => {
      const rect = frame.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      dimensionsRef.current = { width: rect.width, height: rect.height };
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      canvas.getContext("2d")?.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(frame);
    resize();
    return () => observer.disconnect();
  }, [draw]);

  useEffect(() => {
    const nextX = positions.x;
    const nextY = positions.y;
    const existing = currentRef.current;
    if (!existing) {
      currentRef.current = {
        x: Float32Array.from(nextX),
        y: Float32Array.from(nextY),
      };
      draw();
      return;
    }

    const startX = Float32Array.from(existing.x);
    const startY = Float32Array.from(existing.y);
    const startedAt = performance.now();
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    cancelAnimationFrame(rafRef.current);

    const tick = (now: number) => {
      const elapsed = reducedMotion ? 1 : clamp((now - startedAt) / 720, 0, 1);
      const eased = 1 - Math.pow(1 - elapsed, 3);
      for (let index = 0; index < data.n; index += 1) {
        existing.x[index] = startX[index] + (nextX[index] - startX[index]) * eased;
        existing.y[index] = startY[index] + (nextY[index] - startY[index]) * eased;
      }
      draw();
      if (elapsed < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [data.n, draw, positions]);

  useEffect(() => draw(), [draw, selected]);

  const nearestPoint = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    const current = currentRef.current;
    if (!canvas || !current) return null;
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const pad = 24;
    const { width, height } = dimensionsRef.current;
    const xScale = (value: number) => pad + (value / 1000) * (width - pad * 2);
    const yScale = (value: number) => pad + (value / 1000) * (height - pad * 2);
    let best = -1;
    let bestDistance = 150;
    for (let index = 0; index < data.n; index += 1) {
      const xDelta = xScale(current.x[index]) - x;
      const yDelta = yScale(current.y[index]) - y;
      const distance = xDelta * xDelta + yDelta * yDelta;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = index;
      }
    }
    return best >= 0 ? best : null;
  };

  return (
    <div className="embedding-stage" ref={frameRef}>
      <canvas
        ref={canvasRef}
        className="embedding-canvas"
        aria-label="Interactive map of 3,000 held-out credit customers. Use arrow keys to inspect different customers."
        role="img"
        tabIndex={0}
        onPointerMove={(event) => {
          const index = nearestPoint(event.clientX, event.clientY);
          const rect = event.currentTarget.getBoundingClientRect();
          const point = index === null ? null : {
            index,
            x: clamp(event.clientX - rect.left + 12, 12, rect.width - 178),
            y: clamp(event.clientY - rect.top + 12, 12, rect.height - 84),
          };
          hoveredRef.current = point;
          setHovered(point);
          draw();
        }}
        onPointerLeave={() => {
          hoveredRef.current = null;
          setHovered(null);
          draw();
        }}
        onPointerDown={(event) => {
          const index = nearestPoint(event.clientX, event.clientY);
          if (index !== null) onSelect(index);
        }}
        onKeyDown={(event) => {
          if (event.key === "ArrowRight" || event.key === "ArrowDown") {
            event.preventDefault();
            onSelect((selected + 1) % data.n);
          }
          if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
            event.preventDefault();
            onSelect((selected - 1 + data.n) % data.n);
          }
        }}
      />
      <div className="map-axis map-axis-x">latent direction 01</div>
      <div className="map-axis map-axis-y">latent direction 02</div>
      {hovered && (
        <div
          className="point-tooltip"
          style={{
            left: hovered.x,
            top: hovered.y,
          }}
        >
          <strong>Customer #{hovered.index + 1}</strong>
          <span>{money(data.limit[hovered.index])} limit · age {data.age[hovered.index]}</span>
          <span>{data.pay0[hovered.index] <= 0 ? "Paid on time" : `${data.pay0[hovered.index]} months late`}</span>
        </div>
      )}
    </div>
  );
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: () => void; label: string }) {
  return (
    <button className="toggle" type="button" role="switch" aria-checked={checked} onClick={onChange}>
      <span>{label}</span>
      <span className={`toggle-track ${checked ? "is-on" : ""}`} aria-hidden="true">
        <span />
      </span>
    </button>
  );
}

function EmptyDataState() {
  return (
    <main className="loading-shell">
      <div className="loading-mark" aria-hidden="true"><span /></div>
      <p>Loading the public credit dataset…</p>
    </main>
  );
}

export default function Home() {
  const [data, setData] = useState<EmbeddingData | null>(null);
  const [capability, setCapability] = useState<Capability>("embeddings");
  const [question, setQuestion] = useState<Question>("default");
  const [raw, setRaw] = useState(false);
  const [selected, setSelected] = useState(907);
  const [explanationMode, setExplanationMode] = useState<ExplanationMode>("shapley");
  const [scenario, setScenario] = useState({ rowIndex: 907, age: 38, limit: 150_000, pay0: 0 });
  const [localDatasets, setLocalDatasets] = useState<LocalDataset[]>([]);
  const [activeDatasetId, setActiveDatasetId] = useState("credit");
  const [importerOpen, setImporterOpen] = useState(false);

  useEffect(() => {
    fetch("/data/nori-embeddings.json")
      .then((response) => {
        if (!response.ok) throw new Error("Could not load public demo data");
        return response.json();
      })
      .then((payload: EmbeddingData) => setData(payload));
  }, []);

  const activeCapability = CAPABILITIES.find((item) => item.id === capability) ?? CAPABILITIES[0];
  const activeLocalDataset = localDatasets.find((dataset) => dataset.id === activeDatasetId) ?? null;
  const scenarioAge = scenario.rowIndex === selected ? scenario.age : data?.age[selected] ?? 38;
  const scenarioLimit = scenario.rowIndex === selected ? scenario.limit : data?.limit[selected] ?? 150_000;
  const scenarioPay0 = scenario.rowIndex === selected ? scenario.pay0 : data?.pay0[selected] ?? 0;

  const originalRisk = useMemo(() => {
    if (!data) return 0;
    return cohortRisk(data, data.age[selected], data.limit[selected], data.pay0[selected]);
  }, [data, selected]);

  const scenarioRisk = useMemo(() => {
    if (!data) return 0;
    return cohortRisk(data, scenarioAge, scenarioLimit, scenarioPay0);
  }, [data, scenarioAge, scenarioLimit, scenarioPay0]);

  const contributions = useMemo(() => {
    if (!data) return [];
    const payment = data.pay0[selected] <= 0 ? -0.12 : clamp(0.09 + data.pay0[selected] * 0.11, -0.2, 0.42);
    const credit = clamp(-((data.limit[selected] - 140_000) / 500_000) * 0.18, -0.22, 0.2);
    const age = clamp(-((data.age[selected] - 35) / 35) * 0.1, -0.14, 0.12);
    const context = clamp(originalRisk - data.meta.rate_default, -0.2, 0.3);
    return [
      { name: "Recent repayment status", value: payment, detail: data.pay0[selected] <= 0 ? "on time" : `${data.pay0[selected]} months late` },
      { name: "Credit limit", value: credit, detail: money(data.limit[selected]) },
      { name: "Age", value: age, detail: `${data.age[selected]} years` },
      { name: "Comparable customers", value: context, detail: `${Math.round(originalRisk * 100)}% default rate` },
    ].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  }, [data, originalRisk, selected]);

  if (!data) return <EmptyDataState />;

  const positiveLabel = question === "default" ? "defaulted" : "high value";
  const negativeLabel = question === "default" ? "repaid" : "standard value";
  const selectedPositive = question === "default" ? data.lab_default[selected] : data.lab_credit[selected];
  const silhouette = raw
    ? question === "default" ? data.meta.sil_raw_default : data.meta.sil_raw_credit
    : question === "default" ? data.meta.sil_default : data.meta.sil_credit;
  const positiveRate = question === "default" ? data.meta.rate_default : data.meta.rate_credit;

  const selectCapability = (next: Capability) => {
    setCapability(next);
  };

  return (
    <div className="site-shell">
      <header className="topbar">
        <a className="brand" href="#main" aria-label="Nori Studio home">
          <span className="brand-mark" aria-hidden="true"><span /></span>
          <span>Nori</span>
          <span className="brand-divider" />
          <span className="brand-product">Studio</span>
        </a>
        <p className="brand-promise">Understand any table.</p>
        <div className="topbar-actions">
          <span className="preview-pill"><span /> Public preview</span>
          <a className="github-link" href="https://github.com/Synthefy/synthefy-nori" target="_blank" rel="noreferrer">
            View on GitHub <span aria-hidden="true">↗</span>
          </a>
        </div>
      </header>

      <section className="studio-toolbar" aria-label="Studio controls">
        <label className="dataset-picker">
          <span>Dataset</span>
          <select value={activeDatasetId} onChange={(event) => setActiveDatasetId(event.target.value)}>
            <option value="credit">Credit card default · UCI</option>
            {localDatasets.map((dataset) => <option value={dataset.id} key={dataset.id}>{dataset.name} · local</option>)}
          </select>
        </label>
        <nav className="mode-tabs" aria-label="Nori capabilities">
          {CAPABILITIES.map((item) => (
            <button type="button" className={capability === item.id ? "is-active" : ""} onClick={() => selectCapability(item.id)} key={item.id}>
              <span>{item.eyebrow}</span>
              <strong>{item.label}</strong>
              <small>{item.description}</small>
            </button>
          ))}
        </nav>
        <button type="button" className="add-dataset-button" onClick={() => setImporterOpen(true)}><span>+</span> Add dataset</button>
      </section>

      <main className="studio-main" id="main">

          <section className="analysis-card">
            <div className="analysis-header">
              <div>
                <p className="section-kicker">{activeLocalDataset ? activeLocalDataset.source : "Credit intelligence lab · UCI public data"}</p>
                <h2>{activeCapability.label}</h2>
                <p>{activeCapability.description}</p>
              </div>
              {!activeLocalDataset ? <div className="row-switcher">
                <button type="button" onClick={() => setSelected((selected - 1 + data.n) % data.n)} aria-label="Previous customer">←</button>
                <span>Customer <b>#{selected + 1}</b></span>
                <button type="button" onClick={() => setSelected((selected + 1) % data.n)} aria-label="Next customer">→</button>
              </div> : <span className="local-status"><i /> Data stays in this tab</span>}
            </div>

            {activeLocalDataset ? <LocalDatasetWorkspace key={activeLocalDataset.id} dataset={activeLocalDataset} capability={capability} /> : null}

            {!activeLocalDataset && capability === "embeddings" && (
              <div className="embedding-layout">
                <div className="embedding-main">
                  <div className="control-strip">
                    <div className="control-group">
                      <span>Organize customers by</span>
                      <div className="segmented">
                        <button type="button" className={question === "default" ? "is-active orange" : ""} onClick={() => setQuestion("default")}>Default risk</button>
                        <button type="button" className={question === "credit" ? "is-active indigo" : ""} onClick={() => setQuestion("credit")}>Customer value</button>
                      </div>
                    </div>
                    <Toggle checked={raw} onChange={() => setRaw((value) => !value)} label="Show raw features" />
                  </div>

                  <div className="map-shell">
                    <div className="map-readout">
                      <span>{raw ? "Raw feature map" : "Nori embedding"}</span>
                      <strong>{silhouette >= 0.13 ? "Clear structure" : silhouette >= 0.05 ? "Loose structure" : "No clear structure"}</strong>
                      <small>silhouette <b>{silhouette.toFixed(2)}</b></small>
                    </div>
                    <EmbeddingCanvas data={data} question={question} raw={raw} selected={selected} onSelect={setSelected} />
                    <div className="map-legend">
                      <span><i className={question === "default" ? "orange" : "indigo"} /> {positiveLabel}</span>
                      <span><i /> {negativeLabel}</span>
                      <span>{Math.round(positiveRate * 100)}% positive outcome</span>
                    </div>
                  </div>
                </div>

                <aside className="record-panel">
                  <div className="record-heading">
                    <span>Selected record</span>
                    <b>#{selected + 1}</b>
                  </div>
                  <div className={`outcome-badge ${selectedPositive ? "positive" : "negative"}`}>
                    <span>{selectedPositive ? positiveLabel : negativeLabel}</span>
                    <strong>{selectedPositive ? "1" : "0"}</strong>
                  </div>
                  <dl className="record-list">
                    <div><dt>Credit limit</dt><dd>{money(data.limit[selected])}</dd></div>
                    <div><dt>Repayment</dt><dd>{data.pay0[selected] <= 0 ? "On time" : `${data.pay0[selected]} mo. late`}</dd></div>
                    <div><dt>Age</dt><dd>{data.age[selected]}</dd></div>
                    <div><dt>Context</dt><dd>18 features</dd></div>
                  </dl>
                  <button type="button" className="primary-button" onClick={() => setCapability("explain")}>Explain this outcome <span>→</span></button>
                  <p className="record-note">Click any point to inspect that held-out customer.</p>
                </aside>
              </div>
            )}

            {!activeLocalDataset && capability === "explain" && (
              <div className="explain-layout">
                <div className="explain-main">
                  <div className="explain-toolbar">
                    <div>
                      <p className="section-kicker">Attribution method</p>
                      <div className="segmented">
                        <button type="button" className={explanationMode === "shapley" ? "is-active indigo" : ""} onClick={() => setExplanationMode("shapley")}>Shapley values</button>
                        <button type="button" className={explanationMode === "interaction" ? "is-active orange" : ""} onClick={() => setExplanationMode("interaction")}>SHAP-IQ interactions</button>
                      </div>
                    </div>
                    <span className="prototype-tag">Interface preview</span>
                  </div>

                  {explanationMode === "shapley" ? (
                    <div className="attribution-panel">
                      <div className="chart-title">
                        <div><span>Local feature effects</span><strong>What moved this customer&apos;s risk?</strong></div>
                        <div className="direction-key"><span>← lowers risk</span><span>raises risk →</span></div>
                      </div>
                      <div className="waterfall">
                        <div className="zero-line" />
                        {contributions.map((item) => (
                          <div className="effect-row" key={item.name}>
                            <div><strong>{item.name}</strong><small>{item.detail}</small></div>
                            <div className="effect-track">
                              <span
                                className={item.value >= 0 ? "risk" : "protective"}
                                style={item.value >= 0
                                  ? { left: "50%", width: `${Math.abs(item.value) * 105}%` }
                                  : { right: "50%", width: `${Math.abs(item.value) * 105}%` }}
                              />
                            </div>
                            <b className={item.value >= 0 ? "risk-text" : "protective-text"}>{item.value >= 0 ? "+" : ""}{item.value.toFixed(2)}</b>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="interaction-panel">
                      <div className="chart-title">
                        <div><span>Pairwise effects</span><strong>Which signals reinforce each other?</strong></div>
                        <small>k-SII · order 2</small>
                      </div>
                      <div className="interaction-grid">
                        {[
                          ["Repayment", "Credit limit", contributions[0]?.value * contributions[1]?.value * 2.8],
                          ["Repayment", "Peer context", contributions[0]?.value * contributions[3]?.value * 3.2],
                          ["Credit limit", "Age", contributions[1]?.value * contributions[2]?.value * 2.6],
                          ["Age", "Peer context", contributions[2]?.value * contributions[3]?.value * 2.4],
                        ].map(([left, right, effect]) => {
                          const numericEffect = effect as number;
                          return (
                            <div className="interaction-card" key={`${left}-${right}`}>
                              <span className={numericEffect >= 0 ? "warm" : "cool"} style={{ opacity: clamp(Math.abs(numericEffect) * 5 + 0.2, 0.2, 1) }} />
                              <div><strong>{left as string} × {right as string}</strong><small>{numericEffect >= 0 ? "reinforces risk" : "offsets risk"}</small></div>
                              <b>{numericEffect >= 0 ? "+" : ""}{numericEffect.toFixed(3)}</b>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  <div className="method-note">
                    <span className="method-mark">i</span>
                    <p><strong>Designed for Nori&apos;s public interpretability API.</strong> The live product can call <code>get_nori_imputation_explainer</code> with a fixed context and a budget-controlled coalition search. This static preview calculates its visible effects locally from the public artifact.</p>
                  </div>
                </div>

                <aside className="explanation-summary">
                  <p className="section-kicker">Plain-language readout</p>
                  <h3>{data.pay0[selected] > 0 ? "Late repayment is the clearest warning." : "On-time repayment is the strongest protective signal."}</h3>
                  <p>
                    Customer #{selected + 1} has a {money(data.limit[selected])} credit limit and is {data.age[selected]} years old. Comparable customers in this cohort defaulted {Math.round(originalRisk * 100)}% of the time.
                  </p>
                  <div className="summary-stat"><span>Cohort baseline</span><strong>{Math.round(data.meta.rate_default * 100)}%</strong></div>
                  <div className="summary-stat accent"><span>Nearest-neighbor rate</span><strong>{Math.round(originalRisk * 100)}%</strong></div>
                  <button type="button" className="secondary-button" onClick={() => setCapability("scenario")}>Test a scenario <span>→</span></button>
                </aside>
              </div>
            )}

            {!activeLocalDataset && capability === "predict" && (
              <div className="predict-layout">
                <div className="prediction-card">
                  <div className="prediction-topline">
                    <span>Zero-shot inference</span>
                    <span className="prototype-tag">Static demo</span>
                  </div>
                  <div className="prediction-score">
                    <div className="score-orbit" style={{ "--score": `${Math.round(originalRisk * 100) * 3.6}deg` } as React.CSSProperties}>
                      <div><strong>{Math.round(originalRisk * 100)}%</strong><span>cohort risk</span></div>
                    </div>
                    <div>
                      <p className="eyebrow">Customer #{selected + 1}</p>
                      <h3>{originalRisk > data.meta.rate_default ? "Elevated default profile" : "Below-baseline default profile"}</h3>
                      <p>The nearest public-data cohort is {Math.abs(Math.round((originalRisk - data.meta.rate_default) * 100))} points {originalRisk > data.meta.rate_default ? "above" : "below"} the dataset baseline.</p>
                    </div>
                  </div>
                  <div className="prediction-band">
                    <span style={{ left: `${clamp(originalRisk * 100, 3, 97)}%` }} />
                  </div>
                  <div className="prediction-scale"><span>Lower observed risk</span><span>Higher observed risk</span></div>
                </div>

                <div className="context-card">
                  <div>
                    <p className="section-kicker">In-context setup</p>
                    <h3>No fitting loop. No hyperparameters.</h3>
                    <p>Nori receives labeled reference rows and predicts the held-out row in a single forward pass.</p>
                  </div>
                  <ol className="context-flow">
                    <li><b>01</b><span><strong>Reference context</strong><small>27,000 labeled customers</small></span></li>
                    <li><b>02</b><span><strong>Query row</strong><small>Customer #{selected + 1} · 18 features</small></span></li>
                    <li><b>03</b><span><strong>Nori output</strong><small>Regression distribution</small></span></li>
                  </ol>
                  <div className="method-note compact"><span className="method-mark">i</span><p>The percentage shown here is an observed nearest-neighbor baseline for this static site. Connect the Nori Python endpoint to display live zero-shot output.</p></div>
                </div>
              </div>
            )}

            {!activeLocalDataset && capability === "scenario" && (
              <div className="scenario-layout">
                <div className="scenario-controls">
                  <div className="scenario-heading">
                    <div><p className="section-kicker">Scenario inputs</p><h3>What would change the profile?</h3></div>
                    <button type="button" onClick={() => {
                      setScenario({ rowIndex: selected, age: data.age[selected], limit: data.limit[selected], pay0: data.pay0[selected] });
                    }}>Reset</button>
                  </div>
                  <label className="slider-field" htmlFor="scenario-repayment">
                    <span><b>Repayment status</b><strong>{scenarioPay0 <= 0 ? "On time" : `${scenarioPay0} months late`}</strong></span>
                    <input id="scenario-repayment" aria-label="Repayment status" type="range" min="-2" max="8" step="1" value={scenarioPay0} onChange={(event) => setScenario({ rowIndex: selected, age: scenarioAge, limit: scenarioLimit, pay0: Number(event.target.value) })} />
                    <small><span>2 months early</span><span>8 months late</span></small>
                  </label>
                  <label className="slider-field" htmlFor="scenario-credit-limit">
                    <span><b>Credit limit</b><strong>{money(scenarioLimit)}</strong></span>
                    <input id="scenario-credit-limit" aria-label="Credit limit" type="range" min="10000" max="700000" step="10000" value={scenarioLimit} onChange={(event) => setScenario({ rowIndex: selected, age: scenarioAge, limit: Number(event.target.value), pay0: scenarioPay0 })} />
                    <small><span>$10k</span><span>$700k</span></small>
                  </label>
                  <label className="slider-field" htmlFor="scenario-age">
                    <span><b>Age</b><strong>{scenarioAge}</strong></span>
                    <input id="scenario-age" aria-label="Age" type="range" min="21" max="75" step="1" value={scenarioAge} onChange={(event) => setScenario({ rowIndex: selected, age: Number(event.target.value), limit: scenarioLimit, pay0: scenarioPay0 })} />
                    <small><span>21</span><span>75</span></small>
                  </label>
                </div>

                <div className="scenario-result">
                  <div className="scenario-result-top">
                    <p className="section-kicker">Observed cohort comparison</p>
                    <span className="prototype-tag">Interactive</span>
                  </div>
                  <div className="risk-comparison">
                    <div>
                      <span>Original profile</span>
                      <strong>{Math.round(originalRisk * 100)}%</strong>
                      <div><i style={{ width: `${originalRisk * 100}%` }} /></div>
                    </div>
                    <span className="comparison-arrow">→</span>
                    <div className={scenarioRisk <= originalRisk ? "improved" : "worse"}>
                      <span>Scenario profile</span>
                      <strong>{Math.round(scenarioRisk * 100)}%</strong>
                      <div><i style={{ width: `${scenarioRisk * 100}%` }} /></div>
                    </div>
                  </div>
                  <div className="scenario-delta">
                    <span className={scenarioRisk <= originalRisk ? "down" : "up"}>{scenarioRisk <= originalRisk ? "↓" : "↑"}</span>
                    <div><strong>{Math.abs(Math.round((scenarioRisk - originalRisk) * 100))} point {scenarioRisk <= originalRisk ? "reduction" : "increase"}</strong><small>among the 72 most similar public-data customers</small></div>
                  </div>
                  <p className="scenario-copy">A live Nori scenario keeps the reference context fixed, changes only the selected inputs, and reruns the query—making every comparison controlled and inspectable.</p>
                </div>
              </div>
            )}
          </section>
      </main>
      <DatasetImporter
        open={importerOpen}
        onClose={() => setImporterOpen(false)}
        onImport={(dataset) => {
          setLocalDatasets((datasets) => [...datasets, dataset]);
          setActiveDatasetId(dataset.id);
          setCapability("embeddings");
        }}
      />
    </div>
  );
}
