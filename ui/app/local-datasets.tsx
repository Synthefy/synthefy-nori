"use client";

import { useMemo, useRef, useState } from "react";

export type LocalDataset = {
  id: string;
  name: string;
  source: string;
  headers: string[];
  rows: string[][];
  numericColumns: string[];
  target: string;
  trainIndices: number[];
  testIndices: number[];
  splitSeed: number;
};

export type StarterDataset = {
  id: string;
  name: string;
  description: string;
  path: string;
  rows: string;
  features: string;
  target: string;
  task: "Classification" | "Regression";
  glyph: string;
};

export const STARTER_DATASETS: StarterDataset[] = [
  {
    id: "customer-lifetime-value",
    name: "Customer Lifetime Value",
    description: "Model the next 12 months of customer value from orders, loyalty, and engagement.",
    path: "/data/retail/customer-lifetime-value.csv",
    rows: "1,050",
    features: "10",
    target: "lifetime_value_12m",
    task: "Regression",
    glyph: "Lv",
  },
  {
    id: "customer-churn",
    name: "Customer Churn",
    description: "Find customers likely to lapse using recency, purchase, return, and service behavior.",
    path: "/data/retail/customer-churn.csv",
    rows: "1,000",
    features: "10",
    target: "churned_90d",
    task: "Classification",
    glyph: "Ch",
  },
  {
    id: "customer-conversion",
    name: "Customer Conversion",
    description: "Predict purchase conversion from sessions, product interest, carts, and channel.",
    path: "/data/retail/customer-conversion.csv",
    rows: "1,200",
    features: "10",
    target: "converted_14d",
    task: "Classification",
    glyph: "Cv",
  },
  {
    id: "promotion-uplift",
    name: "Promotion Uplift",
    description: "Study heterogeneous offer response across treated and untreated customer cohorts.",
    path: "/data/retail/promotion-uplift.csv",
    rows: "1,300",
    features: "10",
    target: "purchased_30d",
    task: "Classification",
    glyph: "Up",
  },
  {
    id: "campaign-response",
    name: "Campaign Response",
    description: "Model marketing response from RFM, channel, engagement, and audience signals.",
    path: "/data/retail/campaign-response.csv",
    rows: "1,100",
    features: "10",
    target: "responded_30d",
    task: "Classification",
    glyph: "Mk",
  },
];

type Capability = "embeddings" | "explain" | "predict" | "scenario";

const MAX_ROWS = 3_000;
const MAX_BYTES = 12 * 1024 * 1024;
const TEST_FRACTION = 0.2;

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));
const formatNumber = (value: number) => new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value);
const formatTargetValue = (dataset: LocalDataset, value: number) => /value|spend|revenue|amount|sales|price|cost|income/i.test(dataset.target)
  ? new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(value)
  : formatNumber(value);

function hashString(value: string) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function seededRandom(seed: number) {
  let state = seed || 1;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffled(indices: number[], random: () => number) {
  const result = [...indices];
  for (let index = result.length - 1; index > 0; index -= 1) {
    const next = Math.floor(random() * (index + 1));
    [result[index], result[next]] = [result[next], result[index]];
  }
  return result;
}

function parseCSV(text: string, name: string, source: string): LocalDataset {
  const matrix: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') {
      if (quoted && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      row.push(field.trim());
      field = "";
    } else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      row.push(field.trim());
      if (row.some((value) => value.length > 0)) matrix.push(row);
      row = [];
      field = "";
    } else {
      field += character;
    }
  }
  row.push(field.trim());
  if (row.some((value) => value.length > 0)) matrix.push(row);

  if (matrix.length < 2) throw new Error("This CSV needs a header row and at least one data row.");
  const seen = new Map<string, number>();
  const headers = matrix[0].map((rawHeader, index) => {
    const base = rawHeader || `Column ${index + 1}`;
    const count = seen.get(base) ?? 0;
    seen.set(base, count + 1);
    return count === 0 ? base : `${base} ${count + 1}`;
  });
  const rows = matrix.slice(1, MAX_ROWS + 1).map((values) => headers.map((_, index) => values[index] ?? ""));
  const numericColumns = headers.filter((_, columnIndex) => {
    const present = rows.map((values) => values[columnIndex]).filter((value) => value !== "");
    if (present.length < Math.min(3, rows.length)) return false;
    return present.filter((value) => Number.isFinite(Number(value))).length / present.length >= 0.85;
  });

  if (numericColumns.length < 2) throw new Error("Nori Studio needs at least two mostly numeric columns for an interactive projection.");
  return {
    id: `local-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    name: name.replace(/\.csv$/i, "") || "Untitled dataset",
    source,
    headers,
    rows,
    numericColumns,
    target: "",
    trainIndices: [],
    testIndices: [],
    splitSeed: 0,
  };
}

function isClassificationTarget(dataset: LocalDataset, indices: number[]) {
  const targetIndex = dataset.headers.indexOf(dataset.target);
  const values = indices.map((index) => dataset.rows[index]?.[targetIndex] ?? "Missing");
  const present = values.filter((value) => value !== "");
  const numeric = present.filter((value) => Number.isFinite(Number(value))).length / Math.max(present.length, 1) >= 0.85;
  const unique = new Set(present).size;
  return !numeric || unique <= Math.min(20, Math.max(2, Math.round(Math.sqrt(present.length))));
}

export function prepareDataset(dataset: LocalDataset, target: string) {
  const next = { ...dataset, target };
  const seed = hashString(`${dataset.name}:${target}:${dataset.rows.length}`);
  const random = seededRandom(seed);
  const allIndices = dataset.rows.map((_, index) => index);
  const testIndices: number[] = [];
  const trainIndices: number[] = [];

  if (isClassificationTarget(next, allIndices)) {
    const targetIndex = dataset.headers.indexOf(target);
    const groups = new Map<string, number[]>();
    allIndices.forEach((index) => {
      const label = dataset.rows[index][targetIndex] || "Missing";
      groups.set(label, [...(groups.get(label) ?? []), index]);
    });
    groups.forEach((indices) => {
      const group = shuffled(indices, random);
      const testCount = group.length > 1 ? Math.max(1, Math.round(group.length * TEST_FRACTION)) : 0;
      testIndices.push(...group.slice(0, testCount));
      trainIndices.push(...group.slice(testCount));
    });
  } else {
    const indices = shuffled(allIndices, random);
    const testCount = Math.max(1, Math.round(indices.length * TEST_FRACTION));
    testIndices.push(...indices.slice(0, testCount));
    trainIndices.push(...indices.slice(testCount));
  }

  if (testIndices.length === 0 && trainIndices.length > 1) testIndices.push(trainIndices.pop() as number);
  return { ...next, trainIndices: shuffled(trainIndices, random), testIndices: shuffled(testIndices, random), splitSeed: seed };
}

export async function loadStarterDataset(starter: StarterDataset) {
  const response = await fetch(starter.path);
  if (!response.ok) throw new Error(`Could not load ${starter.name}.`);
  const dataset = parseCSV(await response.text(), starter.name, `Retail demo cohort · ${starter.task}`);
  return prepareDataset({ ...dataset, id: `starter-${starter.id}` }, starter.target);
}

function valuesFor(dataset: LocalDataset, column: string) {
  const index = dataset.headers.indexOf(column);
  return dataset.rows.map((row) => row[index] === "" ? Number.NaN : Number(row[index])).filter(Number.isFinite);
}

function rangeFor(dataset: LocalDataset, column: string) {
  const values = valuesFor(dataset, column);
  return { min: Math.min(...values), max: Math.max(...values), mean: values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1) };
}

function numericValue(dataset: LocalDataset, row: string[], column: string) {
  const raw = row[dataset.headers.indexOf(column)];
  const value = raw === "" ? Number.NaN : Number(raw);
  return Number.isFinite(value) ? value : rangeFor(dataset, column).mean;
}

function targetProfile(dataset: LocalDataset, indices = dataset.rows.map((_, index) => index)) {
  const index = dataset.headers.indexOf(dataset.target);
  const raw = indices.map((rowIndex) => dataset.rows[rowIndex][index]);
  const numeric = raw.filter((value) => value !== "" && Number.isFinite(Number(value)));
  const numericTarget = numeric.length / Math.max(raw.length, 1) >= 0.85;
  const classification = isClassificationTarget(dataset, indices);
  if (numericTarget && !classification) {
    const values = numeric.map(Number);
    return {
      numeric: true,
      values: raw.map(Number),
      baseline: values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1),
      threshold: [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)] ?? 0,
      labels: [] as string[],
    };
  }
  const counts = new Map<string, number>();
  raw.forEach((value) => counts.set(value || "Missing", (counts.get(value || "Missing") ?? 0) + 1));
  const labels = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([label]) => label);
  return { numeric: false, values: [] as number[], baseline: (counts.get(labels[0]) ?? 0) / Math.max(raw.length, 1), threshold: 0, labels };
}

function pearson(left: number[], right: number[]) {
  const pairs = left.map((value, index) => [value, right[index]]).filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b));
  if (pairs.length < 3) return 0;
  const leftMean = pairs.reduce((sum, pair) => sum + pair[0], 0) / pairs.length;
  const rightMean = pairs.reduce((sum, pair) => sum + pair[1], 0) / pairs.length;
  let numerator = 0;
  let leftScale = 0;
  let rightScale = 0;
  pairs.forEach(([a, b]) => {
    numerator += (a - leftMean) * (b - rightMean);
    leftScale += (a - leftMean) ** 2;
    rightScale += (b - rightMean) ** 2;
  });
  return numerator / Math.sqrt(leftScale * rightScale || 1);
}

function neighborEstimate(dataset: LocalDataset, query: Record<string, number>, selected: number) {
  const targetIndex = dataset.headers.indexOf(dataset.target);
  const features = dataset.numericColumns.filter((column) => column !== dataset.target).slice(0, 8);
  const ranges = Object.fromEntries(features.map((column) => [column, rangeFor(dataset, column)]));
  const contextIndices = dataset.trainIndices.length > 0 ? dataset.trainIndices : dataset.rows.map((_, index) => index);
  const profile = targetProfile(dataset, contextIndices);
  const distances = contextIndices.map((rowIndex) => {
    const row = dataset.rows[rowIndex];
    let distance = 0;
    features.forEach((column) => {
      const value = numericValue(dataset, row, column);
      const range = ranges[column];
      const scale = range.max - range.min || 1;
      distance += ((value - (query[column] ?? value)) / scale) ** 2;
    });
    return { row, rowIndex, distance };
  }).filter((item) => item.rowIndex !== selected).sort((a, b) => a.distance - b.distance).slice(0, Math.min(42, Math.max(8, Math.round(Math.sqrt(contextIndices.length)))));

  if (profile.numeric) {
    const values = distances.map((item) => Number(item.row[targetIndex])).filter(Number.isFinite);
    return { value: values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1), label: dataset.target, numeric: true };
  }
  const counts = new Map<string, number>();
  distances.forEach((item) => {
    const label = item.row[targetIndex] || "Missing";
    counts.set(label, (counts.get(label) ?? 0) + 1);
  });
  const [predictedLabel, matches] = [...counts.entries()].sort((a, b) => b[1] - a[1])[0] ?? [profile.labels[0] ?? "Unknown", 0];
  return { value: matches / Math.max(distances.length, 1), label: predictedLabel, numeric: false };
}

export function DatasetImporter({ open, onClose, onImport }: { open: boolean; onClose: () => void; onImport: (dataset: LocalDataset) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [draft, setDraft] = useState<LocalDataset | null>(null);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  if (!open) return null;

  const closeImporter = () => {
    setDraft(null);
    setError("");
    setBusy(false);
    onClose();
  };

  const readFile = async (file: File) => {
    setError("");
    if (file.size > MAX_BYTES) {
      setError("Choose a CSV under 12 MB for this browser preview.");
      return;
    }
    try {
      setDraft(parseCSV(await file.text(), file.name, "Local CSV · stays in this browser tab"));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not read this CSV.");
    }
  };

  const loadUrl = async () => {
    setBusy(true);
    setError("");
    try {
      const parsedUrl = new URL(url);
      if (!/^https?:$/.test(parsedUrl.protocol)) throw new Error("Enter a public http(s) CSV URL.");
      const response = await fetch(parsedUrl.toString());
      if (!response.ok) throw new Error(`The CSV server returned ${response.status}.`);
      const text = await response.text();
      if (new Blob([text]).size > MAX_BYTES) throw new Error("That CSV is over the 12 MB browser-preview limit.");
      const filename = parsedUrl.pathname.split("/").at(-1) || "Linked dataset";
      setDraft(parseCSV(text, filename, `Public CSV link · ${parsedUrl.hostname}`));
    } catch (reason) {
      setError(reason instanceof Error ? `${reason.message} The host must allow browser access (CORS).` : "Could not load this CSV URL.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="importer-backdrop" role="presentation" onPointerDown={(event) => event.target === event.currentTarget && closeImporter()}>
      <section className="importer" role="dialog" aria-modal="true" aria-labelledby="importer-title">
        <div className="importer-heading">
          <div><p className="section-kicker">New demo dataset</p><h2 id="importer-title">Bring your own table.</h2></div>
          <button type="button" onClick={closeImporter} aria-label="Close dataset importer">×</button>
        </div>
        {!draft ? (
          <div className="importer-options">
            <button
              className="drop-zone"
              type="button"
              onClick={() => inputRef.current?.click()}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                const file = event.dataTransfer.files[0];
                if (file) void readFile(file);
              }}
            >
              <span className="upload-glyph" aria-hidden="true">↑</span>
              <strong>Drop a CSV here</strong>
              <small>or choose a file · up to 3,000 rows / 12 MB</small>
            </button>
            <input ref={inputRef} type="file" accept=".csv,text/csv" hidden onChange={(event) => event.target.files?.[0] && void readFile(event.target.files[0])} />
            <div className="import-divider"><span>or link a public file</span></div>
            <div className="url-import">
              <input type="url" value={url} placeholder="https://…/dataset.csv" aria-label="Public CSV URL" onChange={(event) => setUrl(event.target.value)} />
              <button type="button" disabled={!url || busy} onClick={() => void loadUrl()}>{busy ? "Reading…" : "Load link"}</button>
            </div>
            <p className="privacy-note"><span>●</span> Files are parsed locally and are not uploaded to Nori Studio. Linked files must permit browser access.</p>
          </div>
        ) : (
          <div className="dataset-ready">
            <div className="ready-summary">
              <span className="ready-mark">✓</span>
              <div><strong>{draft.name}</strong><small>{draft.rows.length.toLocaleString()} rows · {draft.headers.length} columns · {draft.numericColumns.length} numeric</small></div>
            </div>
            <label className="target-field">
              <span>1. Pick the target column</span>
              <select value={draft.target} onChange={(event) => setDraft({ ...draft, target: event.target.value })}>
                <option value="" disabled>Select an outcome…</option>
                {draft.headers.map((header) => <option value={header} key={header}>{header}</option>)}
              </select>
              <small>This becomes the outcome Nori predicts, organizes around, and explains.</small>
            </label>
            <div className="split-preview">
              <div><span>2. Random context / query split</span><strong>80% <i>context</i> · 20% <i>test</i></strong></div>
              <div className="split-bar"><span /><i /></div>
              <small>Classification targets are stratified. The seeded split stays fixed across every lens.</small>
            </div>
            <div className="column-preview">
              {draft.headers.slice(0, 8).map((header) => <span key={header}>{header}{draft.numericColumns.includes(header) ? <i>#</i> : null}</span>)}
              {draft.headers.length > 8 ? <span>+{draft.headers.length - 8} more</span> : null}
            </div>
            <div className="import-actions">
              <button type="button" className="quiet-button" onClick={() => setDraft(null)}>Choose another</button>
              <button type="button" className="solid-button" disabled={!draft.target} onClick={() => { onImport(prepareDataset(draft, draft.target)); closeImporter(); }}>Create demo <span>→</span></button>
            </div>
          </div>
        )}
        {error ? <p className="import-error" role="alert">{error}</p> : null}
      </section>
    </div>
  );
}

function pcaCoordinates(dataset: LocalDataset, features: string[]) {
  const activeFeatures = features.slice(0, 10);
  const columns = activeFeatures.map((column) => {
    const values = dataset.rows.map((row) => numericValue(dataset, row, column));
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const deviation = Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(values.length - 1, 1)) || 1;
    return { values, mean, deviation };
  });
  const matrix = dataset.rows.map((_, rowIndex) => columns.map((column) => (column.values[rowIndex] - column.mean) / column.deviation));
  const size = activeFeatures.length;
  const covariance = Array.from({ length: size }, (_, left) => Array.from({ length: size }, (_, right) => matrix.reduce((sum, row) => sum + row[left] * row[right], 0) / Math.max(matrix.length - 1, 1)));

  const component = (seed: number[], previous?: number[]) => {
    let vector = seed;
    for (let iteration = 0; iteration < 36; iteration += 1) {
      let next = covariance.map((row) => row.reduce((sum, value, index) => sum + value * vector[index], 0));
      if (previous) {
        const projection = next.reduce((sum, value, index) => sum + value * previous[index], 0);
        next = next.map((value, index) => value - projection * previous[index]);
      }
      const length = Math.sqrt(next.reduce((sum, value) => sum + value * value, 0)) || 1;
      vector = next.map((value) => value / length);
    }
    return vector;
  };

  const first = component(Array.from({ length: size }, (_, index) => 1 + index / Math.max(size, 1)));
  const second = component(Array.from({ length: size }, (_, index) => index % 2 === 0 ? 1 : -1), first);
  const raw = matrix.map((row) => ({
    x: row.reduce((sum, value, index) => sum + value * first[index], 0),
    y: row.reduce((sum, value, index) => sum + value * second[index], 0),
  }));
  const normalize = (values: number[]) => {
    const sorted = [...values].sort((a, b) => a - b);
    const low = sorted[Math.floor(sorted.length * 0.02)] ?? 0;
    const high = sorted[Math.floor(sorted.length * 0.98)] ?? 1;
    return values.map((value) => clamp((value - low) / (high - low || 1), 0, 1));
  };
  const normalizedX = normalize(raw.map((point) => point.x));
  const normalizedY = normalize(raw.map((point) => point.y));
  return raw.map((_, index) => ({ x: normalizedX[index], y: normalizedY[index] }));
}

function LocalProjection({ dataset, features, mode, xColumn, yColumn, selected, onSelect }: { dataset: LocalDataset; features: string[]; mode: "pca" | "axes"; xColumn: string; yColumn: string; selected: number; onSelect: (index: number) => void }) {
  const targetIndex = dataset.headers.indexOf(dataset.target);
  const profile = targetProfile(dataset, dataset.trainIndices);
  const testSet = new Set(dataset.testIndices);
  const palette = ["#c45f10", "#1e2a78", "#4f9b73", "#8b63b8", "#c45f7b", "#54748f"];
  const categoryColors = new Map(profile.labels.slice(0, palette.length).map((label, index) => [label, palette[index]]));
  const regressionRange = rangeFor(dataset, dataset.target);
  const xRange = rangeFor(dataset, xColumn);
  const yRange = rangeFor(dataset, yColumn);
  const points = mode === "pca"
    ? pcaCoordinates(dataset, features)
    : dataset.rows.map((row) => {
      return {
        x: clamp((numericValue(dataset, row, xColumn) - xRange.min) / (xRange.max - xRange.min || 1), 0, 1),
        y: clamp((numericValue(dataset, row, yColumn) - yRange.min) / (yRange.max - yRange.min || 1), 0, 1),
      };
    });
  const colorFor = (row: string[]) => {
    if (!profile.numeric) return categoryColors.get(row[targetIndex] || "Missing") ?? "#9da3a1";
    const value = Number(row[targetIndex]);
    const ratio = Number.isFinite(value) ? clamp((value - regressionRange.min) / (regressionRange.max - regressionRange.min || 1), 0, 1) : 0.5;
    const red = Math.round(30 + (196 - 30) * ratio);
    const green = Math.round(42 + (95 - 42) * ratio);
    const blue = Math.round(120 + (16 - 120) * ratio);
    return `rgb(${red}, ${green}, ${blue})`;
  };
  return (
    <div className="local-projection">
      <svg viewBox="0 0 1000 600" preserveAspectRatio="xMidYMid meet" role="img" aria-label={`${mode === "pca" ? "PCA overview" : "Feature projection"} of ${dataset.rows.length} rows`}>
        <g className="projection-grid">
          {[1, 2, 3, 4].map((line) => <line key={`v${line}`} x1={line * 200} x2={line * 200} y1="0" y2="600" />)}
          {[1, 2, 3, 4].map((line) => <line key={`h${line}`} x1="0" x2="1000" y1={line * 120} y2={line * 120} />)}
        </g>
        {dataset.rows.map((row, index) => {
          const x = 55 + points[index].x * 890;
          const y = 550 - points[index].y * 500;
          const isTest = testSet.has(index);
          return <circle key={index} cx={x} cy={y} r={index === selected ? 11 : isTest ? 6.2 : 4.8} style={{ fill: colorFor(row) }} className={`${isTest ? "test-point" : "train-point"} ${index === selected ? "selected-point" : ""}`} onClick={isTest ? () => onSelect(index) : undefined} />;
        })}
      </svg>
      <span className="projection-label x-label">{mode === "pca" ? "principal direction 01" : xColumn}</span>
      <span className="projection-label y-label">{mode === "pca" ? "principal direction 02" : yColumn}</span>
      <div className="projection-badge"><span>Random 80 / 20 split</span><strong>{dataset.trainIndices.length.toLocaleString()} context · {dataset.testIndices.length.toLocaleString()} test</strong></div>
      <div className="projection-target-legend">
        <strong>Color · {dataset.target}</strong>
        {profile.numeric ? <><span><i style={{ background: palette[1] }} /> lower</span><span><i style={{ background: palette[0] }} /> higher</span></> : profile.labels.slice(0, palette.length).map((label, index) => <span key={label}><i style={{ background: palette[index] }} /> {label}</span>)}
      </div>
      <div className="projection-split-key"><span><i /> context</span><span><i /> test query</span></div>
    </div>
  );
}

export function LocalDatasetWorkspace({ dataset, capability }: { dataset: LocalDataset; capability: Capability }) {
  const [selected, setSelected] = useState(dataset.testIndices[0] ?? 0);
  const features = useMemo(() => dataset.numericColumns.filter((column) => column !== dataset.target), [dataset]);
  const [projectionMode, setProjectionMode] = useState<"pca" | "axes">("pca");
  const [xColumn, setXColumn] = useState(features[0] ?? dataset.numericColumns[0]);
  const [yColumn, setYColumn] = useState(features[1] ?? dataset.numericColumns[1] ?? dataset.numericColumns[0]);
  const selectedRow = dataset.rows[selected] ?? dataset.rows[0];
  const targetIndex = dataset.headers.indexOf(dataset.target);
  const query = useMemo(() => Object.fromEntries(features.map((column) => [column, numericValue(dataset, selectedRow, column)])), [dataset, features, selectedRow]);
  const [scenarioState, setScenarioState] = useState<{ rowIndex: number; values: Record<string, number> }>({ rowIndex: selected, values: query });
  const scenario = scenarioState.rowIndex === selected ? scenarioState.values : query;
  const profile = useMemo(() => targetProfile(dataset, dataset.trainIndices), [dataset]);
  const estimate = useMemo(() => neighborEstimate(dataset, query, selected), [dataset, query, selected]);
  const scenarioEstimate = useMemo(() => neighborEstimate(dataset, scenario, selected), [dataset, scenario, selected]);

  const effects = useMemo(() => {
    const contextRows = dataset.trainIndices.map((index) => dataset.rows[index]);
    const targetValues = profile.numeric
      ? contextRows.map((row) => row[targetIndex] === "" ? Number.NaN : Number(row[targetIndex]))
      : contextRows.map((row) => row[targetIndex] === selectedRow[targetIndex] ? 1 : 0);
    return features.map((column) => {
      const range = rangeFor(dataset, column);
      const values = contextRows.map((row) => numericValue(dataset, row, column));
      const correlation = pearson(values, targetValues);
      const selectedValue = numericValue(dataset, selectedRow, column);
      const standardized = (selectedValue - range.mean) / (range.max - range.min || 1);
      return { name: column, value: clamp(correlation * standardized * 2.2, -0.46, 0.46), detail: formatNumber(selectedValue), correlation };
    }).sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 6);
  }, [dataset, features, profile.numeric, selectedRow, targetIndex]);

  const selectedTarget = profile.numeric && Number.isFinite(Number(selectedRow[targetIndex])) ? formatTargetValue(dataset, Number(selectedRow[targetIndex])) : selectedRow[targetIndex] || "Missing";
  const resultLabel = estimate.numeric ? formatTargetValue(dataset, estimate.value) : `${Math.round(estimate.value * 100)}%`;
  const baselineLabel = profile.numeric ? formatTargetValue(dataset, profile.baseline) : `${Math.round(profile.baseline * 100)}%`;
  const nextTestRow = () => {
    const position = dataset.testIndices.indexOf(selected);
    setSelected(dataset.testIndices[(position + 1) % dataset.testIndices.length] ?? selected);
  };

  return (
    <div className="local-workspace">
      <div className="local-context-bar">
        <span><b>{dataset.trainIndices.length.toLocaleString()}</b> context rows</span><span><b>{dataset.testIndices.length}</b> test queries</span><span>Target <b>{dataset.target}</b></span>
        <span className="local-only">Browser-local preview</span>
      </div>

      {capability === "embeddings" ? (
        <div className="local-embedding-layout">
          <div className="local-embedding-main">
            <div className="local-axis-controls">
              <div className="projection-mode-control"><span>Projection</span><div className="segmented"><button type="button" className={projectionMode === "pca" ? "is-active indigo" : ""} onClick={() => setProjectionMode("pca")}>PCA overview</button><button type="button" className={projectionMode === "axes" ? "is-active orange" : ""} onClick={() => setProjectionMode("axes")}>Feature axes</button></div></div>
              {projectionMode === "axes" ? <div className="projection-axis-pickers"><label><span>X axis</span><select value={xColumn} onChange={(event) => setXColumn(event.target.value)}>{features.map((column) => <option key={column}>{column}</option>)}</select></label><label><span>Y axis</span><select value={yColumn} onChange={(event) => setYColumn(event.target.value)}>{features.map((column) => <option key={column}>{column}</option>)}</select></label></div> : <div className="projection-summary"><strong>{features.length} numeric signals</strong><span>standardized into two principal directions</span></div>}
              <p>{projectionMode === "pca" ? "A browser-local PCA overview colored by the chosen target. Outlined points are held-out test queries." : "Compare two raw features directly. Missing numeric values are mean-imputed for display."}</p>
            </div>
            <LocalProjection dataset={dataset} features={features} mode={projectionMode} xColumn={xColumn} yColumn={yColumn} selected={selected} onSelect={setSelected} />
          </div>
          <aside className="local-record-panel">
            <div className="record-heading"><span>Held-out test row</span><b>#{selected + 1}</b></div>
            <div className="local-target-value"><span>{dataset.target}</span><strong>{selectedTarget}</strong></div>
            <dl className="record-list">
              {dataset.headers.slice(0, 7).map((header, index) => <div key={header}><dt>{header}</dt><dd title={selectedRow[index]}>{selectedRow[index] || "—"}</dd></div>)}
            </dl>
          </aside>
        </div>
      ) : null}

      {capability === "explain" ? (
        <div className="local-explain-layout">
          <div className="local-explain-chart">
            <div className="local-pane-title"><div><p className="section-kicker">Local signal preview</p><h3>Which values distinguish row #{selected + 1}?</h3></div><span className="prototype-tag">Correlation × deviation</span></div>
            <div className="waterfall local-waterfall">
              <div className="zero-line" />
              {effects.map((item) => <div className="effect-row" key={item.name}>
                <div><strong>{item.name}</strong><small>{item.detail} · r {item.correlation.toFixed(2)}</small></div>
                <div className="effect-track"><span className={item.value >= 0 ? "risk" : "protective"} style={item.value >= 0 ? { left: "50%", width: `${Math.abs(item.value) * 100}%` } : { right: "50%", width: `${Math.abs(item.value) * 100}%` }} /></div>
                <b className={item.value >= 0 ? "risk-text" : "protective-text"}>{item.value >= 0 ? "+" : ""}{item.value.toFixed(2)}</b>
              </div>)}
            </div>
            <div className="method-note"><span className="method-mark">i</span><p>This browser-local diagnostic is deliberately not labeled as SHAP or SHAP-IQ. It becomes a Nori explanation when the public explainer endpoint is connected.</p></div>
          </div>
          <aside className="local-insight-panel"><p className="section-kicker">Held-out outcome</p><strong className="giant-local-value">{selectedTarget}</strong><p>The largest local signal is <b>{effects[0]?.name ?? "not available"}</b>. Effects use only the training context as their baseline.</p><button type="button" className="row-step" onClick={nextTestRow}>Next test row <span>→</span></button></aside>
        </div>
      ) : null}

      {capability === "predict" ? (
        <div className="local-predict-layout">
          <div className="local-prediction-hero">
            <div className="local-pane-title"><div><p className="section-kicker">Nearest-context baseline</p><h3>A useful result before the Nori endpoint is connected.</h3></div><span className="prototype-tag">Local kNN</span></div>
            <div className="local-score"><strong>{resultLabel}</strong><span>{estimate.numeric ? `estimated ${dataset.target}` : `neighbors matching “${estimate.label}”`}</span></div>
            <div className="prediction-band"><span style={{ left: `${estimate.numeric ? clamp(((estimate.value - rangeFor(dataset, dataset.target).min) / (rangeFor(dataset, dataset.target).max - rangeFor(dataset, dataset.target).min || 1)) * 100, 3, 97) : clamp(estimate.value * 100, 3, 97)}%` }} /></div>
            <div className="prediction-scale"><span>Dataset baseline {baselineLabel}</span><span>Selected row {selectedTarget}</span></div>
          </div>
          <aside className="local-method-panel"><p className="section-kicker">Nori-ready setup</p><h3>{features.length} numeric signals, one target, zero training UI.</h3><ol className="context-flow"><li><b>01</b><span><strong>Reference context</strong><small>{dataset.trainIndices.length.toLocaleString()} randomly sampled training rows</small></span></li><li><b>02</b><span><strong>Held-out query</strong><small>Test row #{selected + 1} · target hidden</small></span></li><li><b>03</b><span><strong>Evaluate output</strong><small>Reveal actual {dataset.target}: {selectedTarget}</small></span></li></ol><button type="button" className="row-step" onClick={nextTestRow}>Try another test row <span>→</span></button></aside>
        </div>
      ) : null}

      {capability === "scenario" ? (
        <div className="local-scenario-layout">
          <div className="local-scenario-controls">
            <div className="scenario-heading"><div><p className="section-kicker">Scenario inputs</p><h3>Change the row. Keep the cohort fixed.</h3></div><button type="button" onClick={() => setScenarioState({ rowIndex: selected, values: query })}>Reset</button></div>
            {features.slice(0, 4).map((column) => {
              const range = rangeFor(dataset, column);
              const step = (range.max - range.min) / 100 || 1;
              const inputId = `local-scenario-${dataset.id}-${column}`;
              return <label className="slider-field" key={column} htmlFor={inputId}><span><b>{column}</b><strong>{formatNumber(scenario[column] ?? query[column])}</strong></span><input id={inputId} aria-label={column} type="range" min={range.min} max={range.max} step={step} value={scenario[column] ?? query[column]} onChange={(event) => setScenarioState({ rowIndex: selected, values: { ...scenario, [column]: Number(event.target.value) } })} /><small><span>{formatNumber(range.min)}</span><span>{formatNumber(range.max)}</span></small></label>;
            })}
          </div>
          <div className="local-scenario-result">
            <div className="scenario-result-top"><p className="section-kicker">Local cohort comparison</p><span className="prototype-tag">Interactive</span></div>
            <div className="local-comparison"><div><span>Original estimate</span><strong>{resultLabel}</strong></div><b>→</b><div><span>Scenario estimate</span><strong>{scenarioEstimate.numeric ? formatTargetValue(dataset, scenarioEstimate.value) : `${Math.round(scenarioEstimate.value * 100)}%`}</strong></div></div>
            <div className="scenario-delta"><span className={scenarioEstimate.value <= estimate.value ? "down" : "up"}>{scenarioEstimate.value <= estimate.value ? "↓" : "↑"}</span><div><strong>{estimate.numeric ? formatTargetValue(dataset, Math.abs(scenarioEstimate.value - estimate.value)) : `${Math.round(Math.abs(scenarioEstimate.value - estimate.value) * 100)} points`} estimated change</strong><small>using the same browser-local reference cohort</small></div></div>
            <p className="scenario-copy">This gives you a working demo flow on any numeric CSV. A connected Nori service can use the same controls to rerun true zero-shot inference and explanation.</p>
          </div>
        </div>
      ) : null}
    </div>
  );
}
