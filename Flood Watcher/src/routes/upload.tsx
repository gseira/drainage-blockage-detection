import { useCallback, useRef, useState, useEffect } from "react";
import { createFileRoute } from "@tanstack/react-router";
import {
  analyzeUpload,
  analyzeWebcam,
  fetchHealth,
  webcamProxyUrl,
  reclassifyResult,
  REVIEW_ALL_CASES,
  type PredictResult,
} from "@/lib/api";
import { AppShell } from "@/components/dashboard/app-shell";
import { useAssets } from "@/hooks/useAssets";
import type { Asset } from "@/lib/dashboard-data";

export const Route = createFileRoute("/upload")({
  head: () => ({ meta: [{ title: "Flood Watcher — Analyse" }] }),
  component: AnalysePage,
});

// ── Types ──────────────────────────────────────────────────────────────────────

type Screen =
  | { kind: "home" }
  | { kind: "webcam-pick" }
  | { kind: "upload-pick" }
  | { kind: "results"; items: ResultItem[] };

interface ResultItem {
  id:       string;
  label:    string;
  region?:  string;
  preview:  string;           // blob URL (upload) or webcam proxy (live)
  status:   "pending" | "running" | "done" | "error";
  result?:  PredictResult;
  error?:   string;
  file?:    File;             // upload only
  campath?: string;           // live only
  camName?: string;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function uid() { return Math.random().toString(36).slice(2); }

function riskColor(risk?: string) {
  if (risk === "HIGH" || risk === "BLOCKED") return "var(--color-risk-high)";
  if (risk === "FLAGGED")                   return "var(--color-risk-medium)";
  if (risk === "LOW" || risk === "CLEAR")   return "var(--color-ok)";
  if (risk === "UNKNOWN" || risk === "OFFLINE" || risk === "OTHER") return "var(--color-muted-foreground)";
  return "var(--color-foreground)";
}

function predLabel(r?: PredictResult) {
  if (!r) return "";
  if (r.prediction === "BLOCKED") return "BLOCKED";
  if (r.prediction === "FLAGGED") return "UNDER REVIEW";
  if (r.prediction === "UNKNOWN" || r.prediction === "OFFLINE") return "MANUAL CHECK REQUIRED";
  if (r.prediction === "OTHER")   return "NOT A DRAIN";
  return "CLEAR";
}

// ── LLM explanation renderer ───────────────────────────────────────────────────

const SECTION_META: Record<string, { num: string; accent: string }> = {
  "OBSERVATION":        { num: "01", accent: "var(--color-foreground)" },
  "RISK ASSESSMENT":    { num: "02", accent: "var(--color-risk-high)"  },
  "RECOMMENDED ACTION": { num: "03", accent: "var(--color-ok)"         },
};

function ExplanationBlock({ text, risk }: { text: string; risk?: string }) {
  const SECTIONS = ["OBSERVATION", "RISK ASSESSMENT", "RECOMMENDED ACTION"];
  const parts: { heading: string; body: string }[] = [];
  let rem = text.trim();

  for (let i = 0; i < SECTIONS.length; i++) {
    const key  = SECTIONS[i];
    const next = SECTIONS[i + 1];
    const idx  = rem.toUpperCase().indexOf(key);
    if (idx === -1) continue;
    const s    = idx + key.length;
    const ni   = next ? rem.toUpperCase().indexOf(next, s) : -1;
    const body = (ni === -1 ? rem.slice(s) : rem.slice(s, ni)).replace(/^[\s:]+/, "").trim();
    if (ni !== -1) rem = rem.slice(ni);
    parts.push({ heading: key, body });
  }

  if (parts.length === 0) {
    // The current (v2) prompt returns plain sentences with no section
    // headers at all, so this is now the common path — give it the same
    // clear, boxed treatment as a sectioned report rather than a bare
    // unstyled paragraph.
    const accent = riskColor(risk);
    return (
      <div style={{
        borderRadius: "0 8px 8px 0",
        padding: "24px 28px",
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
        borderLeft: `5px solid ${accent}`,
        boxShadow: "0 1px 4px rgba(0,0,0,0.07)",
        display: "flex", flexDirection: "column", alignItems: "center",
        textAlign: "center",
      }}>
        <div style={{
          fontSize: 11, fontWeight: 700, textTransform: "uppercase",
          letterSpacing: "0.14em", color: accent, marginBottom: 12,
        }}>
          AI Assessment
        </div>
        <p style={{
          fontSize: 19, lineHeight: 1.85, fontWeight: 500,
          color: "var(--color-foreground)", whiteSpace: "pre-line", margin: 0,
          textAlign: "center",
        }}>
          {text}
        </p>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {parts.map(({ heading, body }) => {
        const meta   = SECTION_META[heading] ?? { num: "—", accent: "var(--color-foreground)" };
        const accent = heading === "RISK ASSESSMENT" ? riskColor(risk) : meta.accent;
        return (
          <div key={heading} style={{
            borderLeft: `4px solid ${accent}`,
            borderRadius: "0 8px 8px 0",
            padding: "20px 28px",
            background: "var(--color-surface)",
            border: `1px solid var(--color-border)`,
            borderLeftWidth: 4,
            borderLeftColor: accent,
          }}>
            <div style={{
              display: "flex", alignItems: "center", gap: 10, marginBottom: 12,
            }}>
              <span style={{
                fontSize: 11, fontWeight: 800, fontVariantNumeric: "tabular-nums",
                color: accent, letterSpacing: "0.05em",
              }}>
                {meta.num}
              </span>
              <span style={{
                fontSize: 11, fontWeight: 700, textTransform: "uppercase",
                letterSpacing: "0.14em", color: accent,
              }}>
                {heading}
              </span>
            </div>
            <p style={{ fontSize: 17, lineHeight: 1.8, color: "var(--color-foreground)", margin: 0 }}>
              {body}
            </p>
          </div>
        );
      })}
    </div>
  );
}

// ── Home screen ────────────────────────────────────────────────────────────────

function HomeScreen({ onChoose }: { onChoose: (mode: "webcam" | "upload") => void }) {
  return (
    <div className="flex flex-1 flex-col overflow-auto">

      {/* Hero */}
      <div className="flex flex-col items-center justify-center px-8 pt-16 pb-10 text-center">
        {/* Badge */}
        <div className="mb-5 inline-flex items-center gap-2 rounded-full px-4 py-1.5"
          style={{ background: "var(--color-surface)", border: "1px solid var(--color-border)" }}>
          <span className="size-1.5 rounded-full animate-pulse" style={{ background: "var(--color-ok)" }} />
          <span style={{ fontSize: 11.5, fontWeight: 600, letterSpacing: "0.05em", color: "var(--color-muted-foreground)" }}>
            AI-powered drainage inspection
          </span>
        </div>

        <h1 style={{ fontSize: 32, fontWeight: 700, letterSpacing: "-0.02em", lineHeight: 1.2, marginBottom: 12 }}>
          Inspect a drainage camera
        </h1>
        <p style={{ fontSize: 15, color: "var(--color-muted-foreground)", maxWidth: 480, lineHeight: 1.6, marginBottom: 40 }}>
          Run blockage detection on live EA webcam feeds or your own images.
          The model highlights at-risk areas and generates a structured AI inspection report.
        </p>

        {/* Main action buttons */}
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", justifyContent: "center" }}>
          <button onClick={() => onChoose("webcam")}
            style={{
              display: "flex", flexDirection: "column", alignItems: "center", gap: 16,
              padding: "32px 40px", minWidth: 240, borderRadius: 10,
              border: "1.5px solid var(--color-border)", background: "var(--color-foreground)",
              color: "var(--color-background)", cursor: "pointer", transition: "opacity 0.15s",
            }}
            onMouseEnter={e => (e.currentTarget.style.opacity = "0.88")}
            onMouseLeave={e => (e.currentTarget.style.opacity = "1")}>
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="1.4" strokeLinecap="round">
              <path d="M23 7 16 12 23 17z"/>
              <rect x="1" y="5" width="15" height="14" rx="2"/>
            </svg>
            <div>
              <div style={{ fontSize: 16, fontWeight: 700 }}>Live webcams</div>
              <div style={{ fontSize: 12, marginTop: 4, opacity: 0.6 }}>Select from EA camera network</div>
            </div>
          </button>

          <button onClick={() => onChoose("upload")}
            style={{
              display: "flex", flexDirection: "column", alignItems: "center", gap: 16,
              padding: "32px 40px", minWidth: 240, borderRadius: 10,
              border: "1.5px solid var(--color-border)", background: "var(--color-surface)",
              cursor: "pointer", transition: "background 0.15s",
            }}
            onMouseEnter={e => (e.currentTarget.style.background = "var(--color-muted)")}
            onMouseLeave={e => (e.currentTarget.style.background = "var(--color-surface)")}>
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"
              style={{ color: "var(--color-foreground)", opacity: 0.7 }}>
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <div>
              <div style={{ fontSize: 16, fontWeight: 700 }}>Upload images</div>
              <div style={{ fontSize: 12, marginTop: 4, color: "var(--color-muted-foreground)" }}>JPEG or PNG · batch supported</div>
            </div>
          </button>
        </div>
      </div>

      {/* How it works */}
      <div style={{ borderTop: "1px solid var(--color-border)", padding: "40px 48px 48px", background: "var(--color-surface)" }}>
        <div style={{ maxWidth: 860, margin: "0 auto" }}>
          <div style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.14em", color: "var(--color-muted-foreground)", marginBottom: 28, textAlign: "center" }}>
            How it works
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 24 }}>
            {[
              {
                step: "01",
                title: "Select source",
                desc: "Choose one or more live EA webcam feeds, or upload your own JPEG / PNG images for offline inspection.",
                icon: (
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                    <path d="M23 7 16 12 23 17z"/><rect x="1" y="5" width="15" height="14" rx="2"/>
                  </svg>
                ),
              },
              {
                step: "02",
                title: "Run detection",
                desc: "ResNet-50 classifies each frame. Grad-CAM highlights the regions that influenced the prediction.",
                icon: (
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                    <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
                    <line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/>
                  </svg>
                ),
              },
              {
                step: "03",
                title: "AI inspection report",
                desc: "An LLM generates a structured report with observation, risk assessment, and recommended action for each camera.",
                icon: (
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                    <line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>
                  </svg>
                ),
              },
            ].map(({ step, title, desc, icon }) => (
              <div key={step} style={{
                padding: "24px 24px", borderRadius: 8,
                border: "1px solid var(--color-border)", background: "var(--color-background)",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14 }}>
                  <span style={{ fontSize: 11, fontWeight: 800, color: "var(--color-muted-foreground)", letterSpacing: "0.05em" }}>{step}</span>
                  <span style={{ color: "var(--color-foreground)", opacity: 0.6 }}>{icon}</span>
                </div>
                <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 8 }}>{title}</div>
                <div style={{ fontSize: 13, lineHeight: 1.6, color: "var(--color-muted-foreground)" }}>{desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

    </div>
  );
}

// ── Webcam pick screen ─────────────────────────────────────────────────────────

function WebcamPickScreen({
  onBack,
  onAnalyse,
}: {
  onBack:    () => void;
  onAnalyse: (items: ResultItem[]) => void;
}) {
  const { assets }  = useAssets();
  const [sel, setSel]         = useState<Set<string>>(new Set());
  const [offline, setOffline] = useState<Set<string>>(new Set());
  const [search, setSearch]   = useState("");
  const [region, setRegion]   = useState<"all" | "Cornwall" | "Devon">("all");

  const filtered = assets.filter(a => {
    const mr = region === "all" || a.region === region;
    const ms = !search || a.name.toLowerCase().includes(search.toLowerCase());
    return mr && ms;
  });

  const toggle = (id: string) => setSel(p => {
    const n = new Set(p); n.has(id) ? n.delete(id) : n.add(id); return n;
  });

  const allSel  = filtered.length > 0 && filtered.every(a => sel.has(a.id));
  const toggleAll = () => {
    if (allSel) setSel(p => { const n = new Set(p); filtered.forEach(a => n.delete(a.id)); return n; });
    else        setSel(p => { const n = new Set(p); filtered.forEach(a => n.add(a.id)); return n; });
  };

  const handleAnalyse = () => {
    const chosen = assets.filter(a => sel.has(a.id) && !offline.has(a.id));
    onAnalyse(chosen.map(a => ({
      id:      uid(),
      label:   a.name,
      region:  a.region,
      preview: webcamProxyUrl(a.campath),
      status:  "pending" as const,
      campath: a.campath,
      camName: a.name,
    })));
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Toolbar */}
      <div className="flex shrink-0 items-center gap-4 px-5 py-3"
        style={{ borderBottom: "1px solid var(--color-border)" }}>
        <button onClick={onBack}
          className="flex items-center gap-1.5 text-[12.5px] text-muted-foreground hover:text-foreground">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5M12 5l-7 7 7 7"/>
          </svg>
          Back
        </button>
        <div className="h-4 w-px" style={{ background: "var(--color-border)" }} />
        <h2 className="text-[14px] font-semibold flex-1">Select cameras</h2>

        {/* Search */}
        <input value={search} onChange={e => setSearch(e.target.value)}
          placeholder="Search…"
          className="w-44 rounded-[3px] px-3 py-1.5 text-[12px] outline-none"
          style={{ background: "var(--color-muted)", border: "1px solid var(--color-border)", color: "var(--color-foreground)" }} />

        {/* Region filter */}
        <div className="flex gap-1">
          {(["all", "Cornwall", "Devon"] as const).map(r => (
            <button key={r} onClick={() => setRegion(r)}
              className="rounded-[3px] px-3 py-1.5 text-[12px] font-medium transition-colors"
              style={{
                background: region === r ? "var(--color-foreground)" : "var(--color-muted)",
                color:      region === r ? "var(--color-background)" : "var(--color-muted-foreground)",
                border: "1px solid var(--color-border)",
              }}>
              {r === "all" ? "Both" : r}
            </button>
          ))}
        </div>

        {/* Analyse button */}
        <button
          onClick={handleAnalyse}
          disabled={sel.size === 0}
          className="rounded-[3px] px-5 py-1.5 text-[12.5px] font-semibold transition-colors disabled:opacity-30"
          style={{ background: "var(--color-foreground)", color: "var(--color-background)" }}>
          Analyse {sel.size > 0 ? sel.size : ""} {sel.size === 1 ? "camera" : "cameras"}
        </button>
      </div>

      {/* Select-all row */}
      <div className="flex shrink-0 items-center gap-3 px-5 py-2"
        style={{ borderBottom: "1px solid var(--color-border)", background: "var(--color-surface)" }}>
        <button onClick={toggleAll}
          className="flex items-center gap-2 text-[12px] text-muted-foreground hover:text-foreground">
          <Checkbox checked={allSel} />
          {allSel ? "Deselect all" : `Select all (${filtered.length})`}
        </button>
        {sel.size > 0 && (
          <span className="text-[11.5px] text-muted-foreground">{sel.size} selected</span>
        )}
      </div>

      {/* Camera grid */}
      <div className="min-h-0 flex-1 overflow-auto px-5 py-4">
        <div className="grid grid-cols-3 gap-4" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))" }}>
          {filtered.map(a => {
            const checked  = sel.has(a.id);
            const isOffline = offline.has(a.id);
            return (
              <button key={a.id}
                onClick={() => !isOffline && toggle(a.id)}
                className="group relative rounded-[5px] overflow-hidden text-left transition-all"
                style={{
                  border: isOffline  ? "2px solid var(--color-border)"     :
                          checked    ? "2px solid var(--color-foreground)"  :
                                       "2px solid var(--color-border)",
                  background: "var(--color-surface)",
                  opacity: isOffline ? 0.45 : 1,
                  cursor: isOffline ? "default" : "pointer",
                }}>
                {/* Thumbnail */}
                <div className="relative" style={{ aspectRatio: "16/9", background: "#111" }}>
                  <img src={webcamProxyUrl(a.campath)} alt={a.name}
                    className="h-full w-full object-cover"
                    onError={() => setOffline(p => { const n = new Set(p); n.add(a.id); setSel(s => { const s2 = new Set(s); s2.delete(a.id); return s2; }); return n; })} />
                  {isOffline ? (
                    <div className="absolute inset-0 flex items-center justify-center"
                      style={{ background: "rgba(0,0,0,0.55)" }}>
                      <span style={{ fontSize: 10.5, fontWeight: 700, letterSpacing: "0.1em", color: "rgba(255,255,255,0.5)", textTransform: "uppercase" }}>
                        Offline
                      </span>
                    </div>
                  ) : (
                    <div className="absolute top-2 right-2">
                      <Checkbox checked={checked} />
                    </div>
                  )}
                </div>
                {/* Label */}
                <div className="px-3 py-2.5">
                  <div className="text-[12.5px] font-semibold truncate">{a.name}</div>
                  <div className="mt-0.5" style={{ fontSize: 11, color: isOffline ? "var(--color-risk-high)" : "var(--color-muted-foreground)" }}>
                    {isOffline ? "No signal" : a.region}
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ── Upload pick screen ─────────────────────────────────────────────────────────

function UploadPickScreen({
  onBack,
  onAnalyse,
}: {
  onBack:    () => void;
  onAnalyse: (items: ResultItem[]) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const [staged,   setStaged]   = useState<{ id: string; file: File; preview: string }[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);

  const addFiles = (files: File[]) => {
    const imgs = files.filter(f => f.type.startsWith("image/"));
    if (!imgs.length) return;
    setStaged(prev => [
      ...prev,
      ...imgs.map(f => ({ id: uid(), file: f, preview: URL.createObjectURL(f) })),
    ]);
  };

  const remove = (id: string) =>
    setStaged(prev => prev.filter(s => s.id !== id));

  const handleAnalyse = () => {
    if (!staged.length) return;
    onAnalyse(staged.map(s => ({
      id:      s.id,
      label:   s.file.name.replace(/\.[^.]+$/, ""),
      preview: s.preview,
      status:  "pending" as const,
      file:    s.file,
    })));
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Toolbar */}
      <div className="flex shrink-0 items-center gap-4 px-5 py-3"
        style={{ borderBottom: "1px solid var(--color-border)" }}>
        <button onClick={onBack}
          className="flex items-center gap-1.5 text-[12.5px] text-muted-foreground hover:text-foreground">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5M12 5l-7 7 7 7"/>
          </svg>
          Back
        </button>
        <div className="h-4 w-px" style={{ background: "var(--color-border)" }} />
        <h2 className="text-[14px] font-semibold flex-1">Upload images</h2>

        {staged.length > 0 && (
          <>
            <button onClick={() => setStaged([])}
              className="rounded-[3px] px-4 py-1.5 text-[12px] font-medium transition-colors"
              style={{ background: "var(--color-muted)", color: "var(--color-muted-foreground)", border: "1px solid var(--color-border)" }}>
              Clear all
            </button>
            <button onClick={handleAnalyse}
              className="rounded-[3px] px-5 py-1.5 text-[12.5px] font-semibold"
              style={{ background: "var(--color-foreground)", color: "var(--color-background)" }}>
              Analyse {staged.length} image{staged.length !== 1 ? "s" : ""}
            </button>
          </>
        )}
      </div>

      {/* Drop zone — compact when images are staged */}
      <div
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => { e.preventDefault(); setDragging(false); addFiles(Array.from(e.dataTransfer.files)); }}
        onClick={() => fileRef.current?.click()}
        className="shrink-0 flex cursor-pointer items-center justify-center gap-4 transition-colors"
        style={{
          padding: staged.length ? "14px 24px" : "56px 24px",
          borderBottom: "1px solid var(--color-border)",
          background: dragging ? "var(--color-muted)" : staged.length ? "var(--color-surface)" : "transparent",
          borderColor: dragging ? "var(--color-foreground)" : "var(--color-border)",
        }}>
        <svg width={staged.length ? 18 : 40} height={staged.length ? 18 : 40}
          viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"
          style={{ color: "var(--color-muted-foreground)", flexShrink: 0 }}>
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/>
          <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
        {staged.length ? (
          <span style={{ fontSize: 13, color: "var(--color-muted-foreground)" }}>
            Drop more images here or click to browse
          </span>
        ) : (
          <div className="text-center">
            <div style={{ fontSize: 15, fontWeight: 600 }}>Drop images here</div>
            <div style={{ fontSize: 13, color: "var(--color-muted-foreground)", marginTop: 6 }}>or click to browse · JPEG · PNG</div>
          </div>
        )}
      </div>
      <input ref={fileRef} type="file" accept="image/jpeg,image/png" multiple
        className="hidden"
        onChange={e => { addFiles(Array.from(e.target.files ?? [])); e.target.value = ""; }} />

      {/* Staged image grid */}
      {staged.length > 0 ? (
        <div className="min-h-0 flex-1 overflow-auto p-4">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: 12 }}>
            {staged.map(s => (
              <div key={s.id} className="group relative rounded-[5px] overflow-hidden"
                style={{ border: "1px solid var(--color-border)", background: "var(--color-surface)", aspectRatio: "4/3" }}>
                <img src={s.preview} alt={s.file.name} className="h-full w-full object-cover" />
                {/* Remove button */}
                <button onClick={() => remove(s.id)}
                  className="absolute top-1.5 right-1.5 flex size-6 items-center justify-center rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ background: "rgba(0,0,0,0.7)" }}>
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
                    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                  </svg>
                </button>
                {/* Filename */}
                <div className="absolute bottom-0 inset-x-0 px-2 py-1.5"
                  style={{ background: "linear-gradient(transparent, rgba(0,0,0,0.65))" }}>
                  <div className="truncate" style={{ fontSize: 10.5, color: "rgba(255,255,255,0.85)", fontWeight: 500 }}>
                    {s.file.name}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="flex flex-1 items-center justify-center"
          style={{ color: "var(--color-muted-foreground)", fontSize: 13, opacity: 0.4 }}>
          No images selected yet
        </div>
      )}
    </div>
  );
}

// ── Results screen ─────────────────────────────────────────────────────────────

function ResultsScreen({
  initialItems,
  onBack,
}: {
  initialItems: ResultItem[];
  onBack:       () => void;
}) {
  const [items,    setItems]   = useState<ResultItem[]>(initialItems);
  const [activeId, setActiveId] = useState<string>(initialItems[0]?.id ?? "");
  const [reclassifying, setReclassifying] = useState<Set<string>>(new Set());
  const runningRef   = useRef(false);
  const queueRef     = useRef<ResultItem[]>([]);
  const processedRef = useRef<Set<string>>(new Set());
  const addMoreRef   = useRef<HTMLInputElement>(null);

  // Start analysis immediately on mount
  useEffect(() => {
    runAll(initialItems);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Manually resolve an UNDER REVIEW (FLAGGED) result — webcam items only.
  // Uploaded files never get a campath/DB row (POST /predict doesn't persist
  // anything — there's no recurring camera identity to key a correction by),
  // so this is only meaningful for items analysed from the live-webcam picker.
  const handleReclassify = async (item: ResultItem, newRisk: "BLOCKED" | "LOW") => {
    if (!item.campath) return;
    setReclassifying(prev => new Set(prev).add(item.id));
    let response;
    try {
      response = await reclassifyResult(item.campath, newRisk);
    } catch (e) {
      console.error("Reclassify failed:", e);
      alert("Could not save the correction — check the backend connection.");
      setReclassifying(prev => { const n = new Set(prev); n.delete(item.id); return n; });
      return;
    }
    const label = newRisk === "BLOCKED" ? "BLOCKED" : "CLEAR";
    setItems(prev => prev.map(i => {
      if (i.id !== item.id || !i.result) return i;
      return {
        ...i,
        result: {
          ...i.result,
          risk: newRisk,
          prediction: label,
          p_blocked: newRisk === "BLOCKED" ? 1 : 0,
          // Marking BLOCKED generates a real GradCAM overlay + LLM report
          // server-side (the automatic pipeline never runs either for a
          // FLAGGED result) — use them if present; fall back to a generic
          // note only if the backend had nothing stored to run GradCAM on.
          explanation: response.explanation ?? `Manually reclassified as ${label} by reviewer.`,
          ...(response.overlay_generated ? {
            overlay_b64:      response.overlay_b64 ?? i.result.overlay_b64,
            heatmap_b64:      response.heatmap_b64 ?? i.result.heatmap_b64,
            heatmap_coverage: response.heatmap_coverage ?? i.result.heatmap_coverage,
          } : {}),
        },
      };
    }));
    setReclassifying(prev => { const n = new Set(prev); n.delete(item.id); return n; });
  };

  const runAll = async (toRun: ResultItem[]) => {
    // Skip items already queued or processed — guards against React Strict Mode
    // double-invoking effects, which would otherwise restart analysis after completion.
    const fresh = toRun.filter(i => !processedRef.current.has(i.id));
    if (!fresh.length) return;
    fresh.forEach(i => processedRef.current.add(i.id));

    queueRef.current = [...queueRef.current, ...fresh];
    if (runningRef.current) return;
    runningRef.current = true;

    while (queueRef.current.length > 0) {
      const item = queueRef.current.shift()!;
      setItems(prev => prev.map(i => i.id === item.id ? { ...i, status: "running" } : i));
      setActiveId(item.id);
      try {
        let result: PredictResult;
        if (item.file) {
          result = await analyzeUpload(item.file);
        } else if (item.campath) {
          result = await analyzeWebcam(item.campath, item.camName ?? "");
        } else {
          throw new Error("Unknown source");
        }
        setItems(prev => prev.map(i => i.id === item.id ? { ...i, status: "done", result } : i));
      } catch (err) {
        const error = err instanceof Error ? err.message : "Failed";
        setItems(prev => prev.map(i => i.id === item.id ? { ...i, status: "error", error } : i));
      }
    }
    runningRef.current = false;
  };

  // Add more images — appends to existing results without replacing
  const addMoreFiles = (files: File[]) => {
    const imgs = files.filter(f => f.type.startsWith("image/"));
    if (!imgs.length) return;
    const newItems: ResultItem[] = imgs.map(f => ({
      id:      uid(),
      label:   f.name.replace(/\.[^.]+$/, ""),
      preview: URL.createObjectURL(f),
      status:  "pending" as const,
      file:    f,
    }));
    setItems(prev => [...prev, ...newItems]);
    runAll(newItems);
  };

  const active = items.find(i => i.id === activeId) ?? items[0];
  const r = active?.result;

  return (
    <div className="flex min-h-0 flex-1 overflow-hidden">

      {/* Left sidebar: list of results */}
      <aside className="flex w-64 shrink-0 flex-col"
        style={{ borderRight: "1px solid var(--color-border)", background: "var(--color-panel)" }}>

        {/* Toolbar */}
        <div className="flex shrink-0 items-center gap-2 px-3 py-3"
          style={{ borderBottom: "1px solid var(--color-border)" }}>
          <button onClick={onBack}
            className="flex items-center gap-1.5 text-[12px] text-muted-foreground hover:text-foreground">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M19 12H5M12 5l-7 7 7 7"/>
            </svg>
            Back
          </button>
          <span className="flex-1" />
          {/* Clear all */}
          <button
            onClick={() => { setItems([]); setActiveId(""); }}
            title="Clear all"
            className="flex items-center justify-center rounded-[3px] p-1 transition-colors hover:bg-muted"
            style={{ color: "var(--color-muted-foreground)" }}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
              <path d="M10 11v6M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
            </svg>
          </button>
          {/* Add more (upload only) */}
          <button
            onClick={() => addMoreRef.current?.click()}
            title="Add more images"
            className="flex items-center justify-center rounded-[3px] p-1 transition-colors hover:bg-muted"
            style={{ color: "var(--color-muted-foreground)" }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
            </svg>
          </button>
          <input ref={addMoreRef} type="file" accept="image/jpeg,image/png" multiple className="hidden"
            onChange={e => { addMoreFiles(Array.from(e.target.files ?? [])); e.target.value = ""; }} />
          <span style={{ fontSize: 11, color: "var(--color-muted-foreground)" }}>
            {items.length} image{items.length !== 1 ? "s" : ""}
          </span>
        </div>

        {/* Camera list */}
        <ul className="min-h-0 flex-1 overflow-auto">
          {items.map(item => {
            const isActive = item.id === activeId;
            return (
              <li key={item.id}>
                <button onClick={() => setActiveId(item.id)}
                  className="flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors"
                  style={{
                    borderBottom: "1px solid var(--color-border)",
                    background: isActive ? "var(--color-muted)" : "transparent",
                    borderLeft: `3px solid ${isActive ? "var(--color-foreground)" : "transparent"}`,
                  }}>
                  {/* Thumbnail */}
                  <img src={item.preview} alt={item.label}
                    className="h-9 w-14 shrink-0 rounded-[2px] object-cover"
                    style={{ border: "1px solid var(--color-border)" }} />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-[12px] font-medium">{item.label}</div>
                    <div className="mt-0.5 text-[10.5px]"
                      style={{
                        color: item.status === "done"    ? riskColor(item.result?.risk) :
                               item.status === "running" ? "var(--color-risk-medium)" :
                               item.status === "error"   ? "var(--color-muted-foreground)" :
                               "var(--color-muted-foreground)",
                      }}>
                      {item.status === "done"    ? predLabel(item.result) :
                       item.status === "running" ? "Analysing…" :
                       item.status === "error"   ? (item.campath ? "Offline" : "Error") : "Pending"}
                    </div>
                  </div>
                  {/* Status dot */}
                  <span className="size-1.5 shrink-0 rounded-full"
                    style={{
                      background: item.status === "done"    ? riskColor(item.result?.risk) :
                                  item.status === "running" ? "var(--color-risk-medium)" :
                                  item.status === "error"   ? "var(--color-border)" :
                                  "var(--color-border)",
                    }} />
                </button>
              </li>
            );
          })}
        </ul>
      </aside>

      {/* Right: single scrolling column — header, images, then report */}
      {active ? (
        <div className="min-h-0 flex-1 overflow-y-auto" style={{ background: "var(--color-background)" }}>

          {/* Header — sticky so camera name stays visible while scrolling */}
          <div className="sticky top-0 z-10 flex items-center justify-between px-6 py-3"
            style={{ borderBottom: "1px solid var(--color-border)", background: "var(--color-surface)" }}>
            <div>
              {active.region && (
                <div style={{ fontSize: 11, color: "var(--color-muted-foreground)" }}>{active.region}</div>
              )}
              <div style={{ fontSize: 15, fontWeight: 600 }}>{active.label}</div>
            </div>
            {active.status === "error" && active.campath ? (
              <span style={{ fontSize: 13, fontWeight: 600, color: "var(--color-muted-foreground)" }}>
                OFFLINE
              </span>
            ) : r ? (
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <span style={{ fontSize: 14, fontWeight: 700, color: riskColor(r.risk) }}>
                  {predLabel(r)}
                </span>
                {/* Manual review buttons — webcam-sourced items only (uploads
                    have no campath/DB row to correct). REVIEW_ALL_CASES
                    (lib/api.ts) toggles between every result being
                    correctable (review passes) and FLAGGED-only (launch
                    mode), matching the main Live Monitor tab's behaviour. */}
                {active.campath && (REVIEW_ALL_CASES || active.result?.risk === "FLAGGED") && (
                  <>
                    {active.result?.risk !== "BLOCKED" && (
                      <button
                        disabled={reclassifying.has(active.id)}
                        onClick={() => handleReclassify(active, "BLOCKED")}
                        className="rounded-[3px] bg-transparent px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide transition-colors hover:bg-[color-mix(in_oklch,var(--color-risk-high)_10%,transparent)] disabled:opacity-40"
                        style={{ border: "1.5px solid var(--color-risk-high)", color: "var(--color-risk-high)" }}>
                        Mark Blocked
                      </button>
                    )}
                    {active.result?.risk !== "LOW" && (
                      <button
                        disabled={reclassifying.has(active.id)}
                        onClick={() => handleReclassify(active, "LOW")}
                        className="rounded-[3px] bg-transparent px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide transition-colors hover:bg-[color-mix(in_oklch,var(--color-ok)_10%,transparent)] disabled:opacity-40"
                        style={{ border: "1.5px solid var(--color-ok)", color: "var(--color-ok)" }}>
                        Mark Clear
                      </button>
                    )}
                    {reclassifying.has(active.id) && (
                      <span style={{ fontSize: 11, color: "var(--color-muted-foreground)" }}>saving…</span>
                    )}
                  </>
                )}
              </div>
            ) : null}
          </div>

          {/* Images — single for CLEAR, side-by-side for BLOCKED/FLAGGED */}
          {(() => {
            const isClear = r?.prediction === "CLEAR" || r?.prediction === "OTHER" || active.status === "running" && !r;
            const showSingle = active.status === "done" && (r?.prediction === "CLEAR" || r?.prediction === "OTHER");

            return (
              <div style={{
                display: "grid",
                gridTemplateColumns: showSingle ? "1fr" : "1fr 1fr",
                borderBottom: "1px solid var(--color-border)",
              }}>
                {/* Original */}
                <div className="relative bg-black"
                  style={{ aspectRatio: "16/9", borderRight: showSingle ? "none" : "1px solid var(--color-border)" }}>
                  <img src={active.preview} alt="Original" className="h-full w-full object-contain" />
                  <span className="absolute bottom-2 left-3 pointer-events-none"
                    style={{ fontSize: 10, fontWeight: 500, color: "rgba(255,255,255,0.4)" }}>
                    Original
                  </span>
                  {showSingle && (
                    <span className="absolute bottom-2 right-3 pointer-events-none"
                      style={{ fontSize: 12, fontWeight: 700, color: riskColor(r!.risk) }}>
                      {predLabel(r)}
                    </span>
                  )}
                </div>

                {/* Grad-CAM — only for non-CLEAR results */}
                {!showSingle && (
                  <div className="relative bg-black" style={{ aspectRatio: "16/9" }}>
                    {active.status === "running" ? (
                      <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
                        <span className="size-6 animate-spin rounded-full border-2"
                          style={{ borderColor: "var(--color-muted)", borderTopColor: "var(--color-foreground)" }} />
                        <span style={{ fontSize: 12, color: "var(--color-muted-foreground)" }}>Running model…</span>
                      </div>
                    ) : r?.overlay_b64 ? (
                      <>
                        <img src={`data:image/jpeg;base64,${r.overlay_b64}`}
                          alt="Grad-CAM" className="h-full w-full object-contain" />
                        <span className="absolute bottom-2 left-3 pointer-events-none"
                          style={{ fontSize: 10, fontWeight: 500, color: "rgba(255,255,255,0.4)" }}>
                          Grad-CAM
                        </span>
                        <span className="absolute bottom-2 right-3 pointer-events-none"
                          style={{ fontSize: 12, fontWeight: 700, color: riskColor(r.risk) }}>
                          {predLabel(r)}
                        </span>
                      </>
                    ) : active.status === "error" ? (
                      <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-6 text-center">
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" style={{ color: "rgba(255,255,255,0.2)" }}>
                          <line x1="1" y1="1" x2="23" y2="23"/><path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6"/>
                          <path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23"/>
                          <line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/>
                        </svg>
                        <span style={{ fontSize: 11.5, color: "rgba(255,255,255,0.25)", letterSpacing: "0.08em", textTransform: "uppercase", fontWeight: 600 }}>
                          {active.campath ? "Camera offline" : "Analysis failed"}
                        </span>
                      </div>
                    ) : (
                      <div className="absolute inset-0 flex items-center justify-center"
                        style={{ fontSize: 12, color: "rgba(255,255,255,0.2)" }}>
                        Grad-CAM appears after analysis
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })()}

          {/* Quality flags */}
          {(r?.quality_flags?.length ?? 0) > 0 && (
            <div style={{
              display: "flex", alignItems: "center", gap: 12, padding: "10px 24px",
              borderBottom: "1px solid var(--color-border)",
              background: "color-mix(in oklch, var(--color-risk-medium) 7%, transparent)",
            }}>
              <span style={{ fontSize: 10.5, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.1em", color: "var(--color-risk-medium)" }}>
                Quality flags
              </span>
              <span style={{ fontSize: 12, color: "var(--color-muted-foreground)" }}>
                {r!.quality_flags.join(" · ")}
              </span>
            </div>
          )}

          {/* LLM report — always below images in natural flow */}
          <div style={{ padding: "36px 40px 72px", maxWidth: 960, margin: "0 auto", width: "100%" }}>
            {active.status === "running" ? (
              <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "48px 0", color: "var(--color-muted-foreground)", fontSize: 15 }}>
                <span className="size-5 animate-spin rounded-full border-2"
                  style={{ borderColor: "var(--color-muted)", borderTopColor: "var(--color-foreground)" }} />
                Generating inspection report…
              </div>
            ) : r?.explanation ? (
              <>
                {/* Report header */}
                <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 28 }}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" style={{ color: "var(--color-muted-foreground)", flexShrink: 0 }}>
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                    <line x1="16" y1="13" x2="8" y2="13"/>
                    <line x1="16" y1="17" x2="8" y2="17"/>
                    <polyline points="10 9 9 9 8 9"/>
                  </svg>
                  <span style={{ fontSize: 13, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.14em", color: "var(--color-foreground)" }}>
                    AI Inspection Report
                  </span>
                  <div style={{ flex: 1, height: 1, background: "var(--color-border)" }} />
                  <span style={{ fontSize: 11, color: "var(--color-muted-foreground)", whiteSpace: "nowrap" }}>
                    Powered by Cerebras LLM
                  </span>
                </div>
                <ExplanationBlock text={r.explanation} risk={r.risk} />
              </>
            ) : active.status === "done" ? (
              <div style={{ padding: "48px 0", fontSize: 14, color: "var(--color-muted-foreground)", opacity: 0.4 }}>
                LLM report unavailable
              </div>
            ) : null}
          </div>

        </div>
      ) : null}
    </div>
  );
}

// ── Checkbox ───────────────────────────────────────────────────────────────────

function Checkbox({ checked }: { checked: boolean }) {
  return (
    <span className="inline-flex size-4 items-center justify-center rounded-sm border transition-colors"
      style={{
        background:  checked ? "var(--color-foreground)" : "rgba(255,255,255,0.15)",
        borderColor: checked ? "var(--color-foreground)" : "rgba(255,255,255,0.5)",
        backdropFilter: "blur(2px)",
      }}>
      {checked && (
        <svg width="9" height="7" viewBox="0 0 9 7" fill="none">
          <path d="M1 3.5l2.5 2.5 4.5-5" stroke="var(--color-background)" strokeWidth="1.6" strokeLinecap="round"/>
        </svg>
      )}
    </span>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────────

function AnalysePage() {
  const [screen, setScreen] = useState<Screen>({ kind: "home" });

  return (
    <AppShell>
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">

        {screen.kind === "home" && (
          <HomeScreen
            onChoose={mode =>
              setScreen({ kind: mode === "webcam" ? "webcam-pick" : "upload-pick" })
            }
          />
        )}

        {screen.kind === "webcam-pick" && (
          <WebcamPickScreen
            onBack={() => setScreen({ kind: "home" })}
            onAnalyse={items => setScreen({ kind: "results", items })}
          />
        )}

        {screen.kind === "upload-pick" && (
          <UploadPickScreen
            onBack={() => setScreen({ kind: "home" })}
            onAnalyse={items => setScreen({ kind: "results", items })}
          />
        )}

        {screen.kind === "results" && (
          <ResultsScreen
            initialItems={screen.items}
            onBack={() => setScreen({ kind: "home" })}
          />
        )}

      </div>
    </AppShell>
  );
}
