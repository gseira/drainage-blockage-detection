/**
 * metrics.tsx — Metrics & Reports page
 * -------------------------------------
 * KPI summary cards, daily HIGH/LOW trend chart (recharts),
 * region breakdown, and a PDF export via window.print().
 */

import { useEffect, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line,
} from "recharts";
import { fetchStats, fetchLatestResults, type StatRow, type ResultRow } from "@/lib/api";
import { AppShell } from "@/components/dashboard/app-shell";
import { STATIC_ASSETS } from "@/lib/dashboard-data";

export const Route = createFileRoute("/metrics")({
  head: () => ({
    meta: [{ title: "DrainWatch — Metrics & Reports" }],
  }),
  component: MetricsPage,
});

function MetricsPage() {
  const [days, setDays]           = useState(30);
  const [stats, setStats]         = useState<StatRow[]>([]);
  const [latest, setLatest]       = useState<ResultRow[]>([]);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([fetchStats(days), fetchLatestResults()])
      .then(([s, l]) => { setStats(s); setLatest(l); })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [days]);

  // ── KPI calculations ─────────────────────────────────────────────────────────
  const totalScans    = stats.reduce((s, r) => s + r.total, 0);
  const totalHigh     = stats.reduce((s, r) => s + r.high, 0);
  const totalLow      = stats.reduce((s, r) => s + r.low, 0);
  const avgPBlocked   = stats.length ? (stats.reduce((s, r) => s + (r.avg_p_blocked ?? 0), 0) / stats.length * 100) : 0;

  const latestHigh    = latest.filter((r) => r.risk === "HIGH").length;
  const latestLow     = latest.filter((r) => r.risk === "LOW").length;
  const latestOffline = latest.filter((r) => r.risk === "OFFLINE").length;
  const latestFlagged = latest.filter((r) => r.risk === "FLAGGED").length;

  // Cornwall vs Devon from latest
  const cornwallHigh  = latest.filter((r) => r.campath.startsWith("cornwall/") && r.risk === "HIGH").length;
  const devonHigh     = latest.filter((r) => !r.campath.startsWith("cornwall/") && r.risk === "HIGH").length;
  const cornwallTotal = latest.filter((r) => r.campath.startsWith("cornwall/")).length;
  const devonTotal    = latest.filter((r) => !r.campath.startsWith("cornwall/")).length;

  // Chart data: abbreviate date
  const chartData = stats.map((r) => ({
    day:         r.day.slice(5),  // MM-DD
    HIGH:        r.high,
    LOW:         r.low,
    total:       r.total,
    avg_blocked: parseFloat((r.avg_p_blocked * 100).toFixed(1)),
  }));

  // Use static assets count if backend offline
  const camCount = latest.length > 0 ? latest.length : STATIC_ASSETS.length;

  const handlePrint = () => {
    window.print();
  };

  return (
    <AppShell>
      <style>{`
        @media print {
          header, footer, nav, .no-print { display: none !important; }
          body { background: white !important; color: black !important; }
          .print-break { page-break-before: always; }
        }
      `}</style>

      {/* Page header */}
      <div className="flex shrink-0 items-end justify-between border-b border-border bg-surface px-4 pb-3 pt-3">
        <div>
          <div className="label-eyebrow">Analytics</div>
          <h1 className="text-[18px] font-semibold leading-tight tracking-tight">Metrics & Reports</h1>
          <div className="mt-0.5 text-[12px] text-muted-foreground">
            {camCount} cameras · Devon & Cornwall SW Region
          </div>
        </div>
        <div className="no-print flex items-center gap-2">
          {/* Days selector */}
          <div className="flex items-center gap-1 rounded-[3px] border border-border text-[11px]">
            {([7, 14, 30] as const).map((d, i) => (
              <button key={d} onClick={() => setDays(d)}
                className={`px-2.5 py-1 ${i > 0 ? "border-l border-border" : ""} ${
                  days === d ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground"
                }`}>
                {d}d
              </button>
            ))}
          </div>
          <button onClick={handlePrint}
            className="flex items-center gap-1.5 rounded-[3px] bg-foreground px-3 py-1.5 text-[12px] font-medium text-background hover:bg-foreground/90 transition-colors">
            ⬇ Export PDF
          </button>
        </div>
      </div>

      {loading && (
        <div className="flex flex-1 items-center justify-center text-[12px] text-muted-foreground">
          Loading metrics…
        </div>
      )}

      {error && (
        <div className="m-4 rounded-[3px] border border-[var(--color-risk-high)]/30 bg-[var(--color-risk-high)]/10 px-4 py-3 text-[12px] text-[var(--color-risk-high)]">
          Backend offline — metrics unavailable. Start the FastAPI server at localhost:8000.
        </div>
      )}

      {!loading && (
        <div className="flex-1 overflow-auto p-4 space-y-6">
          {/* ── KPI Cards ── */}
          <div>
            <h2 className="label-eyebrow mb-3">Current status</h2>
            <div className="grid grid-cols-4 gap-3">
              <KpiCard label="Cameras monitored" value={String(camCount)} />
              <KpiCard label="Currently BLOCKED" value={String(latestHigh)} accent="high" />
              <KpiCard label="Currently CLEAR" value={String(latestLow)} accent="low" />
              <KpiCard label="Offline / Flagged" value={String(latestOffline + latestFlagged)} />
            </div>
          </div>

          {/* ── Trend KPIs (only if backend live) ── */}
          {stats.length > 0 && (
            <div>
              <h2 className="label-eyebrow mb-3">Last {days} days</h2>
              <div className="grid grid-cols-4 gap-3">
                <KpiCard label="Total scans" value={String(totalScans)} />
                <KpiCard label="HIGH risk scans" value={String(totalHigh)} accent="high" />
                <KpiCard label="LOW risk scans"  value={String(totalLow)}  accent="low" />
                <KpiCard label="Avg p_blocked"   value={`${avgPBlocked.toFixed(1)}%`} />
              </div>
            </div>
          )}

          {/* ── Region breakdown ── */}
          {latest.length > 0 && (
            <div>
              <h2 className="label-eyebrow mb-3">Region breakdown (latest scan)</h2>
              <div className="grid grid-cols-2 gap-3">
                <RegionCard region="Cornwall" high={cornwallHigh} total={cornwallTotal} />
                <RegionCard region="Devon"    high={devonHigh}    total={devonTotal}    />
              </div>
            </div>
          )}

          {/* ── Daily HIGH/LOW bar chart ── */}
          {chartData.length > 0 && (
            <div>
              <h2 className="label-eyebrow mb-3">Daily scan results — last {days} days</h2>
              <div className="rounded-[3px] border border-border bg-surface p-4">
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={chartData} barCategoryGap="30%">
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                    <XAxis dataKey="day" tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }} />
                    <YAxis tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }} allowDecimals={false} />
                    <Tooltip
                      contentStyle={{ background: "var(--color-surface)", border: "1px solid var(--color-border)", borderRadius: "3px", fontSize: 12 }}
                      labelStyle={{ color: "var(--color-foreground)" }}
                    />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="HIGH" fill="var(--color-risk-high)"    name="HIGH risk" />
                    <Bar dataKey="LOW"  fill="var(--color-risk-low)"     name="LOW risk"  />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* ── Average p_blocked trend line ── */}
          {chartData.length > 0 && (
            <div>
              <h2 className="label-eyebrow mb-3">Average blockage probability trend</h2>
              <div className="rounded-[3px] border border-border bg-surface p-4">
                <ResponsiveContainer width="100%" height={160}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                    <XAxis dataKey="day" tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }}
                      tickFormatter={(v) => `${v}%`} />
                    <Tooltip
                      formatter={(v: number) => [`${v}%`, "avg p_blocked"]}
                      contentStyle={{ background: "var(--color-surface)", border: "1px solid var(--color-border)", borderRadius: "3px", fontSize: 12 }}
                    />
                    <Line dataKey="avg_blocked" stroke="var(--color-risk-high)" strokeWidth={2}
                      dot={{ r: 3 }} name="avg p_blocked (%)" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* ── Camera detail table ── */}
          {latest.length > 0 && (
            <div className="print-break">
              <h2 className="label-eyebrow mb-3">All cameras — latest result</h2>
              <div className="rounded-[3px] border border-border bg-surface overflow-hidden">
                <table className="w-full border-separate border-spacing-0 text-[12px]">
                  <thead className="sticky top-0 bg-surface">
                    <tr>
                      {["Camera", "Region", "Risk", "p_blocked", "Heatmap cov.", "Captured at", "Scan at"].map((h, i) => (
                        <th key={h} className={`border-b border-border px-3 py-2 label-eyebrow text-left ${i >= 3 ? "text-right" : ""}`}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {latest.map((r) => (
                      <tr key={r.id} className="hover:bg-muted/40 transition-colors">
                        <td className="border-b border-border px-3 py-1.5 font-medium">{r.name}</td>
                        <td className="border-b border-border px-3 py-1.5 text-muted-foreground">
                          {r.campath.startsWith("cornwall/") ? "Cornwall" : "Devon"}
                        </td>
                        <td className={`border-b border-border px-3 py-1.5 font-semibold mono ${
                          r.risk === "HIGH" ? "text-[var(--color-risk-high)]" :
                          r.risk === "LOW"  ? "text-[var(--color-risk-low)]"  : "text-muted-foreground"
                        }`}>{r.risk}</td>
                        <td className="border-b border-border px-3 py-1.5 text-right mono tnum">
                          {r.p_blocked !== null ? `${(r.p_blocked * 100).toFixed(1)}%` : "—"}
                        </td>
                        <td className="border-b border-border px-3 py-1.5 text-right mono tnum text-muted-foreground">
                          {r.heatmap_coverage !== null ? `${((r.heatmap_coverage ?? 0) * 100).toFixed(0)}%` : "—"}
                        </td>
                        <td className="border-b border-border px-3 py-1.5 text-right mono text-muted-foreground">{r.captured_at}</td>
                        <td className="border-b border-border px-3 py-1.5 text-right mono text-muted-foreground">
                          {new Date(r.created_at).toLocaleString("en-GB", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" })}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Footer for print */}
          <div className="hidden print:block border-t border-border pt-4 text-[11px] text-muted-foreground">
            DrainWatch · ResNet-50 + Grad-CAM · SW Region Devon & Cornwall ·
            Report generated {new Date().toLocaleString("en-GB")}
          </div>
        </div>
      )}
    </AppShell>
  );
}

function KpiCard({ label, value, accent }: { label: string; value: string; accent?: "high" | "low" }) {
  const colour = accent === "high" ? "text-[var(--color-risk-high)]"
               : accent === "low"  ? "text-[var(--color-risk-low)]"
               : "text-foreground";
  return (
    <div className="rounded-[3px] border border-border bg-surface px-4 py-3">
      <div className="label-eyebrow mb-1">{label}</div>
      <div className={`mono tnum text-[28px] font-semibold leading-none ${colour}`}>{value}</div>
    </div>
  );
}

function RegionCard({ region, high, total }: { region: string; high: number; total: number }) {
  const pct = total > 0 ? Math.round((high / total) * 100) : 0;
  return (
    <div className="rounded-[3px] border border-border bg-surface px-4 py-3">
      <div className="flex items-baseline justify-between">
        <div className="text-[13px] font-semibold">{region}</div>
        <div className="mono tnum text-[12px] text-muted-foreground">{total} cameras</div>
      </div>
      <div className="mt-2">
        <div className="flex justify-between text-[12px]">
          <span className="text-muted-foreground">HIGH risk</span>
          <span className="mono text-[var(--color-risk-high)] font-semibold">{high} ({pct}%)</span>
        </div>
        <div className="mt-1 h-1.5 w-full rounded-full bg-border overflow-hidden">
          <div className="h-full bg-[var(--color-risk-high)] transition-all"
            style={{ width: `${pct}%` }} />
        </div>
      </div>
    </div>
  );
}
