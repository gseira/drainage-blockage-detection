import { useMemo, useState, useCallback, useEffect, useRef, lazy, Suspense } from "react";
import { createFileRoute, ClientOnly } from "@tanstack/react-router";
import jsPDF from "jspdf";
import { useAssets } from "@/hooks/useAssets";
import { type Asset, type Risk } from "@/lib/dashboard-data";
import { analyzeWebcam, fetchHealth, webcamProxyUrl, fetchStaticMap, reclassifyResult, REVIEW_ALL_CASES, type PredictResult, type StaticMapMarker, type StaticMapRoute } from "@/lib/api";
import { AppShell } from "@/components/dashboard/app-shell";
import type { MapRoute, MapSummary, BatchRouteMapHandle } from "@/components/dashboard/batch-route-map";

const BatchRouteMap = lazy(() =>
  import("@/components/dashboard/batch-route-map").then(m => ({ default: m.BatchRouteMap }))
);

export const Route = createFileRoute("/")({
  head: () => ({ meta: [{ title: "DrainWatch — Monitor" }] }),
  component: Dashboard,
});

type RegionFilter = "all" | "Cornwall" | "Devon";

// ── Spatial helpers ───────────────────────────────────────────────────────────

function distKm(lat1: number, lng1: number, lat2: number, lng2: number) {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLng = ((lng2 - lng1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) *
    Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

type RouteCamera = Asset & { km: number };

/** All permutations of an array. Only use for small N (≤ 7). */
function permute<T>(arr: T[]): T[][] {
  if (arr.length <= 1) return [arr];
  const out: T[][] = [];
  for (let i = 0; i < arr.length; i++) {
    const rest = arr.filter((_, j) => j !== i);
    for (const p of permute(rest)) out.push([arr[i], ...p]);
  }
  return out;
}

/**
 * Optimal route ordering for small clusters (N ≤ 8).
 * Exhaustively searches ALL permutations (no fixed start) so the solver
 * can choose whichever entry point produces the shortest one-way path.
 * This avoids "go there, come back past it, go there again" patterns.
 */
function optimalOrder<T extends { lat: number; lng: number }>(cameras: T[]): T[] {
  if (cameras.length <= 2) return cameras;
  let bestOrder = cameras;
  let bestDist  = routeLength(cameras);
  for (const perm of permute(cameras)) {
    const d = routeLength(perm);
    if (d < bestDist) { bestDist = d; bestOrder = perm; }
  }
  return bestOrder;
}

function routeLength(cameras: { lat: number; lng: number }[]): number {
  let d = 0;
  for (let i = 1; i < cameras.length; i++)
    d += distKm(cameras[i-1].lat, cameras[i-1].lng, cameras[i].lat, cameras[i].lng);
  return d;
}

/**
 * Re-orders cameras using actual road travel times from OSRM's table service.
 * Falls back to the input order if the request fails or is aborted.
 */
async function reorderByRoadTime<T extends { lat: number; lng: number }>(
  cameras: T[],
  signal: AbortSignal,
): Promise<T[]> {
  if (cameras.length <= 2) return cameras;
  const coords = cameras.map(c => `${c.lng},${c.lat}`).join(";");
  try {
    const res = await fetch(
      `https://router.project-osrm.org/table/v1/driving/${coords}?annotations=duration`,
      { signal },
    );
    if (!res.ok) throw new Error(`OSRM table ${res.status}`);
    const data = await res.json();
    const matrix: number[][] = data.durations;
    // Exhaustive open-path TSP on the real duration matrix
    const indices = Array.from({ length: cameras.length }, (_, i) => i);
    let bestPerm  = indices;
    let bestCost  = Infinity;
    for (const perm of permute(indices)) {
      let cost = 0;
      for (let i = 1; i < perm.length; i++) cost += matrix[perm[i - 1]][perm[i]];
      if (cost < bestCost) { bestCost = cost; bestPerm = perm; }
    }
    return bestPerm.map(i => cameras[i]);
  } catch {
    return cameras; // fallback: keep straight-line order
  }
}

// ── Route decision types ──────────────────────────────────────────────────────

interface RouteDecision {
  type: "immediate" | "bundle" | "monitor";
  cameras: RouteCamera[];
  totalKm: number;
  reason: string;
}

function buildRoute(analysed: Asset, p: number, allAssets: Asset[]): RouteDecision {
  const self: RouteCamera = { ...analysed, km: 0 };
  const candidates: RouteCamera[] = allAssets
    .filter(a => a.id !== analysed.id && a.region === analysed.region && a.risk === "high")
    .map(a => ({ ...a, km: distKm(analysed.lat, analysed.lng, a.lat, a.lng) }))
    .sort((a, b) => a.km - b.km);

  const within15 = candidates.filter(a => a.km <= 15);
  const within12 = candidates.filter(a => a.km <= 12);

  if (p >= 0.82) {
    const bundle = optimalOrder([self, ...within15.slice(0, 4)]);
    const totalKm = routeLength(bundle);
    return {
      type: "immediate", cameras: bundle, totalKm,
      reason: p >= 0.92
        ? "Critical blockage confirmed. Dispatch crew immediately."
        : "High-confidence blockage. Schedule crew visit within 24 hours.",
    };
  }
  if (p >= 0.62) {
    if (within12.length >= 2) {
      const cluster = [self, ...within12.slice(0, 4)];
      const ordered = optimalOrder(cluster);
      const totalKm = routeLength(ordered);
      if (ordered.length / Math.max(totalKm, 0.1) >= 0.18)
        return { type: "bundle", cameras: ordered, totalKm,
          reason: `${ordered.length} blocked cameras within ${totalKm.toFixed(1)} km — bundled visit is cost-effective.` };
    }
    if (within12.length >= 1 && within12[0].km <= 6) {
      const pair = optimalOrder([self, within12[0]]);
      return { type: "bundle", cameras: pair, totalKm: routeLength(pair),
        reason: `Adjacent blockage at ${within12[0].name} (${within12[0].km.toFixed(1)} km). Combined visit recommended.` };
    }
    return { type: "monitor", cameras: [self], totalKm: 0,
      reason: "Moderate blockage, no nearby cluster. Add to next maintenance sweep." };
  }
  return { type: "monitor", cameras: [self], totalKm: 0,
    reason: "Low-confidence detection. Continue monitoring — re-scan in next scheduled sweep." };
}

// ── Batch route types ─────────────────────────────────────────────────────────

type ScannedAsset = Asset & { pFresh: number; riskFresh: string; result: PredictResult };
type BatchRouteCamera = ScannedAsset & { km: number };

/** Operational issue (camera offline / not pointed at a drain) vs a genuine
 *  FLAGGED scene the model is just uncertain about — see the quality_flags
 *  categories set server-side ("offline" / "no_drainage" / "low_margin"). */
function isOfflineOrNoDrainage(a: ScannedAsset): boolean {
  const flags = a.result?.quality_flags ?? [];
  return flags.includes("offline") || flags.includes("no_drainage");
}

interface BatchRoute {
  type: "immediate" | "bundle" | "monitor";
  cameras: BatchRouteCamera[];
  totalKm: number;
  summary: string;
}

interface BatchAnalysis {
  region: string;
  routes: BatchRoute[];
  lightBlocked: ScannedAsset[];   // blocked but isolated (no nearby camera to route with)
  scanned: number;
  blocked: number;
  clear: number;
  flagged: number;
  blockedAssets: ScannedAsset[];
  allScanned: ScannedAsset[];
  timestamp: number;
}

interface BatchRoutesResult {
  routes:       BatchRoute[];
  lightBlocked: ScannedAsset[];   // blocked but isolated (no nearby camera to route with)
}

// ── Route clustering rule ──────────────────────────────────────────────────
//
// One clean rule: two blocked cameras belong in the same route if there is a
// CHAIN of blocked cameras between them where every consecutive hop is
// within CLUSTER_KM. This is single-linkage clustering (connected components
// over a "within range" graph) — a camera 10km from A and 10km from B joins
// the same route as A and B even if A and B themselves are 19km apart.
//
// This replaces the old two-threshold approach (a greedy nearest-5 pass at
// 18km, then a second "merge nearby routes" pass at 22km to patch what the
// greedy pass missed). That was order-dependent — whichever camera got
// visited first as an anchor shaped the whole grouping — and needed a fudge
// factor to catch cases it should have caught the first time. Connected
// components need only one threshold and give the same, order-independent
// answer regardless of scan order.
//
// Any component of size 1 (no other blocked camera within range, even
// transitively) is genuinely isolated — it goes to lightBlocked, not routes,
// however severely blocked it is.
const CLUSTER_KM = 18;

function computeBatchRoutes(scanned: ScannedAsset[]): BatchRoutesResult {
  const candidates = scanned.filter(a => a.riskFresh === "BLOCKED");
  if (candidates.length === 0) return { routes: [], lightBlocked: [] };

  // Union-find over candidate indices.
  const n      = candidates.length;
  const parent = Array.from({ length: n }, (_, i) => i);
  function find(x: number): number {
    while (parent[x] !== x) x = parent[x];
    return x;
  }
  function union(a: number, b: number) {
    const ra = find(a), rb = find(b);
    if (ra !== rb) parent[ra] = rb;
  }
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = distKm(candidates[i].lat, candidates[i].lng, candidates[j].lat, candidates[j].lng);
      if (d <= CLUSTER_KM) union(i, j);
    }
  }

  const groups = new Map<number, ScannedAsset[]>();
  for (let i = 0; i < n; i++) {
    const root = find(i);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root)!.push(candidates[i]);
  }

  const routes:       BatchRoute[]   = [];
  const lightBlocked: ScannedAsset[] = [];

  for (const group of groups.values()) {
    if (group.length === 1) {
      lightBlocked.push(group[0]);
      continue;
    }
    const withKm    = group.map(a => ({ ...a, km: 0 } as BatchRouteCamera));
    const ordered   = orderRoute(withKm);
    const maxP      = Math.max(...group.map(a => a.pFresh ?? 0));
    const immediate = maxP >= 0.82;
    routes.push({
      type: immediate ? "immediate" : "bundle",
      cameras: ordered,
      totalKm: routeLength(ordered),
      summary: immediate
        ? `Severe blockage confidence — recommended for immediate dispatch.`
        : `Blocked drains in close proximity — efficient to inspect on a single visit.`,
    });
  }

  return { routes, lightBlocked };
}

/**
 * Order a cluster's stops for a sensible driving route.
 * optimalOrder() is exhaustive (O(n!)) and explicitly only safe for n ≤ 7-8 —
 * single-linkage clustering can in principle chain a long strip of cameras
 * into one large group (e.g. a coastline), so anything bigger falls back to
 * a cheap nearest-neighbour greedy heuristic instead of hanging the tab.
 */
function orderRoute<T extends BatchRouteCamera>(cameras: T[]): T[] {
  return cameras.length <= 7 ? optimalOrder(cameras) : nearestNeighbourOrder(cameras);
}

function nearestNeighbourOrder<T extends { lat: number; lng: number }>(cameras: T[]): T[] {
  const remaining = [...cameras];
  const route     = [remaining.shift()!];
  while (remaining.length) {
    const last = route[route.length - 1];
    let bestIdx = 0, bestDist = Infinity;
    remaining.forEach((c, i) => {
      const d = distKm(last.lat, last.lng, c.lat, c.lng);
      if (d < bestDist) { bestDist = d; bestIdx = i; }
    });
    route.push(remaining.splice(bestIdx, 1)[0]);
  }
  return route;
}

// ── Module-level state persistence (survives tab navigation) ─────────────────
// React unmounts this component when the user switches routes, wiping useState.
// Storing the heavy scan result here means it's restored when they come back.
let _batchAnalysis:  BatchAnalysis | null = null;
let _panelFilter:    "blocked" | "flagged" | "clear" = "blocked";
let _panelRegion:    "all" | "Cornwall" | "Devon"   = "all";
let _regionFilter:   RegionFilter                   = "all";

// ── Component ─────────────────────────────────────────────────────────────────

function Dashboard() {
  const { assets, state, lastUpdated, refresh } = useAssets();

  const [selectedId, setSelectedId]       = useState<string>("");
  const [regionFilter, setRegionFilter]   = useState<RegionFilter>(_regionFilter);
  const [batchRegion, setBatchRegion]     = useState<string | null>(null);
  const [batchProgress, setBatchProgress] = useState<{ done: number; total: number } | null>(null);
  const [batchAnalysis, setBatchAnalysis] = useState<BatchAnalysis | null>(_batchAnalysis);
  const [batchError, setBatchError]       = useState<string | null>(null);
  const [detailAsset, setDetailAsset]     = useState<ScannedAsset | null>(null);
  const [routeInfo, setRouteInfo]         = useState<{
    routeN: number; accentColor: string; cameras: BatchRouteCamera[]; totalKm: number; summary: string;
  } | null>(null);
  const [panelFilter, setPanelFilter]     = useState<"blocked" | "flagged" | "clear">(_panelFilter);
  const [panelRegion, setPanelRegion]     = useState<"all" | "Cornwall" | "Devon">(_panelRegion);
  // Sub-filter within "Needs Review": operational issues (camera offline or
  // not actually pointed at a drain) vs genuine model uncertainty (a real
  // scene the classifier just isn't confident about — see the confidence-
  // margin check in inference.py). These are very different kinds of
  // "needs review" and were previously all lumped together.
  const [reviewFilter, setReviewFilter]   = useState<"all" | "offline" | "uncertain">("all");
  // Campaths currently mid-flight to /results/reclassify — disables that
  // row's buttons so a slow connection can't be double-clicked.
  const [reclassifying, setReclassifying] = useState<Set<string>>(new Set());
  const [detailRoute, setDetailRoute]     = useState<BatchRouteCamera[] | null>(null);
  const [detailRouteIdx, setDetailRouteIdx] = useState<number>(0);
  const [hoveredRouteIdx, setHoveredRouteIdx] = useState<number | null>(null);
  const [generatingReport, setGeneratingReport] = useState(false);
  const mapRef = useRef<BatchRouteMapHandle>(null);

  // Keep module-level cache in sync
  useEffect(() => { _batchAnalysis = batchAnalysis; }, [batchAnalysis]);
  useEffect(() => { _panelFilter   = panelFilter;   }, [panelFilter]);
  useEffect(() => { _panelRegion   = panelRegion;   }, [panelRegion]);
  useEffect(() => { _regionFilter  = regionFilter;  }, [regionFilter]);

  function openRouteModal(cameras: BatchRouteCamera[], idx: number) {
    setDetailRoute(cameras);
    setDetailRouteIdx(idx);
    setDetailAsset(cameras[idx]);
  }
  function closeDetailModal() {
    setDetailAsset(null);
    setDetailRoute(null);
    setDetailRouteIdx(0);
  }
  function navModal(dir: 1 | -1) {
    if (!detailRoute) return;
    const next = detailRouteIdx + dir;
    if (next >= 0 && next < detailRoute.length) {
      setDetailRouteIdx(next);
      setDetailAsset(detailRoute[next]);
    }
  }

  /**
   * Resolves an "Uncertain" (genuinely borderline, not offline/no-drainage)
   * FLAGGED camera into a definite BLOCKED or CLEAR verdict — persists the
   * correction server-side (POST /results/reclassify) and moves the camera
   * into the right category in the current view immediately, recomputing
   * routes/counts so it isn't just relabelled but actually re-filed.
   */
  async function handleReclassify(campath: string, newRisk: "BLOCKED" | "LOW") {
    if (!batchAnalysis) return;
    setReclassifying(prev => new Set(prev).add(campath));
    let reclassifyResponse;
    try {
      reclassifyResponse = await reclassifyResult(campath, newRisk);
    } catch (e) {
      console.error("Reclassify failed:", e);
      alert("Could not save the correction — check the backend connection.");
      setReclassifying(prev => { const n = new Set(prev); n.delete(campath); return n; });
      return;
    }

    const label = newRisk === "BLOCKED" ? "BLOCKED" : "CLEAR";
    // Marking BLOCKED generates a real GradCAM overlay AND a real LLM
    // inspection report server-side (the automatic pipeline never runs
    // either for a FLAGGED result) — use both here so the case reads
    // exactly like one the model called BLOCKED itself, instead of a
    // generic "manually reclassified" placeholder. Fall back to that
    // placeholder only if no overlay/explanation came back (e.g. a legacy
    // row with no stored original to run GradCAM on).
    const freshOverlay = reclassifyResponse.overlay_generated
      ? {
          overlay_b64:      reclassifyResponse.overlay_b64 ?? undefined,
          heatmap_b64:      reclassifyResponse.heatmap_b64 ?? undefined,
          heatmap_coverage: reclassifyResponse.heatmap_coverage ?? undefined,
        }
      : {};
    const updatedScanned = batchAnalysis.allScanned.map(a => {
      if (a.campath !== campath) return a;
      return {
        ...a,
        riskFresh: newRisk,
        pFresh: newRisk === "BLOCKED" ? 1 : 0,
        result: {
          ...a.result,
          risk: newRisk,
          prediction: label,
          p_blocked: newRisk === "BLOCKED" ? 1 : 0,
          explanation: reclassifyResponse.explanation ?? (
            `Manually reclassified as ${label} by reviewer.` +
            (a.result.explanation ? ` Model's own read: ${a.result.explanation}` : "")
          ),
          ...freshOverlay,
        },
      };
    });

    const { routes, lightBlocked } = computeBatchRoutes(updatedScanned);
    setBatchAnalysis({
      ...batchAnalysis,
      allScanned:    updatedScanned,
      routes,
      lightBlocked,
      blocked:       updatedScanned.filter(c => c.riskFresh === "BLOCKED").length,
      clear:         updatedScanned.filter(c => c.riskFresh === "LOW").length,
      flagged:       updatedScanned.filter(c => c.riskFresh === "FLAGGED").length,
      blockedAssets: updatedScanned.filter(c => c.riskFresh === "BLOCKED"),
    });
    setReclassifying(prev => { const n = new Set(prev); n.delete(campath); return n; });

    // detailAsset/detailRoute are snapshots taken when the modal opened, not
    // a live view into batchAnalysis — sync them too so a reviewer looking
    // at the image sees it flip to BLOCKED/CLEAR immediately instead of the
    // header still reading "NEEDS REVIEW" after they've just resolved it.
    const updated = updatedScanned.find(a => a.campath === campath);
    if (updated) {
      setDetailAsset(prev => (prev && prev.campath === campath ? updated : prev));
      setDetailRoute(prev => prev ? prev.map(c => (c.campath === campath ? { ...c, ...updated } : c)) : prev);
    }
  }

  const effectiveId = selectedId || assets[0]?.id || "";

  const filtered = useMemo(() =>
    assets.filter(a => regionFilter === "all" || a.region === regionFilter),
  [assets, regionFilter]);

  const counts = useMemo(() => {
    const c = { all: assets.length, high: 0, cornwall: 0, devon: 0 };
    for (const a of assets) {
      if (a.risk === "high" && a.status !== "offline") c.high++;
      if (a.region === "Cornwall") c.cornwall++;
      else c.devon++;
    }
    return c;
  }, [assets]);

  // ── Event handlers ────────────────────────────────────────────────────────

  const handleSelect = useCallback((id: string) => {
    setSelectedId(id);
    const cam = assets.find(a => a.id === id);
    if (cam?.lat && cam?.lng) mapRef.current?.flyTo(cam.lat, cam.lng);
  }, [assets]);

  const handleBatch = useCallback(async (region: "Cornwall" | "Devon" | "All") => {
    setBatchError(null);
    setBatchAnalysis(null);
    try { await fetchHealth(); } catch {
      setBatchError("Backend unreachable."); return;
    }
    const targets = assets.filter(a => region === "All" || a.region === region);
    setBatchRegion(region);
    setBatchProgress({ done: 0, total: targets.length });
    setPanelFilter("blocked");
    setPanelRegion("all");

    const collected: ScannedAsset[] = [];
    let errors = 0;
    for (let i = 0; i < targets.length; i++) {
      try {
        const result = await analyzeWebcam(targets[i].campath, targets[i].name);
        collected.push({ ...targets[i], pFresh: result.p_blocked ?? 0, riskFresh: result.risk, result });
      } catch (e) {
        errors++;
        if (i === 0) {
          setBatchProgress(null); setBatchRegion(null);
          setBatchError(`Batch failed: ${e instanceof Error ? e.message : e}`);
          return;
        }
      }
      setBatchProgress({ done: i + 1, total: targets.length });
    }

    setBatchProgress(null); setBatchRegion(null);
    if (errors > 0) setBatchError(`Done — ${errors} camera(s) unavailable.`);

    // Compute and store route analysis immediately (straight-line order)
    const { routes, lightBlocked } = computeBatchRoutes(collected);
    const summary = {
      region,
      lightBlocked,
      scanned:       collected.length,
      blocked:       collected.filter(c => c.riskFresh === "BLOCKED").length,
      clear:         collected.filter(c => c.riskFresh === "LOW").length,
      flagged:       collected.filter(c => c.riskFresh === "FLAGGED").length,
      blockedAssets: collected.filter(c => c.riskFresh === "BLOCKED"),
      allScanned:    collected,
      timestamp:     Date.now(),
    };
    setBatchAnalysis({ routes, ...summary });

    // Silently re-order each route using actual road travel times from OSRM,
    // then update state. Falls back to straight-line order on failure.
    const osrmCtrl = new AbortController();
    Promise.all(
      routes.map(r =>
        reorderByRoadTime(r.cameras, osrmCtrl.signal).then(reordered => {
          r.cameras = reordered;
        })
      )
    )
      .then(() => { setBatchAnalysis({ routes: [...routes], ...summary }); })
      .catch(() => { /* network error or abort — straight-line order kept */ });

    refresh();
  }, [assets, refresh]);

  // Categorise batch routes for display
  const immediateRoutes = batchAnalysis?.routes.filter(r => r.type === "immediate") ?? [];

  // Map data — hoisted so the map can always render (empty routes = base map)
  const mapRoutes: MapRoute[] = batchAnalysis
    ? [
        ...batchAnalysis.routes.map(r => ({ type: r.type, cameras: r.cameras, totalKm: r.totalKm })),
        // Isolated blockages (noted but not dispatched) — shown on the map as
        // single violet pins so they're not invisible next to the dispatched routes.
        ...(batchAnalysis.lightBlocked ?? []).map(cam => ({
          type: "isolated" as const, cameras: [cam], totalKm: 0,
        })),
      ]
    : [];
  const mapSummary: MapSummary = batchAnalysis
    ? {
        activeRoutes: batchAnalysis.routes.length,
        blocked:      batchAnalysis.blocked,
        critical:     immediateRoutes.length,
        scanTime:     `Scanned at ${new Date(batchAnalysis.timestamp).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })}`,
      }
    : { activeRoutes: 0, blocked: 0, critical: 0, scanTime: "" };

  return (
    <AppShell camCount={assets.length} state={state}>
      <div className="grid min-h-0 flex-1" style={{ gridTemplateColumns: "440px 1fr" }}>

        {/* ── Left: camera list ── */}
        <aside className="flex min-h-0 flex-col bg-surface/40" style={{ borderRight: "1px solid var(--color-border)" }}>

          {/* Region tabs */}
          <div className="flex shrink-0" style={{ borderBottom: "1px solid var(--color-border)" }}>
            {([
              ["all",      "All",      counts.all],
              ["Cornwall", "Cornwall", counts.cornwall],
              ["Devon",    "Devon",    counts.devon],
            ] as [RegionFilter, string, number][]).map(([k, l, n]) => {
              const active = regionFilter === k;
              return (
                <button key={k} onClick={() => setRegionFilter(k)}
                  className="flex flex-1 flex-col items-center justify-center transition-colors"
                  style={{
                    padding: "14px 0",
                    borderBottom: active ? "3px solid var(--color-foreground)" : "3px solid transparent",
                    background: active ? "color-mix(in oklch, var(--color-foreground) 5%, transparent)" : "transparent",
                  }}>
                  <span style={{
                    fontSize: 16,
                    fontWeight: active ? 700 : 500,
                    color: active ? "var(--color-foreground)" : "var(--color-muted-foreground)",
                  }}>{l}</span>
                  <span style={{
                    fontSize: 13,
                    fontVariantNumeric: "tabular-nums",
                    color: active ? "var(--color-foreground)" : "var(--color-muted-foreground)",
                    opacity: active ? 0.55 : 0.35,
                    marginTop: 2,
                  }}>{n}</span>
                </button>
              );
            })}
          </div>

          {/* Controls row: scan + report buttons */}
          <div className="flex shrink-0 items-center gap-2 px-3 py-2.5" style={{ borderBottom: "1px solid var(--color-border)" }}>
            <button
              onClick={() => handleBatch(regionFilter === "all" ? "All" : regionFilter as "Cornwall" | "Devon")}
              disabled={!!batchProgress}
              className="flex-1 rounded-[3px] px-3 py-1.5 text-[12px] font-medium transition-colors disabled:opacity-40"
              style={{ background: "var(--color-foreground)", color: "var(--color-background)" }}>
              {batchProgress && batchRegion === (regionFilter === "all" ? "All" : regionFilter)
                ? `${batchProgress.done}/${batchProgress.total}`
                : `Scan ${regionFilter === "all" ? "all regions" : regionFilter}`}
            </button>
            <button
              onClick={async () => {
                if (!batchAnalysis) return;
                setGeneratingReport(true);
                try { await generateReportPdf(batchAnalysis); }
                finally { setGeneratingReport(false); }
              }}
              disabled={!batchAnalysis || generatingReport}
              title={batchAnalysis ? "Download a PDF inspection report" : "Run a scan first"}
              className="shrink-0 rounded-[3px] border px-3 py-1.5 text-[12px] font-medium transition-colors hover:bg-muted disabled:opacity-40"
              style={{ borderColor: "var(--color-border)", color: "var(--color-foreground)" }}>
              {generatingReport ? "Generating…" : "Download PDF"}
            </button>
          </div>

          {/* Progress bar */}
          {batchProgress && (
            <div className="shrink-0 px-3 pt-2 pb-1">
              <div className="h-[3px] overflow-hidden rounded-full bg-border">
                <div className="h-full transition-all" style={{ width: `${batchProgress.done/batchProgress.total*100}%`, background: "var(--color-foreground)" }} />
              </div>
              <div className="mt-1 text-[10.5px] text-muted-foreground">
                Scanning {batchRegion} — {batchProgress.done} of {batchProgress.total}
              </div>
            </div>
          )}

          {/* List meta */}
          <div className="flex shrink-0 items-center justify-between px-3 py-2">
            <span className="text-[13px] font-semibold text-foreground">
              {filtered.length} camera{filtered.length !== 1 ? "s" : ""}
              {regionFilter !== "all" && <span className="font-normal text-muted-foreground"> · {regionFilter}</span>}
            </span>
            {lastUpdated && state === "live" && (
              <button onClick={refresh} className="text-[11px] text-muted-foreground hover:text-foreground">
                {lastUpdated.toLocaleTimeString("en-GB",{hour:"2-digit",minute:"2-digit"})} ↻
              </button>
            )}
          </div>

          {/* Camera list */}
          {(() => {
            // Build lookup of scan results for this list
            const scanMap = new Map(batchAnalysis?.allScanned.map(a => [a.campath, a]));
            const scanTime = batchAnalysis
              ? new Date(batchAnalysis.timestamp).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })
              : null;

            return (
              <ul className="min-h-0 flex-1 overflow-auto">
                {filtered.map((a, i) => {
                  const scanned = scanMap.get(a.campath);
                  const isBlocked = scanned && (scanned.riskFresh === "BLOCKED" || scanned.riskFresh === "HIGH" || scanned.riskFresh === "CRITICAL" || scanned.riskFresh === "MODERATE");
                  const isFlagged = scanned?.riskFresh === "FLAGGED";
                  const isClear   = scanned?.riskFresh === "LOW";
                  const isActive  = a.id === effectiveId;

                  return (
                    <li key={a.campath} style={{ borderBottom: i < filtered.length - 1 ? "1px solid var(--color-border)" : undefined }}>
                      <button onClick={() => handleSelect(a.id)}
                        className={`flex w-full items-start gap-0 border-l-2 px-3 py-3 text-left transition-colors ${
                          isActive ? "border-foreground bg-muted/50" : "border-transparent hover:bg-muted/20"
                        }`}>
                        <div className="min-w-0 flex-1">
                          {/* Name row */}
                          <div className="flex items-center justify-between gap-2">
                            <span className={`truncate text-[13px] font-medium ${isActive ? "text-foreground" : "text-foreground/90"}`}>
                              {a.name}
                            </span>
                            {isBlocked && (
                              <span className="shrink-0 text-[10px] font-semibold uppercase tracking-wide"
                                style={{ color: "var(--color-risk-high)" }}>
                                Blocked
                              </span>
                            )}
                            {isFlagged && (
                              <span className="shrink-0 text-[10px] font-semibold uppercase tracking-wide"
                                style={{ color: "var(--color-risk-medium)" }}>
                                Need Review
                              </span>
                            )}
                          </div>
                          {/* Subtitle row */}
                          <div className="mt-0.5 flex items-center gap-1.5 text-[11px] text-muted-foreground">
                            <span>{a.region}</span>
                            {scanned && scanTime && (
                              <>
                                <span className="opacity-30">·</span>
                                <span className={isClear ? "text-[var(--color-ok)]" : ""}>
                                  {scanTime} · {isClear ? "Clear" : isBlocked ? "Blocked" : isFlagged ? "Need Review" : scanned.riskFresh}
                                </span>
                              </>
                            )}
                          </div>
                        </div>
                      </button>
                    </li>
                  );
                })}
              </ul>
            );
          })()}
        </aside>

        {/* ── Main: batch analysis fills full height ── */}
        <main className="flex min-h-0 flex-col">

          {/* Batch error */}
          {batchError && (
            <pre className="shrink-0 px-4 py-2 text-[11px] leading-relaxed text-[var(--color-risk-high)]">
              {batchError}
            </pre>
          )}

          {/* Content area — always side-by-side */}
          <div className="min-h-0 flex-1 overflow-hidden flex flex-col">


            {/* Map left + panel right — always rendered */}
            <div className="flex min-h-0 flex-1 overflow-hidden">

              {/* Map — always visible. minWidth keeps it from being squeezed to
                  nothing when the window is narrower than full screen — the
                  right panel below shrinks proportionally instead (flex-basis
                  is a %, not a fixed px), so the map/panel split stays close
                  to the full-screen ratio at any window size. */}
              <div className="min-h-0 flex-1" style={{ borderRight: "1px solid var(--color-border)", minWidth: 320 }}>
                {/* ClientOnly keeps leaflet (which touches `window` at module
                    load time) from ever being imported during server-side
                    rendering — without this, SSR crashes with
                    "ReferenceError: window is not defined" and takes the
                    whole page down with it. */}
                <ClientOnly fallback={
                  <div className="flex h-full items-center justify-center text-[12px] text-muted-foreground">
                    Loading map…
                  </div>
                }>
                <Suspense fallback={
                  <div className="flex h-full items-center justify-center text-[12px] text-muted-foreground">
                    Loading map…
                  </div>
                }>
                  <BatchRouteMap
                    ref={mapRef}
                    routes={mapRoutes}
                    allCameras={assets.filter(a => a.lat && a.lng).map(a => ({
                      id: a.id, name: a.name, lat: a.lat, lng: a.lng, region: a.region,
                    }))}
                    flaggedCameras={(batchAnalysis?.allScanned ?? [])
                      .filter(a => a.riskFresh === "FLAGGED" && a.lat && a.lng)
                      .map(a => ({ id: a.id, name: a.name, lat: a.lat, lng: a.lng, region: a.region }))}
                    hoveredRouteIndex={hoveredRouteIdx}
                    summary={mapSummary}
                    height="100%"
                    onMarkerClick={(id) => {
                      if (!batchAnalysis) return;
                      for (const r of batchAnalysis.routes) {
                        const idx = r.cameras.findIndex(c => c.id === id);
                        if (idx !== -1) { openRouteModal(r.cameras, idx); return; }
                      }
                      // Isolated (violet) pins aren't part of any route — look them
                      // up in lightBlocked so clicking them opens the detail modal too.
                      const lb = batchAnalysis.lightBlocked ?? [];
                      const lbIdx = lb.findIndex(c => c.id === id);
                      if (lbIdx !== -1) {
                        const browseList = lb.map(a => ({ ...a, km: 0 } as BatchRouteCamera));
                        openRouteModal(browseList, lbIdx);
                        return;
                      }
                      // Needs Review (orange) pins aren't in a route or lightBlocked
                      // either — look them up in allScanned so clicking them opens
                      // the detail modal too.
                      const flagged = batchAnalysis.allScanned.filter(a => a.riskFresh === "FLAGGED");
                      const flagIdx = flagged.findIndex(c => c.id === id);
                      if (flagIdx !== -1) {
                        const browseList = flagged.map(a => ({ ...a, km: 0 } as BatchRouteCamera));
                        openRouteModal(browseList, flagIdx);
                      }
                    }}
                  />
                </Suspense>
                </ClientOnly>
              </div>

              {/* Right panel — results or empty state. Was a fixed w-[680px],
                  which meant a smaller (not full-screen) window left this
                  panel exactly as wide as always and forced the map's flex-1
                  space to absorb the entire loss, making the map tiny. A
                  flex-basis percentage (capped between minWidth/maxWidth)
                  keeps this panel — and therefore the map next to it — at
                  roughly the same visual proportion regardless of window size. */}
              <div className="overflow-hidden flex flex-col" style={{ flex: "0 1 32%", minWidth: 380, maxWidth: 560 }}>
              {batchAnalysis ? (
              <>
                {/* ── Region filter tabs — identical to left column ── */}
                <div className="flex shrink-0" style={{ borderBottom: "1px solid var(--color-border)" }}>
                  {(["all", "Cornwall", "Devon"] as const).map(r => {
                    const active = panelRegion === r;
                    const label  = r === "all" ? "All" : r;
                    const n = batchAnalysis.allScanned.filter(a => r === "all" || a.region === r).length;
                    return (
                      <button key={r} onClick={() => setPanelRegion(r)}
                        className="flex flex-1 flex-col items-center justify-center transition-colors"
                        style={{
                          padding: "14px 0",
                          borderBottom: active ? "3px solid var(--color-foreground)" : "3px solid transparent",
                          background: active ? "color-mix(in oklch, var(--color-foreground) 5%, transparent)" : "transparent",
                        }}>
                        <span style={{
                          fontSize: 16,
                          fontWeight: active ? 700 : 500,
                          color: active ? "var(--color-foreground)" : "var(--color-muted-foreground)",
                        }}>{label}</span>
                        <span style={{
                          fontSize: 13,
                          fontVariantNumeric: "tabular-nums",
                          color: active ? "var(--color-foreground)" : "var(--color-muted-foreground)",
                          opacity: active ? 0.55 : 0.35,
                          marginTop: 2,
                        }}>{n}</span>
                      </button>
                    );
                  })}
                </div>

                {/* ── Status tabs ── */}
                {(() => {
                  const visible  = batchAnalysis.allScanned.filter(a => panelRegion === "all" || a.region === panelRegion);
                  const isBlockedRisk = (r: string) => r === "BLOCKED" || r === "CRITICAL" || r === "MODERATE" || r === "HIGH";
                  const nBlocked = visible.filter(a => isBlockedRisk(a.riskFresh)).length;
                  const nFlagged = visible.filter(a => a.riskFresh === "FLAGGED").length;
                  const nClear   = visible.filter(a => a.riskFresh === "LOW").length;
                  const TABS = [
                    { key: "blocked" as const, label: "Blocked",     count: nBlocked },
                    { key: "flagged" as const, label: "Need Review", count: nFlagged },
                    { key: "clear"   as const, label: "Clear",        count: nClear   },
                  ];
                  return (
                    <div className="flex shrink-0" style={{ borderBottom: "1px solid var(--color-border)" }}>
                      {TABS.map(tab => {
                        const active = panelFilter === tab.key;
                        return (
                          <button key={tab.key} onClick={() => setPanelFilter(tab.key)}
                            className="flex flex-1 items-center justify-center gap-1.5 py-[12px] transition-colors"
                            style={{
                              fontSize: 13,
                              fontWeight: active ? 600 : 400,
                              color: active ? "var(--color-foreground)" : "var(--color-muted-foreground)",
                              borderBottom: active ? "3px solid var(--color-foreground)" : "3px solid transparent",
                              background: "transparent",
                            }}>
                            {tab.label}
                            <span style={{
                              fontSize: 11,
                              fontWeight: 500,
                              color: active ? "var(--color-foreground)" : "var(--color-muted-foreground)",
                              opacity: active ? 0.6 : 0.4,
                            }}>
                              {tab.count}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  );
                })()}

                <div className="flex-1 overflow-auto">
                <div className="px-5 py-4 space-y-5">

                  {/* ── Route list — only on Blocked tab ── */}
                  {panelFilter === "blocked" && batchAnalysis.routes.length > 0 && (() => {
                    type TierKey = "immediate" | "bundle";
                    const TIER: Record<TierKey, { accentColor: string }> = {
                      immediate: { accentColor: "var(--color-risk-high)" },
                      bundle:    { accentColor: "#b07828"                },
                    };
                    const regionRoutes = batchAnalysis.routes.filter(r =>
                      r.cameras.length >= 2 &&
                      (panelRegion === "all" || r.cameras.some(c => c.region === panelRegion))
                    );
                    const immediateR = regionRoutes.filter(r => r.type === "immediate");
                    const bundleR    = regionRoutes.filter(r => r.type === "bundle");
                    const allRoutes  = [
                      ...immediateR.map((r, i) => ({ route: r, tier: "immediate" as TierKey, idx: i })),
                      ...bundleR.map((r, i)    => ({ route: r, tier: "bundle"    as TierKey, idx: immediateR.length + i })),
                    ];
                    function shortName(name: string) {
                      const w = name.split(" ");
                      return w.length > 2 ? w.slice(1, 3).join(" ") : name;
                    }
                    return (
                      <section>
                        <div className="mb-2">
                          <span className="text-[12px] font-bold uppercase tracking-widest text-muted-foreground">Suggested Routes</span>
                        </div>
                        <ul>
                        {allRoutes.map(({ route, tier, idx: routeIdx }, i) => {
                          const t      = TIER[tier];
                          const routeN = routeIdx + 1;
                          return (
                            <li key={routeN}
                              onMouseEnter={() => setHoveredRouteIdx(routeIdx)}
                              onMouseLeave={() => setHoveredRouteIdx(null)}
                              style={{ borderBottom: i < allRoutes.length - 1 ? "1px solid var(--color-border)" : undefined }}>
                              <div className="flex w-full items-start border-l-4 px-3 py-3 hover:bg-muted/20 transition-colors"
                                style={{ borderLeftColor: t.accentColor }}>
                                <div className="min-w-0 flex-1">
                                  <div className="flex items-center justify-between gap-2 mb-1.5">
                                    <button
                                      onClick={() => setRouteInfo({
                                        routeN, accentColor: t.accentColor,
                                        cameras: route.cameras, totalKm: route.totalKm, summary: route.summary,
                                      })}
                                      className="text-[13px] font-semibold hover:underline transition-colors"
                                      style={{ color: t.accentColor }}>
                                      Route {routeN}
                                    </button>
                                  </div>
                                  <div className="flex flex-wrap items-center gap-x-1 gap-y-1">
                                    {route.cameras.map((cam, ci) => (
                                      <span key={cam.campath} className="flex items-center">
                                        {ci > 0 && (
                                          <span className="text-muted-foreground/40 text-[11px] px-0.5 select-none">→</span>
                                        )}
                                        <button
                                          onClick={() => openRouteModal(route.cameras, ci)}
                                          className="text-[12px] text-foreground/80 hover:text-foreground hover:underline transition-colors">
                                          {shortName(cam.name)}
                                        </button>
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              </div>
                            </li>
                          );
                        })}
                        </ul>
                      </section>
                    );
                  })()}

                  {/* ── Camera grid for active filter ── */}
                  {/* For "blocked", this includes every blocked camera — both
                      grouped (routed) and standalone/isolated — shown once,
                      each with its own thumbnail and Mark Clear action.
                      Grouped vs standalone is distinguished by border colour
                      only (red vs purple), not by a separate duplicate list. */}
                  {(() => {
                    const assets = batchAnalysis.allScanned.filter(a => {
                      const inRegion = panelRegion === "all" || a.region === panelRegion;
                      if (!inRegion) return false;
                      if (panelFilter === "blocked") {
                        return a.riskFresh === "BLOCKED" || a.riskFresh === "CRITICAL" || a.riskFresh === "MODERATE" || a.riskFresh === "HIGH";
                      }
                      if (panelFilter === "flagged") {
                        if (a.riskFresh !== "FLAGGED") return false;
                        if (reviewFilter === "offline") return isOfflineOrNoDrainage(a);
                        if (reviewFilter === "uncertain") return !isOfflineOrNoDrainage(a);
                        return true;
                      }
                      return a.riskFresh === "LOW";
                    });
                    const allFlagged = panelFilter === "flagged"
                      ? batchAnalysis.allScanned.filter(a => a.riskFresh === "FLAGGED" && (panelRegion === "all" || a.region === panelRegion))
                      : [];
                    const offlineCount    = allFlagged.filter(isOfflineOrNoDrainage).length;
                    const uncertainCount  = allFlagged.length - offlineCount;

                    const accentColor =
                      panelFilter === "blocked" ? "var(--color-risk-high)" :
                      panelFilter === "flagged" ? "var(--color-risk-medium)" :
                      "var(--color-ok)";
                    const badgeLabel =
                      panelFilter === "blocked" ? "BLOCKED" :
                      panelFilter === "flagged" ? "NEEDS REVIEW" : "CLEAR";
                    // Standalone (not part of any dispatched route) cameras —
                    // used below to colour each blocked item's left border
                    // purple instead of red, so grouped vs standalone is
                    // visible per-camera without a separate duplicate list.
                    const isolatedPaths = new Set((batchAnalysis.lightBlocked ?? []).map(c => c.campath));
                    return (
                      <section>
                        {panelFilter === "blocked" && (
                          <div className="mb-2">
                            <span className="text-[12px] font-bold uppercase tracking-widest text-muted-foreground">All Blocked Cameras</span>
                          </div>
                        )}
                        {panelFilter === "flagged" && (
                          <div className="flex shrink-0" style={{ borderBottom: "1px solid var(--color-border)" }}>
                            {([
                              ["all", "All", allFlagged.length],
                              ["offline", "Offline / No Drainage", offlineCount],
                              ["uncertain", "Uncertain", uncertainCount],
                            ] as [typeof reviewFilter, string, number][]).map(([key, label, count]) => {
                              const active = reviewFilter === key;
                              return (
                                <button key={key} onClick={() => setReviewFilter(key)}
                                  className="flex flex-1 items-center justify-center gap-1.5 py-[10px] transition-colors"
                                  style={{
                                    fontSize: 12,
                                    fontWeight: active ? 600 : 400,
                                    color: active ? "var(--color-foreground)" : "var(--color-muted-foreground)",
                                    borderBottom: active ? "3px solid var(--color-foreground)" : "3px solid transparent",
                                    background: "transparent",
                                  }}>
                                  {label}
                                  <span style={{
                                    fontSize: 10.5,
                                    fontWeight: 500,
                                    color: active ? "var(--color-foreground)" : "var(--color-muted-foreground)",
                                    opacity: active ? 0.6 : 0.4,
                                  }}>
                                    {count}
                                  </span>
                                </button>
                              );
                            })}
                          </div>
                        )}
                        {assets.length === 0 && (
                          <p className="text-[12px] text-muted-foreground/50 py-4 text-center">
                            No cameras in this category
                          </p>
                        )}
                        <ul>
                          {assets.map((asset, assetIdx) => {
                            const browseList = assets.map(a => ({ ...a, km: 0 } as BatchRouteCamera));
                            const isLast = assetIdx === assets.length - 1;
                            // REVIEW_ALL_CASES (see lib/api.ts): true -> every
                            // camera in every category can be corrected, for
                            // active review passes that also feed
                            // app/correction_memory.py fleet-wide. false ->
                            // back to the narrower FLAGGED-only "launch mode".
                            // Offline/no-drainage cameras get the same
                            // Mark Blocked/Clear controls as every other Need
                            // Review case now — a reviewer may know the feed
                            // is actually fine (mis-flagged) and want to
                            // correct it directly rather than leave it stuck.
                            const canReclassify = REVIEW_ALL_CASES || panelFilter === "flagged";
                            const busy = reclassifying.has(asset.campath);
                            // Already-confident verdicts only need the "other"
                            // button — a BLOCKED case has no reason to offer
                            // "Mark Blocked" again, and vice versa. Needs
                            // Review keeps both, since neither label applies yet.
                            const showMarkBlocked = asset.riskFresh !== "BLOCKED";
                            const showMarkClear   = asset.riskFresh !== "LOW";
                            // Blocked filter only: purple for standalone
                            // (not part of any dispatched route), red for
                            // grouped — same colour pairing as the map and
                            // the route/legend, so it reads consistently
                            // wherever a user looks. Other filters (flagged,
                            // clear) keep the single shared accentColor.
                            const itemAccent = panelFilter === "blocked"
                              ? (isolatedPaths.has(asset.campath) ? "#7c3aed" : "var(--color-risk-high)")
                              : accentColor;
                            return (
                              <li key={asset.campath}
                                style={{ borderBottom: !isLast ? "1px solid var(--color-border)" : undefined }}>
                                <div
                                  onClick={() => openRouteModal(browseList, assetIdx)}
                                  role="button"
                                  tabIndex={0}
                                  className="flex w-full cursor-pointer items-center gap-3 border-l-4 px-3 py-3.5 text-left transition-colors hover:bg-muted/20"
                                  style={{ borderLeftColor: itemAccent }}>
                                  <div className="min-w-0 flex-1">
                                    <div className="flex items-center justify-between gap-2">
                                      <span className="truncate text-[13px] font-medium text-foreground/90">
                                        {asset.name}
                                      </span>
                                      <span className="text-[10px] font-semibold uppercase tracking-wide shrink-0"
                                        style={{ color: itemAccent }}>
                                        {badgeLabel}
                                      </span>
                                    </div>
                                    <div className="mt-0.5 flex items-center gap-1.5 text-[11px] text-muted-foreground">
                                      <span>{asset.region}</span>
                                    </div>
                                    {canReclassify && (
                                      <div className="mt-2 flex items-center gap-1.5">
                                        {showMarkBlocked && (
                                          <button
                                            disabled={busy}
                                            onClick={(e) => { e.stopPropagation(); handleReclassify(asset.campath, "BLOCKED"); }}
                                            className="rounded-[3px] bg-transparent px-2 py-1 text-[10.5px] font-semibold uppercase tracking-wide transition-colors hover:bg-[color-mix(in_oklch,var(--color-risk-high)_10%,transparent)] disabled:opacity-40"
                                            style={{ border: "1.5px solid var(--color-risk-high)", color: "var(--color-risk-high)" }}>
                                            Mark Blocked
                                          </button>
                                        )}
                                        {showMarkClear && (
                                          <button
                                            disabled={busy}
                                            onClick={(e) => { e.stopPropagation(); handleReclassify(asset.campath, "LOW"); }}
                                            className="rounded-[3px] bg-transparent px-2 py-1 text-[10.5px] font-semibold uppercase tracking-wide transition-colors hover:bg-[color-mix(in_oklch,var(--color-ok)_10%,transparent)] disabled:opacity-40"
                                            style={{ border: "1.5px solid var(--color-ok)", color: "var(--color-ok)" }}>
                                            Mark Clear
                                          </button>
                                        )}
                                        {busy && <span className="text-[10.5px] text-muted-foreground">saving…</span>}
                                      </div>
                                    )}
                                  </div>
                                  {/* Thumbnail */}
                                  <div className="shrink-0 overflow-hidden rounded-[3px]"
                                    style={{ width: 160, height: 90, background: "#111" }}>
                                    {panelFilter === "blocked" && asset.result?.overlay_b64 ? (
                                      <img src={`data:image/jpeg;base64,${asset.result.overlay_b64}`}
                                        alt={asset.name} className="h-full w-full object-cover" />
                                    ) : (
                                      <img src={webcamProxyUrl(asset.campath)}
                                        alt={asset.name} className="h-full w-full object-cover" />
                                    )}
                                  </div>
                                </div>
                              </li>
                            );
                          })}
                        </ul>
                      </section>
                    );
                  })()}

                </div>
                </div>
              </>
              ) : (
                <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-10 px-10"
                  style={{ background: "var(--color-panel)" }}>
                  <svg width="140" height="140" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"
                    style={{ opacity: 0.25 }}>
                    <rect x="4" y="16" width="56" height="36" rx="4" stroke="currentColor" strokeWidth="2.5" />
                    <circle cx="32" cy="34" r="9" stroke="currentColor" strokeWidth="2.5" />
                    <circle cx="32" cy="34" r="4" fill="currentColor" />
                    <rect x="24" y="10" width="16" height="6" rx="2" stroke="currentColor" strokeWidth="2.5" />
                    <path d="M10 44 Q16 38 22 44 Q28 50 34 44 Q40 38 46 44 Q52 50 54 47" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity="0.6"/>
                  </svg>
                  <div className="text-center space-y-4 max-w-[420px]">
                    <p className="text-[28px] font-bold text-foreground/70">No scan results yet</p>
                    <p className="text-[16px] text-muted-foreground/60 leading-relaxed">
                      Select a region below to run an AI-powered batch scan. Blocked drains are grouped into optimised field routes.
                    </p>
                  </div>
                  <div className="flex flex-col items-stretch gap-4 w-full max-w-[420px]">
                    {(["Cornwall", "Devon", "All"] as const).map(region => (
                      <button key={region}
                        onClick={() => handleBatch(region)}
                        disabled={!!batchProgress}
                        className="rounded-[8px] px-8 py-5 text-[17px] font-semibold transition-colors disabled:opacity-40 text-left"
                        style={{ border: "1px solid var(--color-border)", background: "var(--color-surface)", color: "var(--color-foreground)" }}>
                        {region === "All"
                          ? `Scan all  ·  ${counts.all} cameras`
                          : `${region}  ·  ${region === "Cornwall" ? counts.cornwall : counts.devon} cameras`}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              </div>{/* right panel */}
            </div>{/* flex side-by-side */}
          </div>{/* content area */}
        </main>

        {/* ── Detail modal — blocked camera full analysis ── */}
        {detailAsset && (
          <div
            className="fixed inset-0 z-[2000] flex items-center justify-center p-6"
            style={{ background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)" }}
            onClick={closeDetailModal}
          >
            <div
              className="relative flex flex-col rounded-[6px] overflow-hidden"
              style={{
                width: "99vw",
                height: "99vh",
                background: "var(--color-surface)",
                border: "1px solid var(--color-border-strong)",
                boxShadow: "0 24px 64px rgba(0,0,0,0.4)",
              }}
              onClick={e => e.stopPropagation()}
            >
              {/* Modal header */}
              <div className="flex shrink-0 items-center justify-between gap-4 px-5 py-3"
                style={{ borderBottom: "1px solid var(--color-border)" }}>
                <div className="flex items-center gap-2.5 min-w-0">
                  <span className="text-[15px] font-bold truncate">{detailAsset.name}</span>
                  {(() => {
                    const risk   = detailAsset.riskFresh ?? detailAsset.risk;
                    const isHigh = risk === "high" || risk === "HIGH" || risk === "BLOCKED" || risk === "CRITICAL" || risk === "MODERATE";
                    const isMed  = risk === "medium" || risk === "MEDIUM" || risk === "FLAGGED";
                    const color  = isHigh ? "var(--color-risk-high)" : isMed ? "var(--color-risk-medium)" : "var(--color-ok)";
                    const label  = isHigh ? "BLOCKED" : isMed ? "NEEDS REVIEW" : "CLEAR";
                    return (
                      <span className="shrink-0 text-[12px] font-bold uppercase tracking-widest" style={{ color }}>
                        {label}
                      </span>
                    );
                  })()}
                  <span className="shrink-0 text-[12px] text-muted-foreground">{detailAsset.region}</span>
                  {detailRoute && detailRoute.length > 1 && (
                    <span className="shrink-0 text-[11px] text-muted-foreground/60 tabular-nums">
                      {detailRouteIdx + 1} / {detailRoute.length}
                    </span>
                  )}
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  {/* REVIEW_ALL_CASES (lib/api.ts) — see the matching comment
                      in the list rows above for the toggle's purpose. Offline/
                      no-drainage cases are no longer excluded here either. */}
                  {(REVIEW_ALL_CASES || detailAsset.riskFresh === "FLAGGED") && (
                    <>
                      {detailAsset.riskFresh !== "BLOCKED" && (
                        <button
                          disabled={reclassifying.has(detailAsset.campath)}
                          onClick={() => handleReclassify(detailAsset.campath, "BLOCKED")}
                          className="rounded-[3px] bg-transparent px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide transition-colors hover:bg-[color-mix(in_oklch,var(--color-risk-high)_10%,transparent)] disabled:opacity-40"
                          style={{ border: "1.5px solid var(--color-risk-high)", color: "var(--color-risk-high)" }}>
                          Mark Blocked
                        </button>
                      )}
                      {detailAsset.riskFresh !== "LOW" && (
                        <button
                          disabled={reclassifying.has(detailAsset.campath)}
                          onClick={() => handleReclassify(detailAsset.campath, "LOW")}
                          className="rounded-[3px] bg-transparent px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide transition-colors hover:bg-[color-mix(in_oklch,var(--color-ok)_10%,transparent)] disabled:opacity-40"
                          style={{ border: "1.5px solid var(--color-ok)", color: "var(--color-ok)" }}>
                          Mark Clear
                        </button>
                      )}
                      {reclassifying.has(detailAsset.campath) && (
                        <span className="text-[11px] text-muted-foreground">saving…</span>
                      )}
                    </>
                  )}
                  <button
                    onClick={closeDetailModal}
                    className="rounded-[3px] px-2 py-1 text-[13px] text-muted-foreground hover:text-foreground hover:bg-muted transition-colors">
                    ✕ Close
                  </button>
                </div>
              </div>

              {/* Images + AI Assessment now scroll together as one column,
                  instead of splitting the modal into two fixed-height halves
                  that fought each other for space — that's what was making
                  the images look cramped/half-hidden. Images get a generous
                  fixed height of their own; if the assessment text below
                  makes the total taller than the modal, this wrapper
                  scrolls rather than shrinking the images to fit. */}
              <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">
              <div className="relative shrink-0 overflow-hidden" style={{ height: "68vh" }}>
                {(detailAsset.riskFresh === "LOW" || detailAsset.riskFresh === "FLAGGED") ? (
                  /* Clear / Needs Review: single full-width original — no Grad-CAM split */
                  <div className="relative h-full bg-black">
                    <img src={webcamProxyUrl(detailAsset.campath)} alt={detailAsset.name}
                      className="h-full w-full object-contain" />
                    <span className="absolute bottom-3 left-3 text-[11px] font-medium text-white/50 pointer-events-none tracking-wide uppercase">Original</span>
                  </div>
                ) : (
                  /* Blocked: side-by-side original + Grad-CAM */
                  <div className="h-full grid grid-cols-2">
                    <div className="relative bg-black" style={{ borderRight: "1px solid var(--color-border)" }}>
                      {/* Use the exact frame GradCAM ran on (stored alongside the
                          overlay), not a fresh live re-fetch — the live camera can
                          refresh between the two requests, showing two different
                          moments side by side otherwise. Live fetch is only a
                          fallback for rows saved before this was tracked. */}
                      <img src={detailAsset.result.original_b64
                          ? `data:image/jpeg;base64,${detailAsset.result.original_b64}`
                          : webcamProxyUrl(detailAsset.campath)} alt={detailAsset.name}
                        className="h-full w-full object-contain" />
                      <span className="absolute bottom-3 left-3 text-[11px] font-medium text-white/50 pointer-events-none tracking-wide uppercase">Original</span>
                    </div>
                    <div className="relative bg-black">
                      {detailAsset.result.overlay_b64 ? (
                        <>
                          <img src={`data:image/jpeg;base64,${detailAsset.result.overlay_b64}`} alt="Grad-CAM"
                            className="h-full w-full object-contain" />
                          <span className="absolute bottom-3 left-3 text-[11px] font-medium text-white/50 pointer-events-none tracking-wide uppercase">Grad-CAM</span>
                        </>
                      ) : (
                        <div className="flex h-full items-center justify-center text-[12px] text-white/25">No overlay available</div>
                      )}
                    </div>
                  </div>
                )}

                {/* ← → arrows overlaid on the image — always visible when list has > 1 */}
                {detailRoute && detailRoute.length > 1 && (
                  <>
                    <button
                      onClick={() => navModal(-1)}
                      disabled={detailRouteIdx === 0}
                      className="absolute left-4 top-1/2 -translate-y-1/2 flex items-center justify-center rounded-full transition-opacity disabled:opacity-0"
                      style={{ width: 52, height: 52, background: "rgba(0,0,0,0.55)", color: "#fff", fontSize: 22, backdropFilter: "blur(4px)" }}>
                      ‹
                    </button>
                    <button
                      onClick={() => navModal(1)}
                      disabled={detailRouteIdx === detailRoute.length - 1}
                      className="absolute right-4 top-1/2 -translate-y-1/2 flex items-center justify-center rounded-full transition-opacity disabled:opacity-0"
                      style={{ width: 52, height: 52, background: "rgba(0,0,0,0.55)", color: "#fff", fontSize: 22, backdropFilter: "blur(4px)" }}>
                      ›
                    </button>
                  </>
                )}
              </div>

              {/* Metadata + explanation — sits in natural flow below the
                  images now (see the shared scroll wrapper above), so it
                  gets whatever height its own content needs instead of being
                  forced into a fixed share of the modal. */}
              <div className="flex shrink-0 flex-col px-8 py-6 gap-3"
                style={{ borderTop: "1px solid var(--color-border)" }}>
                <div className="flex shrink-0 items-center gap-2.5 flex-wrap">
                  <span className="text-[13px] font-semibold">{detailAsset.result.prediction}</span>
                  {(detailAsset.result.quality_flags?.length ?? 0) > 0 && (
                    <span className="text-[11px] text-[var(--color-risk-medium)]">
                      {detailAsset.result.quality_flags.join(" · ")}
                    </span>
                  )}
                </div>
                {detailAsset.result.explanation ? (
                  <div className="flex flex-col items-center justify-center rounded-[10px] px-7 py-6 text-center"
                    style={{
                      background: "var(--color-surface)",
                      border: "1.5px solid var(--color-border)",
                      borderLeft: "5px solid var(--color-foreground)",
                      boxShadow: "0 1px 4px rgba(0,0,0,0.07)",
                      maxWidth: 880,
                      width: "100%",
                      margin: "0 auto",
                    }}>
                    <div className="mb-2.5 text-[12px] font-bold uppercase tracking-widest text-muted-foreground">
                      AI Assessment
                    </div>
                    <p className="text-[21px] font-medium leading-[1.75] text-foreground whitespace-pre-line text-center">
                      {detailAsset.result.explanation}
                    </p>
                  </div>
                ) : (
                  <p className="text-[12px] text-muted-foreground italic">No detailed analysis available.</p>
                )}
              </div>
              </div>
            </div>
          </div>
        )}

        {/* ── Route info popover — opened by clicking "Route N" ── */}
        {routeInfo && (
          <div
            className="fixed inset-0 z-[2100] flex items-center justify-center p-6"
            style={{ background: "rgba(0,0,0,0.45)", backdropFilter: "blur(2px)" }}
            onClick={() => setRouteInfo(null)}
          >
            <div
              className="relative flex flex-col rounded-[8px] overflow-hidden"
              style={{
                width: 720,
                maxHeight: "88vh",
                background: "var(--color-surface)",
                border: "1px solid var(--color-border-strong)",
                boxShadow: "0 24px 64px rgba(0,0,0,0.4)",
              }}
              onClick={e => e.stopPropagation()}
            >
              {/* Header */}
              <div className="flex shrink-0 items-start justify-between gap-4 px-6 py-5"
                style={{ borderBottom: "1px solid var(--color-border)" }}>
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-widest text-muted-foreground mb-1">
                    Suggested Route
                  </div>
                  <div className="text-[24px] font-semibold tracking-tight leading-none" style={{ color: routeInfo.accentColor }}>
                    Route {routeInfo.routeN}
                  </div>
                </div>
                <button onClick={() => setRouteInfo(null)}
                  className="shrink-0 rounded-[4px] px-2 py-1 text-[12px] font-medium text-muted-foreground hover:text-foreground hover:bg-muted/40 transition-colors">
                  Close
                </button>
              </div>

              <div className="flex flex-col overflow-y-auto px-6 py-6 gap-6">
                {/* Overview — one soft box, same treatment as the AI Assessment card */}
                <div className="flex flex-col items-center justify-center rounded-[10px] px-7 py-6 text-center"
                  style={{
                    background: "var(--color-surface)",
                    border: "1.5px solid var(--color-border)",
                    borderLeft: `5px solid ${routeInfo.accentColor}`,
                    boxShadow: "0 1px 4px rgba(0,0,0,0.07)",
                  }}>
                  <div className="mb-2.5 text-[12px] font-bold uppercase tracking-widest text-muted-foreground">
                    Route Overview
                  </div>
                  <p className="text-[16px] font-medium text-foreground">
                    {routeInfo.cameras.length} stop{routeInfo.cameras.length !== 1 ? "s" : ""}
                    {routeInfo.totalKm > 0 && (
                      <> · {routeInfo.totalKm.toFixed(1)} km · ~{Math.round(routeInfo.totalKm / 40 * 60)} min</>
                    )}
                  </p>
                </div>

                {/* Stops — plain list, no boxes or dividers */}
                <div className="flex flex-col">
                  <div className="mb-2 text-[11px] font-semibold uppercase tracking-widest text-muted-foreground">
                    Stops in order
                  </div>
                  <div className="flex flex-col gap-1">
                    {routeInfo.cameras.map((cam, ci) => (
                      <button key={cam.campath}
                        onClick={() => { setRouteInfo(null); openRouteModal(routeInfo.cameras, ci); }}
                        className="flex items-center gap-3 py-1.5 text-left transition-opacity hover:opacity-65">
                        <span className="shrink-0 text-[13px] font-semibold" style={{ color: routeInfo.accentColor }}>
                          {ci + 1}
                        </span>
                        <span className="text-[14px] font-medium text-foreground/90">{cam.name}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

      </div>
    </AppShell>
  );
}

/* ─────────────────────────── Report / PDF generation ─────────────────────────── */

// One distinct colour per ROUTE (not per tier) — so "Route 1" always reads
// as the same colour in the text list, the legend, and on the map. With only
// two tier colours, two "immediate" routes used to be visually identical on
// the map with no way to tell which was which.
//
// Deliberately no purple/violet in this list — that hue is reserved for
// ISOLATED_PDF_COLOUR below, and a 3rd/4th route landing on a near-identical
// purple made it indistinguishable from an isolated case on the map.
const ROUTE_PDF_PALETTE: [number, number, number][] = [
  [192, 70, 58],    // red
  [37, 99, 235],    // blue
  [16, 150, 100],   // green
  [217, 119, 6],    // amber
  [219, 39, 119],   // pink
  [8, 145, 178],    // teal
  [101, 67, 33],    // brown
];
function routeColour(i: number): [number, number, number] {
  return ROUTE_PDF_PALETTE[i % ROUTE_PDF_PALETTE.length];
}
const ISOLATED_PDF_COLOUR: [number, number, number] = [124, 58, 237]; // violet, matches map pins

/** Load a base64/data-URL image just to read its natural pixel dimensions,
 *  so it can be embedded in the PDF at the right aspect ratio. */
function loadImageSize(dataUrl: string): Promise<{ w: number; h: number }> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload  = () => resolve({ w: img.naturalWidth || 4, h: img.naturalHeight || 3 });
    img.onerror = () => resolve({ w: 4, h: 3 });
    img.src = dataUrl;
  });
}

/**
 * Draws a schematic (not tile-based) map of every routed and isolated blocked
 * camera directly onto the PDF page using jsPDF's own vector primitives.
 *
 * This deliberately does NOT screenshot the live Leaflet map: OSM's tile
 * servers don't send CORS headers, so a canvas screenshot of real map tiles
 * comes out blank in the exported image. Plotting camera coordinates
 * directly (simple linear lat/lng projection within the scan's bounding
 * box) guarantees the routes and isolated pins always appear, on every
 * export, regardless of any tile server.
 */
function drawSchematicMap(
  doc: jsPDF, batchAnalysis: BatchAnalysis, x: number, y: number, w: number, h: number,
) {
  const isolatedCams = batchAnalysis.lightBlocked ?? [];
  const all = [...batchAnalysis.routes.flatMap(r => r.cameras), ...isolatedCams];

  doc.setFillColor(245, 246, 248);
  doc.setDrawColor(210, 210, 210);
  doc.rect(x, y, w, h, "FD");

  if (all.length === 0) {
    doc.setFont("helvetica", "italic");
    doc.setFontSize(9);
    doc.setTextColor(140, 140, 140);
    doc.text("No blocked locations to plot.", x + w / 2, y + h / 2, { align: "center" });
    return;
  }

  const lats = all.map(c => c.lat), lngs = all.map(c => c.lng);
  const minLat = Math.min(...lats), maxLat = Math.max(...lats);
  const minLng = Math.min(...lngs), maxLng = Math.max(...lngs);
  const latSpan = Math.max(maxLat - minLat, 0.01);
  const lngSpan = Math.max(maxLng - minLng, 0.01);
  const pad        = 30;
  const topReserve = 18; // room for the compass
  const botReserve = 34; // room for the legend + scale bar

  function project(lat: number, lng: number): [number, number] {
    const plotH = h - topReserve - botReserve;
    const px = x + pad + ((lng - minLng) / lngSpan) * (w - pad * 2);
    const py = y + topReserve + (1 - (lat - minLat) / latSpan) * plotH;
    return [px, py];
  }

  // Alternates each label above/below its pin so a chain of nearby stops
  // doesn't collide into an unreadable pile of overlapping text.
  let labelToggle = 0;
  function drawPin(lat: number, lng: number, label: string, name: string, colour: [number, number, number]) {
    const [px, py] = project(lat, lng);
    doc.setFillColor(colour[0], colour[1], colour[2]);
    doc.circle(px, py, 6, "F");
    doc.setFont("helvetica", "bold");
    doc.setFontSize(7);
    doc.setTextColor(255, 255, 255);
    doc.text(label, px, py + 2.3, { align: "center" });

    const above = labelToggle % 2 === 0;
    labelToggle++;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(7);
    const maxX   = x + w - pad - doc.getTextWidth(name);
    const labelX = Math.min(px + 9, Math.max(maxX, x + pad));
    const labelY = above ? Math.max(py - 9, y + topReserve - 4) : Math.min(py + 15, y + h - botReserve + 6);

    // White pill behind the name — plain text straight on the grey
    // background reads as an afterthought; this makes it read as a label.
    const textW = doc.getTextWidth(name);
    doc.setFillColor(255, 255, 255);
    doc.setDrawColor(220, 220, 220);
    doc.roundedRect(labelX - 3, labelY - 6.5, textW + 6, 9, 2, 2, "FD");
    doc.setTextColor(50, 50, 50);
    doc.text(name, labelX, labelY);
  }

  // Routes: connecting lines first (so pins draw on top), then numbered stops
  batchAnalysis.routes.forEach((route, routeI) => {
    const colour = routeColour(routeI);
    doc.setDrawColor(colour[0], colour[1], colour[2]);
    doc.setLineWidth(1.4);
    for (let i = 1; i < route.cameras.length; i++) {
      const [x1, y1] = project(route.cameras[i - 1].lat, route.cameras[i - 1].lng);
      const [x2, y2] = project(route.cameras[i].lat, route.cameras[i].lng);
      doc.line(x1, y1, x2, y2);
    }
    route.cameras.forEach((cam, i) => drawPin(cam.lat, cam.lng, String(i + 1), cam.name, colour));
  });

  // Isolated cases — always labelled "1", violet, no connecting line
  isolatedCams.forEach(cam => drawPin(cam.lat, cam.lng, "1", cam.name, ISOLATED_PDF_COLOUR));

  // Compass — top-right corner
  const cx = x + w - pad - 4, cy = y + 12;
  doc.setDrawColor(100, 100, 100);
  doc.setLineWidth(1);
  doc.line(cx, cy + 7, cx, cy - 7);
  doc.line(cx, cy - 7, cx - 3, cy - 2);
  doc.line(cx, cy - 7, cx + 3, cy - 2);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(7.5);
  doc.setTextColor(100, 100, 100);
  doc.text("N", cx, cy + 15, { align: "center" });

  // Scale bar — bottom-left, derived from the real distance the plot spans
  const midLat       = (minLat + maxLat) / 2;
  const totalKmWidth  = Math.max(distKm(midLat, minLng, midLat, maxLng), 0.1);
  const plotWidthPt   = w - pad * 2;
  const kmPerPt       = totalKmWidth / plotWidthPt;
  const niceKmSteps   = [1, 2, 5,10,20,25,50,100, 200, 500];
  let scaleKm = niceKmSteps[0];
  for (const step of niceKmSteps) {
    if (step / kmPerPt <= plotWidthPt * 0.4) scaleKm = step;
  }
  const scaleBarPt = scaleKm / kmPerPt;
  const scaleX = x + pad, scaleY = y + h - botReserve + 20;
  doc.setDrawColor(90, 90, 90);
  doc.setLineWidth(1.2);
  doc.line(scaleX, scaleY, scaleX + scaleBarPt, scaleY);
  doc.line(scaleX, scaleY - 3, scaleX, scaleY + 3);
  doc.line(scaleX + scaleBarPt, scaleY - 3, scaleX + scaleBarPt, scaleY + 3);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(7);
  doc.setTextColor(90, 90, 90);
  doc.text(`${scaleKm} km`, scaleX + scaleBarPt / 2, scaleY - 5, { align: "center" });

  // Legend — bottom-right — one entry per actual route, so "Route 1",
  // "Route 2" etc. can be matched by colour back to the text list above.
  const legendY = y + h - botReserve + 20;
  let lx = x + w / 2;
  doc.setFont("helvetica", "normal");
  doc.setFontSize(7.5);
  const legendItems: [string, [number, number, number]][] = batchAnalysis.routes.map(
    (_, i) => [`Route ${i + 1}`, routeColour(i)],
  );
  if (isolatedCams.length > 0) legendItems.push(["Standalone blockage", ISOLATED_PDF_COLOUR]);
  legendItems.forEach(([label, colour]) => {
    doc.setFillColor(colour[0], colour[1], colour[2]);
    doc.circle(lx, legendY, 3, "F");
    doc.setTextColor(70, 70, 70);
    doc.text(label, lx + 7, legendY + 2.5);
    lx += doc.getTextWidth(label) + 24;
  });
}

/**
 * Colour-key legend drawn below the real OSM map image (which has no room
 * reserved inside it for one, unlike the schematic fallback). Wraps to a
 * second line if there are enough routes that one row would overflow.
 * Returns the vertical space it used, so the caller can advance y.
 */
function drawRouteLegendBelow(doc: jsPDF, batchAnalysis: BatchAnalysis, x: number, y: number, w: number): number {
  const items: [string, [number, number, number]][] = batchAnalysis.routes.map(
    (_, i) => [`Route ${i + 1}`, routeColour(i)],
  );
  if ((batchAnalysis.lightBlocked ?? []).length > 0) items.push(["Standalone blockage", ISOLATED_PDF_COLOUR]);
  if (items.length === 0) return 0;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  const rowH = 15;
  let lx = x, ly = y + 9;
  items.forEach(([label, colour]) => {
    const labelW = doc.getTextWidth(label);
    if (lx > x && lx + 7 + labelW > x + w) { lx = x; ly += rowH; }
    doc.setFillColor(colour[0], colour[1], colour[2]);
    doc.circle(lx, ly, 3.2, "F");
    doc.setTextColor(70, 70, 70);
    doc.text(label, lx + 8, ly + 2.5);
    lx += 8 + labelW + 20;
  });
  return (ly - y) + rowH;
}

/**
 * Builds and downloads the full inspection report as a real PDF file —
 * built programmatically with jsPDF (images via addImage, text via text()).
 * Each major section starts on its own page. The map is a real OSM basemap
 * fetched from the backend's /staticmap endpoint (see api.ts), falling back
 * to the hand-drawn schematic (drawSchematicMap) only if that request fails.
 */
function rgbToHex([r, g, b]: [number, number, number]): string {
  return "#" + [r, g, b].map(v => v.toString(16).padStart(2, "0")).join("");
}

/**
 * Split an LLM explanation of the form
 * "OBSERVATION: ...\n\nRISK ASSESSMENT: ...\n\nRECOMMENDED ACTION: ..."
 * into labelled sections so each can get its own bold heading in the PDF,
 * instead of dumping it all in as one unlabelled paragraph.
 */
function parseExplanation(text?: string): { label: string; body: string }[] {
  if (!text) return [];
  const sectionRe = /(OBSERVATION|RISK ASSESSMENT|RECOMMENDED ACTION):\s*/g;
  const matches = [...text.matchAll(sectionRe)];
  if (matches.length === 0) return [{ label: "", body: text.trim() }];
  const parts: { label: string; body: string }[] = [];
  matches.forEach((m, i) => {
    const start = (m.index ?? 0) + m[0].length;
    const end = i + 1 < matches.length ? matches[i + 1].index ?? text.length : text.length;
    const body = text.slice(start, end).trim();
    if (body) parts.push({ label: m[1], body });
  });
  return parts;
}

async function generateReportPdf(batchAnalysis: BatchAnalysis) {
  const doc     = new jsPDF({ unit: "pt", format: "a4" });
  const pageW   = doc.internal.pageSize.getWidth();
  const pageH   = doc.internal.pageSize.getHeight();
  const margin  = 40;
  let y = margin;

  function ensureSpace(neededHeight: number) {
    if (y + neededHeight > pageH - margin) {
      doc.addPage();
      y = margin;
    }
  }

  // Bold, uppercase, tracked-looking title with a short coloured accent
  // underline (instead of a full-width grey hairline) — reads as a proper
  // section header rather than a plain line of bold text.
  function sectionTitle(title: string, accent: [number, number, number] = [30, 30, 30]) {
    ensureSpace(38);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(14.5);
    doc.setTextColor(25, 25, 25);
    doc.text(title.toUpperCase(), margin, y);
    y += 11;
    doc.setFillColor(accent[0], accent[1], accent[2]);
    doc.roundedRect(margin, y, 42, 3.5, 1.5, 1.5, "F");
    y += 24;
  }

  // Forces each major section onto a fresh page — but only if the current
  // page already has something on it, so we don't leave a stray blank page
  // when a section happens to start right at the top of one anyway.
  function startNewSection(title: string, accent?: [number, number, number]) {
    if (y > margin) {
      doc.addPage();
      y = margin;
    }
    sectionTitle(title, accent);
  }

  function emptyNote(msg: string) {
    doc.setFont("helvetica", "italic");
    doc.setFontSize(10);
    doc.setTextColor(130, 130, 130);
    doc.text(msg, margin, y);
    y += 22;
  }

  /** Compact "just mention" list of camera names — no images, one per line. */
  function mentionList(cams: ScannedAsset[]) {
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9.5);
    doc.setTextColor(50, 50, 50);
    cams.forEach(c => {
      ensureSpace(14);
      doc.text(`•  ${c.name} — ${c.region}`, margin, y);
      y += 14;
    });
    y += 8;
  }

  // ── Header ──
  doc.setFont("helvetica", "bold");
  doc.setFontSize(27);
  doc.setTextColor(20, 20, 20);
  doc.text("Blockage Inspection Report", margin, y);
  y += 18;

  // Accent bar under the title
  doc.setFillColor(37, 99, 235);
  doc.roundedRect(margin, y, 64, 4, 1.5, 1.5, "F");
  y += 24;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(10.5);
  doc.setTextColor(110, 110, 110);
  const scanTime = new Date(batchAnalysis.timestamp).toLocaleString("en-GB", {
    day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit",
  });
  const regionLabel = batchAnalysis.region === "All" ? "All regions" : batchAnalysis.region;
  doc.text(`Generated ${scanTime} · ${regionLabel}`, margin, y);
  y += 30;

  // A short line of context, so page 1 reads as an actual cover/summary
  // page rather than a title floating over empty space.
  doc.setFont("helvetica", "normal");
  doc.setFontSize(10.5);
  doc.setTextColor(70, 70, 70);
  const introLines = doc.splitTextToSize(
    `This report summarises the automated inspection of ${batchAnalysis.scanned} drainage ` +
    `camera location${batchAnalysis.scanned !== 1 ? "s" : ""} across ${regionLabel === "All regions" ? "Devon and Cornwall" : regionLabel}. ` +
    `${batchAnalysis.blocked} location${batchAnalysis.blocked !== 1 ? "s" : ""} require immediate attention, ` +
    `${batchAnalysis.flagged} need${batchAnalysis.flagged === 1 ? "s" : ""} manual review, and ${batchAnalysis.clear} ` +
    `${batchAnalysis.clear === 1 ? "is" : "are"} confirmed clear. Full detail for each blocked location — including the ` +
    `model's visual analysis and Grad-CAM heatmap — follows, along with suggested maintenance routes and their positions on a map.`,
    pageW - margin * 2,
  );
  doc.text(introLines, margin, y);
  y += introLines.length * 14 + 26;

  // Stat cards — Blocked / Needs Review / Clear / Scanned. Colour accent
  // bar + bold coloured number + caption, same as before, just without the
  // light grey background box behind each one.
  const statCards: { label: string; value: number; colour: [number, number, number] }[] = [
    { label: "BLOCKED",       value: batchAnalysis.blocked,  colour: [220, 38, 38] },
    { label: "NEEDS REVIEW",  value: batchAnalysis.flagged,  colour: [217, 119, 6] },
    { label: "CLEAR",         value: batchAnalysis.clear,    colour: [22, 163, 74] },
    { label: "SCANNED TOTAL", value: batchAnalysis.scanned,  colour: [70, 70, 70]  },
  ];
  const cardGap = 12;
  const cardW   = (pageW - margin * 2 - cardGap * 3) / 4;
  const cardH   = 70;
  statCards.forEach((s, i) => {
    const cx = margin + i * (cardW + cardGap);
    doc.setFillColor(s.colour[0], s.colour[1], s.colour[2]);
    doc.roundedRect(cx, y, 4, cardH, 2, 2, "F");
    doc.setFont("helvetica", "bold");
    doc.setFontSize(24);
    doc.setTextColor(s.colour[0], s.colour[1], s.colour[2]);
    doc.text(String(s.value), cx + 16, y + 38);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(8.5);
    doc.setTextColor(110, 110, 110);
    doc.text(s.label, cx + 16, y + 54);
  });
  y += cardH + 20;

  // ── Blocked cameras: image + Grad-CAM overlay + LLM output ──
  startNewSection(`Blocked Cameras (${batchAnalysis.blockedAssets.length})`, [220, 38, 38]);
  if (batchAnalysis.blockedAssets.length === 0) {
    emptyNote("No blocked cameras on this scan.");
  }
  for (let idx = 0; idx < batchAnalysis.blockedAssets.length; idx++) {
    const cam = batchAnalysis.blockedAssets[idx];
    const cardPad  = 16;
    const imgColW  = 210;
    const colGap   = 18;
    const innerX   = margin + cardPad;
    const textColW = pageW - margin * 2 - cardPad * 2 - imgColW - colGap;
    const textX    = innerX + imgColW + colGap;

    const origB64    = cam.result?.original_b64;
    const overlayB64 = cam.result?.overlay_b64;
    const [origDims, overlayDims] = await Promise.all([
      origB64    ? loadImageSize(`data:image/jpeg;base64,${origB64}`)    : Promise.resolve(null),
      overlayB64 ? loadImageSize(`data:image/jpeg;base64,${overlayB64}`) : Promise.resolve(null),
    ]);
    const origH    = origDims    ? imgColW * (origDims.h / origDims.w)    : imgColW * 0.62;
    const overlayH = overlayDims ? imgColW * (overlayDims.h / overlayDims.w) : imgColW * 0.62;
    const imgGapV  = 6;

    // Lay out the text column (OBSERVATION / RISK ASSESSMENT / RECOMMENDED
    // ACTION as separate labelled paragraphs) so we know its total height
    // up front, alongside the stacked images, before deciding page breaks.
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    const sections = parseExplanation(cam.result?.explanation).map(s => ({
      label: s.label,
      lines: doc.splitTextToSize(s.body, textColW) as string[],
    }));
    if (sections.length === 0) {
      sections.push({ label: "", lines: doc.splitTextToSize("No detailed analysis available.", textColW) as string[] });
    }
    const textContentH = sections.reduce(
      (acc, s) => acc + (s.label ? 13 : 0) + s.lines.length * 11.5 + 10,
      0,
    );

    // Whole entry is a bordered card — heading, images and text all inset
    // by cardPad, with real breathing room between one card and the next
    // (rather than the previous version, which packed entries edge to edge
    // with only 18pt between them).
    const headingH = 26;
    const contentH = Math.max(origH + imgGapV + overlayH, textContentH);
    const cardH    = headingH + contentH + cardPad * 2;
    ensureSpace(cardH + 30);

    doc.setFillColor(252, 252, 253);
    doc.setDrawColor(228, 228, 231);
    doc.roundedRect(margin, y, pageW - margin * 2, cardH, 7, 7, "FD");

    // Heading — "N. Region · Name", numbered like the reference report
    doc.setFont("helvetica", "bold");
    doc.setFontSize(11.5);
    doc.setTextColor(20, 20, 20);
    doc.text(`${idx + 1}. ${cam.region} · ${cam.name}`, innerX, y + cardPad + 6);

    const blockTop = y + cardPad + headingH;

    // Left column — original stacked above the Grad-CAM overlay
    let imgY = blockTop;
    if (origB64 && origDims) {
      doc.addImage(`data:image/jpeg;base64,${origB64}`, "JPEG", innerX, imgY, imgColW, origH);
      imgY += origH + imgGapV;
    }
    if (overlayB64 && overlayDims) {
      doc.addImage(`data:image/jpeg;base64,${overlayB64}`, "JPEG", innerX, imgY, imgColW, overlayH);
      imgY += overlayH;
    }

    // Right column — labelled explanation sections
    let textY = blockTop + 2;
    sections.forEach(s => {
      if (s.label) {
        doc.setFont("helvetica", "bold");
        doc.setFontSize(9);
        doc.setTextColor(70, 95, 150);
        doc.text(s.label, textX, textY);
        textY += 13;
      }
      doc.setFont("helvetica", "normal");
      doc.setFontSize(9);
      doc.setTextColor(50, 50, 50);
      doc.text(s.lines, textX, textY);
      textY += s.lines.length * 11.5 + 10;
    });

    y += cardH + 26;
  }

  // ── Grouped blockages ──
  startNewSection(`Grouped Blockages (${batchAnalysis.routes.length})`, [37, 99, 235]);
  if (batchAnalysis.routes.length === 0) {
    emptyNote("No multi-stop routes on this scan.");
  }
  batchAnalysis.routes.forEach((route, i) => {
    const estMin = route.totalKm > 0 ? Math.round(route.totalKm / 40 * 60) : null;
    // jsPDF's built-in "helvetica" font has no glyph for the Unicode arrow
    // (→) used elsewhere in the UI — it prints as garbled characters here,
    // so use a plain ASCII separator instead.
    const stopLine = route.cameras.map(c => c.name).join("  ->  ");
    const lines = doc.splitTextToSize(stopLine, pageW - margin * 2);
    ensureSpace(lines.length * 12 + 28);

    const colour = routeColour(i);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(10.5);
    doc.setTextColor(colour[0], colour[1], colour[2]);
    doc.text(`Route ${i + 1}`, margin, y);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.setTextColor(110, 110, 110);
    const meta = `${route.cameras.length} stop${route.cameras.length !== 1 ? "s" : ""}` +
      (route.totalKm > 0 ? ` · ${route.totalKm.toFixed(1)} km` : "") +
      (estMin ? ` · ~${estMin} min` : "");
    doc.text(meta, pageW - margin, y, { align: "right" });
    y += 14;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(9.5);
    doc.setTextColor(50, 50, 50);
    doc.text(lines, margin, y);
    y += lines.length * 12 + 16;
  });

  // ── Map — placed directly under Suggested Routes (same section, no page
  // break forced in between) so it reads as "here are the routes, here is
  // where they are" rather than being separated by several other sections. ──
  const lightBlocked = batchAnalysis.lightBlocked ?? [];
  const mapH = 300;
  ensureSpace(mapH + 20);
  y += 6;

  const routeMarkers: StaticMapMarker[] = batchAnalysis.routes.flatMap((r, i) => {
    const colour = rgbToHex(routeColour(i));
    return r.cameras.map((c, ci) => ({ lat: c.lat, lng: c.lng, label: String(ci + 1), name: c.name, colour }));
  });
  const isolatedMarkers: StaticMapMarker[] = lightBlocked.map(c => ({
    lat: c.lat, lng: c.lng, label: "1", name: c.name, colour: rgbToHex(ISOLATED_PDF_COLOUR),
  }));
  const staticMapRoutes: StaticMapRoute[] = batchAnalysis.routes.map((r, i) => ({
    points: r.cameras.map(c => ({ lat: c.lat, lng: c.lng })),
    colour: rgbToHex(routeColour(i)),
  }));

  const mapImg = await fetchStaticMap(
    [...routeMarkers, ...isolatedMarkers], staticMapRoutes, 1000, 520,
  );
  if (mapImg) {
    const dims = await loadImageSize(mapImg);
    const drawW = pageW - margin * 2;
    const drawH = Math.min(mapH, drawW * (dims.h / dims.w));
    doc.addImage(mapImg, "PNG", margin, y, drawW, drawH);
    y += drawH + 8;
    y += drawRouteLegendBelow(doc, batchAnalysis, margin, y, drawW);
    y += 10;
  } else {
    // Backend unreachable — fall back to the hand-drawn schematic (which
    // already includes its own legend) so the report still has something
    // rather than a gap.
    drawSchematicMap(doc, batchAnalysis, margin, y, pageW - margin * 2, mapH);
    y += mapH + 10;
  }

  // ── Individual / isolated cases ──
  startNewSection(`Standalone Blockage (${lightBlocked.length})`, ISOLATED_PDF_COLOUR);
  if (lightBlocked.length === 0) {
    emptyNote("No isolated blockages on this scan.");
  } else {
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9.5);
    doc.setTextColor(50, 50, 50);
    lightBlocked.forEach(cam => {
      ensureSpace(14);
      doc.text(`•  ${cam.name} — ${cam.region}`, margin, y);
      y += 14;
    });
    y += 8;
  }

  // ── Needs review / Clear — just a mention, no images ──
  const flaggedCams = batchAnalysis.allScanned.filter(c => c.riskFresh === "FLAGGED");
  startNewSection(`Need Review (${flaggedCams.length})`, [217, 119, 6]);
  if (flaggedCams.length === 0) emptyNote("No cameras flagged for manual review on this scan.");
  else mentionList(flaggedCams);

  const clearCams = batchAnalysis.allScanned.filter(c => c.riskFresh === "LOW");
  startNewSection(`Clear (${clearCams.length})`, [22, 163, 74]);
  if (clearCams.length === 0) emptyNote("No clear cameras on this scan.");
  else mentionList(clearCams);

  const dateSlug = new Date(batchAnalysis.timestamp).toISOString().slice(0, 10);
  doc.save(`blockage-report-${dateSlug}.pdf`);
}

