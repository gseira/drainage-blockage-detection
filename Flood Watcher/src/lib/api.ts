/**
 * api.ts
 * ------
 * Types and fetch helpers for the DrainWatch FastAPI backend.
 *
 * BASE defaults to localhost for normal local development, but is
 * overridable via VITE_API_BASE — needed when sharing the app over a
 * tunnel (e.g. Cloudflare Tunnel/ngrok): a visitor's browser resolves
 * "127.0.0.1" to THEIR OWN machine, not yours, so the hardcoded localhost
 * address would silently fail to load any data for anyone but you.
 * Set VITE_API_BASE to the tunnel's public backend URL when demoing:
 *   VITE_API_BASE=https://your-backend-tunnel-url.trycloudflare.com npm run dev
 */

const BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

/**
 * Controls where Mark Blocked / Mark Clear buttons appear.
 *
 * true  (default): every camera, in every category (BLOCKED/CLEAR/Needs
 *        Review) can be corrected — for active review passes, run as many
 *        times a day as you like, to build up real corrected examples and
 *        feed app/correction_memory.py across the whole camera fleet, not
 *        just the uncertain cases.
 * false: buttons only appear on FLAGGED (Needs Review) results — the
 *        original, narrower behaviour, intended for once this is "launched"
 *        and you no longer want every confident BLOCKED/CLEAR call exposed
 *        to correction by default.
 *
 * Set VITE_REVIEW_ALL_CASES=false (e.g. in .env.production) to switch to
 * launch mode without touching any component code.
 */
export const REVIEW_ALL_CASES = import.meta.env.VITE_REVIEW_ALL_CASES !== "false";

// ── Backend types ─────────────────────────────────────────────────────────────

export interface WebcamDef {
  name: string;
  campath: string;
  region: "Cornwall" | "Devon";
  base_url: string;
  gallery_prefix: string;
}

export interface ResultRow {
  id: number;
  name: string;
  campath: string;
  captured_at: string;
  risk: "HIGH" | "LOW" | "FLAGGED" | "OFFLINE" | "UNKNOWN" | "BLOCKED";
  p_blocked: number | null;
  heatmap_coverage?: number | null;
  explanation: string | null;
  overlay_b64: string | null;
  heatmap_b64: string | null;
  original_b64?: string | null;
  quality_flags?: string[];
  needs_review?: boolean;
  created_at: string;
}

export interface HistoryRow {
  id: number;
  name: string;
  campath: string;
  captured_at: string;
  risk: "HIGH" | "LOW" | "FLAGGED" | "OFFLINE" | "UNKNOWN" | "BLOCKED";
  p_blocked: number | null;
  heatmap_coverage?: number | null;
  created_at: string;
}

export interface StatRow {
  day: string;
  total: number;
  high: number;
  low: number;
  avg_p_blocked: number;
}

export interface PredictResult {
  prediction: "BLOCKED" | "CLEAR" | "FLAGGED" | "OFFLINE" | "UNKNOWN";
  risk: "HIGH" | "LOW" | "FLAGGED" | "OFFLINE" | "UNKNOWN" | "BLOCKED";
  p_blocked: number | null;
  heatmap_coverage: number | null;
  explanation: string;
  overlay_b64: string;
  heatmap_b64: string;
  original_b64?: string;
  quality_flags: string[];
  needs_review: boolean;
  sharpness?: number;
  brightness?: number;
  filename?: string;
  camera_name?: string;
  captured_at?: string;
  source_url?: string;
}

export interface BatchResult extends PredictResult {
  filename: string;
  error?: string;
}

// ── Fetch helpers ─────────────────────────────────────────────────────────────

export async function fetchWebcams(): Promise<WebcamDef[]> {
  const res = await fetch(`${BASE}/webcams`);
  if (!res.ok) throw new Error(`/webcams ${res.status}`);
  const data = await res.json();
  return data.cameras ?? data.webcams ?? data;
}

export async function fetchLatestResults(): Promise<ResultRow[]> {
  const res = await fetch(`${BASE}/results/latest`);
  if (!res.ok) throw new Error(`/results/latest ${res.status}`);
  const data = await res.json();
  return data.results ?? data;
}

export async function fetchHistory(
  campath: string,
  days = 7,
  limit = 20,
): Promise<HistoryRow[]> {
  const params = new URLSearchParams({ campath, days: String(days), limit: String(limit) });
  const res = await fetch(`${BASE}/results/history?${params}`);
  if (!res.ok) throw new Error(`/results/history ${res.status}`);
  const data = await res.json();
  return data.results ?? data;
}

export async function fetchStats(days = 30): Promise<StatRow[]> {
  const res = await fetch(`${BASE}/results/stats?days=${days}`);
  if (!res.ok) throw new Error(`/results/stats ${res.status}`);
  const data = await res.json();
  return data.stats ?? data;
}

export async function fetchHealth() {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error(`/health ${res.status}`);
  return res.json() as Promise<{
    status: string;
    model_loaded: boolean;
    monitor_running: boolean;
    next_sweep: string | null;
  }>;
}

/** Trigger live webcam analysis for one camera (POST /predict/webcam). */
export async function analyzeWebcam(campath: string, camname = ""): Promise<PredictResult> {
  const params = new URLSearchParams({ campath, camname });
  const res = await fetch(`${BASE}/predict/webcam?${params}`, { method: "POST" });
  if (!res.ok) throw new Error(`/predict/webcam ${res.status}`);
  return res.json();
}

export interface ReclassifyResponse {
  ok: boolean;
  campath: string;
  risk: "BLOCKED" | "LOW";
  overlay_generated: boolean;
  overlay_b64: string | null;
  heatmap_b64: string | null;
  heatmap_coverage: number | null;
  explanation: string | null;
}

/**
 * Manually resolve an uncertain (FLAGGED) result into a definite BLOCKED or
 * LOW verdict (POST /results/reclassify). Persists the human correction so
 * it survives past this session, not just a client-side relabel.
 *
 * When marking BLOCKED, the backend also generates a real GradCAM overlay
 * from the stored original frame (the automatic pipeline never ran GradCAM
 * for a FLAGGED result) — returns it here so the caller can update its
 * already-loaded result in place instead of needing a full refetch.
 */
export async function reclassifyResult(
  campath: string,
  risk: "BLOCKED" | "LOW",
): Promise<ReclassifyResponse> {
  const params = new URLSearchParams({ campath, risk });
  const res = await fetch(`${BASE}/results/reclassify?${params}`, { method: "POST" });
  if (!res.ok) throw new Error(`/results/reclassify ${res.status}`);
  return res.json();
}

/** Upload a single image file for classification (POST /predict). */
export async function analyzeUpload(file: File): Promise<PredictResult> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${BASE}/predict`, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

/** Upload multiple images for batch classification (POST /predict/batch). */
export async function analyzeBatch(
  files: File[],
): Promise<{ total: number; results: BatchResult[] }> {
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  const res = await fetch(`${BASE}/predict/batch`, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

/** Build the live EA gallery URL for a camera */
export function eaImageUrl(cam: WebcamDef): string {
  return `${cam.base_url}/${cam.gallery_prefix}/gallerylatestpic.php?campath=${cam.campath}`;
}

/**
 * Backend-proxied JPEG URL for a camera.
 * Use this in <img src> — the EA servers don't send CORS headers so the browser
 * can't load their images directly. The backend fetches and forwards the bytes.
 * Add a timestamp query param so the browser refetches on each analysis.
 */
export function webcamProxyUrl(campath: string, bust = 0): string {
  return `${BASE}/webcam/image?campath=${encodeURIComponent(campath)}${bust ? `&t=${bust}` : ""}`;
}

// ── Static map (for PDF export) ────────────────────────────────────────────────

export interface StaticMapMarker {
  lat: number;
  lng: number;
  label?: string; // short label drawn INSIDE the pin (usually a stop number)
  name?: string;  // camera name drawn NEXT TO the pin, on a legible pill background
  colour?: string; // hex, e.g. "#c0463a"
}

export interface StaticMapRoute {
  points: StaticMapMarker[];
  colour?: string;
}

/**
 * Ask the backend to composite a real OSM basemap with the given routes/pins
 * drawn on top, and return it as a data URL ready for jsPDF's addImage().
 *
 * This goes through the backend (not a direct browser fetch of OSM tiles)
 * because OSM's tile servers don't send CORS headers — a browser can't read
 * those bytes back out once drawn to a canvas, which is what embedding an
 * image in a PDF requires. Server-to-server requests aren't subject to CORS
 * at all, so the compositing happens there instead. Returns null if the
 * request fails (e.g. no network), so callers can fall back gracefully.
 */
export async function fetchStaticMap(
  markers: StaticMapMarker[],
  routes: StaticMapRoute[],
  width = 900,
  height = 480,
): Promise<string | null> {
  try {
    const res = await fetch(`${BASE}/staticmap`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ markers, routes, width, height }),
    });
    if (!res.ok) return null;
    const blob = await res.blob();
    return await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload  = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  } catch {
    return null;
  }
}

/**
 * Derive a stable, unique display ID from a campath.
 *
 * Previously this compressed each camera down to "REGION-XXX#" (region +
 * first 3 letters of the location + camera number), which collided
 * constantly — e.g. "cornwall/BodminFlaxmoor/cam1" and
 * "cornwall/BodminPetrocsWell/cam1" both hashed to "CW-BOD1", and
 * "BarnstapleBradiford" / "BarnstapleNewportRoad" / "BarnstaplePortmarshLane"
 * (three different Devon cameras) all hashed to "DV-BAR1". Since selection
 * state elsewhere is a Set keyed by this id, colliding ids meant checking
 * one camera silently checked every other camera sharing its id too.
 *
 * campath is already unique per camera everywhere else in the system, so
 * just sanitise it directly instead of re-compressing it into something lossy.
 */
export function campathToId(campath: string): string {
  return campath.replace(/[^A-Za-z0-9]+/g, "-").toLowerCase();
}
