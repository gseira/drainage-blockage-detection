import { useEffect, useImperativeHandle, useRef, forwardRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

export interface BatchRouteMapHandle {
  flyTo: (lat: number, lng: number) => void;
}

export interface MapRoute {
  type: "immediate" | "bundle" | "monitor" | "isolated";
  cameras: { id: string; name: string; lat: number; lng: number }[];
  totalKm: number;
}

export interface MapSummary {
  activeRoutes: number;
  blocked: number;
  critical: number;
  scanTime: string;
}

export interface CameraPin {
  id: string;
  name: string;
  lat: number;
  lng: number;
  region: string;
}

interface Props {
  routes: MapRoute[];
  allCameras?: CameraPin[];
  flaggedCameras?: CameraPin[];
  hoveredRouteIndex: number | null;
  summary: MapSummary;
  height?: string;
  onMarkerClick?: (cameraId: string) => void;
}

const COLORS = {
  immediate: "#c0463a",
  bundle:    "#b07828",
  monitor:   "#888",
  isolated:  "#7c3aed",
} as const;

// Needs Review (FLAGGED) pins — orange badge, same shape/size as the
// red/purple route badges below.
const FLAGGED_COLOR = "#e08a2e";

type LayerGroup = {
  polyline: L.Polyline | null;
  markers: Array<L.Marker | L.CircleMarker>;
};

async function fetchRoadGeometry(
  cameras: { lat: number; lng: number }[],
  signal: AbortSignal,
): Promise<L.LatLngTuple[]> {
  if (cameras.length < 2) return cameras.map(c => [c.lat, c.lng]);
  const coords = cameras.map(c => `${c.lng},${c.lat}`).join(";");
  const url = `https://router.project-osrm.org/route/v1/driving/${coords}?overview=full&geometries=geojson`;
  try {
    const res = await fetch(url, { signal });
    if (!res.ok) throw new Error(`OSRM ${res.status}`);
    const data = await res.json();
    if (data.routes?.[0]?.geometry?.coordinates) {
      return (data.routes[0].geometry.coordinates as [number, number][])
        .map(([lng, lat]) => [lat, lng]);
    }
  } catch { /* AbortError or network error — fall through to straight lines */ }
  return cameras.map(c => [c.lat, c.lng]);
}

export const BatchRouteMap = forwardRef<BatchRouteMapHandle, Props>(function BatchRouteMap(
  { routes, allCameras = [], flaggedCameras = [], hoveredRouteIndex, summary, height = "480px", onMarkerClick }, ref
) {
  const containerRef     = useRef<HTMLDivElement>(null);
  const mapRef           = useRef<L.Map | null>(null);
  const layersRef        = useRef<LayerGroup[]>([]);
  const pinLayerRef      = useRef<L.CircleMarker[]>([]);   // pre-scan camera dots
  const flaggedLayerRef  = useRef<L.Marker[]>([]);         // Needs Review badges
  const onMarkerClickRef = useRef(onMarkerClick);
  onMarkerClickRef.current = onMarkerClick;

  // ── Effect 1: create the Leaflet map once ────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = L.map(containerRef.current, {
      center: [50.65, -4.1], zoom: 9,
      zoomControl: true, attributionControl: false,
    });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 18 }).addTo(map);
    L.control.attribution({ position: "bottomright", prefix: false })
      .addTo(map).setPrefix("© OSM contributors");
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
      layersRef.current = [];
      pinLayerRef.current = [];
      flaggedLayerRef.current = [];
    };
  }, []);

  // ── Imperative handle — lets parent pan the map to a camera ──────────────────
  useImperativeHandle(ref, () => ({
    flyTo: (lat: number, lng: number) => {
      mapRef.current?.flyTo([lat, lng], 14, { duration: 0.8 });
    },
  }));

  // ── Effect 1b: draw pre-scan camera pins (hidden once routes are shown) ──────
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    // Remove existing pins
    pinLayerRef.current.forEach(m => m.remove());
    pinLayerRef.current = [];

    // Hide pins once a scan produces routes
    if (routes.length > 0 || allCameras.length === 0) return;

    const cornwallColour = "#4a7fcb";
    const devonColour    = "#6a9e5a";

    allCameras.forEach(cam => {
      const colour = cam.region === "Devon" ? devonColour : cornwallColour;
      const m = L.circleMarker([cam.lat, cam.lng], {
        radius: 5, fillColor: colour, color: "#fff",
        weight: 1.5, opacity: 1, fillOpacity: 0.85,
      })
        .bindTooltip(`<b>${cam.name}</b><br/>${cam.region}`, { direction: "top", offset: [0, -6] })
        .addTo(map);
      pinLayerRef.current.push(m);
    });

    // Fit map to all cameras on first load
    const bounds = L.latLngBounds(allCameras.map(c => [c.lat, c.lng] as L.LatLngTuple));
    if (bounds.isValid()) map.fitBounds(bounds, { padding: [32, 32], maxZoom: 10 });

  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allCameras, routes]);

  // ── Effect 1c: draw Needs Review (FLAGGED) pins ──────────────────────────────
  // Same round numbered-badge style as the immediate/bundle/isolated route
  // pins in Effect 2 below (24x24 circle, white border, drop shadow) — just
  // orange, with "?" instead of a route-order number since these aren't
  // part of any route sequence. Unlike the pre-scan `allCameras` pins, these
  // represent a post-scan result (riskFresh === "FLAGGED") so they stay
  // visible alongside the blocked route pins once a scan has run, not just
  // before one.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    flaggedLayerRef.current.forEach(m => m.remove());
    flaggedLayerRef.current = [];

    flaggedCameras.forEach(cam => {
      const cb = onMarkerClickRef.current;
      const cursor = cb ? "pointer" : "default";
      const m = L.marker([cam.lat, cam.lng], {
        icon: L.divIcon({
          className: "",
          // Smaller than the numbered route/isolated pins (24x24) — these
          // don't need to fit a legible digit inside them, so they can shrink
          // while staying clearly visible via the white border + shadow.
          html: `<div style="background:${FLAGGED_COLOR};width:14px;height:14px;border-radius:50%;border:2px solid #fff;cursor:${cursor};box-shadow:0 1px 5px rgba(0,0,0,.5)"></div>`,
          iconSize: [14, 14], iconAnchor: [7, 7],
        }),
      })
        .bindTooltip(`<b>${cam.name}</b><br/>Need Review`, { direction: "top", offset: [0, -9] })
        .addTo(map);
      if (cb) m.on("click", () => cb(cam.id));
      flaggedLayerRef.current.push(m);
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [flaggedCameras]);

  // ── Effect 2: draw / redraw routes whenever `routes` changes ─────────────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    // Clear existing route layers
    layersRef.current.forEach(g => {
      g.polyline?.remove();
      g.markers.forEach(m => m.remove());
    });
    layersRef.current = [];

    if (routes.length === 0) return; // no scan — leave base map clean

    const controller = new AbortController();
    layersRef.current = routes.map(() => ({ polyline: null, markers: [] }));

    // Fit bounds to all cameras — route (blocked) cameras AND flagged
    // (Needs Review) ones. Needs Review cases often sit far from any blocked
    // cluster, so leaving them out here was the actual bug behind "the
    // orange dots don't show up": they were being drawn, just outside the
    // viewport the map zoomed/panned to.
    const allLatLngs = routes.flatMap(r => r.cameras.map<L.LatLngTuple>(c => [c.lat, c.lng]));
    const flaggedLatLngs = flaggedCameras.map<L.LatLngTuple>(c => [c.lat, c.lng]);
    const boundsPoints = [...allLatLngs, ...flaggedLatLngs];
    if (boundsPoints.length > 0) {
      map.fitBounds(L.latLngBounds(boundsPoints), { padding: [32, 32], maxZoom: 11 });
    }

    // Phase 1: placeholder straight lines + markers immediately
    routes.forEach((route, routeIdx) => {
      const color = COLORS[route.type];
      const group = layersRef.current[routeIdx];

      if (route.type !== "monitor" && route.cameras.length > 1) {
        const latlngs = route.cameras.map<L.LatLngTuple>(c => [c.lat, c.lng]);
        group.polyline = L.polyline(latlngs, { color, weight: 3, opacity: 0.4, dashArray: "4 6" }).addTo(map);
      }

      route.cameras.forEach((cam, i) => {
        const cb = onMarkerClickRef.current;
        if (route.type === "monitor") {
          const m = L.circleMarker([cam.lat, cam.lng], {
            radius: 6, fillColor: color, color: "#fff",
            weight: 1.5, opacity: 1, fillOpacity: 0.8,
          }).bindTooltip(`<b>${cam.name}</b><br/>Monitor · click for details`, { direction: "top", offset: [0, -6] })
            .addTo(map);
          if (cb) {
            m.on("click", () => cb(cam.id));
            m.on("add", () => { const el = (m as L.CircleMarker).getElement(); if (el) (el as HTMLElement).style.cursor = "pointer"; });
          }
          group.markers.push(m);
        } else {
          // Numbered pin — route stops read 1, 2, 3…; an isolated camera has
          // no companions to number against, so it always shows "1".
          const cursor = cb ? "pointer" : "default";
          const label  = route.type === "isolated" ? "1" : String(i + 1);
          const tooltip = route.type === "isolated"
            ? `<b>${cam.name}</b><br/>Standalone blockage · click for details`
            : `<b>${cam.name}</b> · click for details`;
          const m = L.marker([cam.lat, cam.lng], {
            icon: L.divIcon({
              className: "",
              html: `<div style="background:${color};color:#fff;font-size:11px;font-weight:700;width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;border:2px solid #fff;cursor:${cursor};box-shadow:0 1px 6px rgba(0,0,0,.45)">${label}</div>`,
              iconSize: [24, 24], iconAnchor: [12, 12],
            }),
          }).bindTooltip(tooltip, { direction: "top", offset: [0, -13] })
            .addTo(map);
          if (cb) m.on("click", () => cb(cam.id));
          group.markers.push(m);
        }
      });
    });

    // Phase 2: replace placeholder lines with real road geometry
    (async () => {
      await Promise.all(
        routes.map(async (route, routeIdx) => {
          if (route.type === "monitor" || route.cameras.length < 2) return;
          const color = COLORS[route.type];
          const roadLatlngs = await fetchRoadGeometry(route.cameras, controller.signal);
          if (controller.signal.aborted || !mapRef.current) return;
          const group = layersRef.current[routeIdx];
          group.polyline?.remove();
          group.polyline = L.polyline(roadLatlngs, {
            color, weight: 3.5, opacity: 0.85,
            dashArray: route.type === "bundle" ? "8 5" : undefined,
          }).addTo(mapRef.current);
          group.polyline.bringToBack();
        })
      );
    })();

    return () => { controller.abort(); };
  // Deliberately NOT keyed on flaggedCameras too: routes and flaggedCameras
  // are recomputed together from the same batchAnalysis update, so this
  // still refits whenever new Needs Review data arrives. Adding
  // flaggedCameras here would also re-run this on every unrelated parent
  // re-render (it's a fresh array reference each render) and needlessly
  // re-fetch OSRM road geometry every time.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [routes]);

  // ── Effect 3: highlight / fade on card hover ──────────────────────────────────
  useEffect(() => {
    layersRef.current.forEach((group, idx) => {
      const active = hoveredRouteIndex === null || hoveredRouteIndex === idx;
      group.polyline?.setStyle({ opacity: active ? 0.85 : 0.07 });
      group.markers.forEach(m => {
        const el = (m as L.Marker).getElement?.();
        if (el) el.style.opacity = active ? "1" : "0.1";
        if (m instanceof L.CircleMarker) {
          m.setStyle({ fillOpacity: active ? 0.8 : 0.07, opacity: active ? 1 : 0.07 });
        }
      });
    });
  }, [hoveredRouteIndex]);

  const hasRoutes = routes.length > 0;

  return (
    <div className="relative overflow-hidden"
      style={{ height, border: "1px solid var(--color-border)" }}>

      <div ref={containerRef} className="h-full w-full" />

      {/* Summary chip — top left, only shown after a scan */}
      {hasRoutes && (
        <div className="pointer-events-none absolute top-3 left-3 z-[1000] flex items-center gap-3 rounded-[6px] px-4 py-3"
          style={{ background: "rgba(255,255,255,0.95)", border: "1px solid rgba(0,0,0,0.12)", backdropFilter: "blur(6px)", boxShadow: "0 2px 8px rgba(0,0,0,0.08)" }}>
          <span className="text-[15px] font-semibold text-foreground/80">
            {summary.activeRoutes} route{summary.activeRoutes !== 1 ? "s" : ""}
          </span>
          <span className="text-foreground/25">·</span>
          <span className="text-[15px] font-semibold" style={{ color: "var(--color-risk-high)" }}>
            {summary.blocked} blocked
          </span>
          <span className="text-foreground/25">·</span>
          <span className="text-[15px] font-semibold" style={{ color: FLAGGED_COLOR }}>
            {flaggedCameras.length} Need Review
          </span>
          {summary.scanTime && (
            <>
              <span className="text-foreground/25">·</span>
              <span className="text-[12px] text-muted-foreground/70">{summary.scanTime}</span>
            </>
          )}
        </div>
      )}

      {/* Legend — separate box, stacked vertically, top left below the summary chip */}
      {hasRoutes && (
        <div className="pointer-events-none absolute top-[76px] left-3 z-[1000] flex flex-col gap-2.5 rounded-[6px] px-4 py-3.5"
          style={{ background: "rgba(255,255,255,0.95)", border: "1px solid rgba(0,0,0,0.12)", backdropFilter: "blur(6px)", boxShadow: "0 2px 8px rgba(0,0,0,0.08)" }}>
          <div className="flex items-center gap-2.5">
            <span style={{ width: 18, height: 18, borderRadius: 3, background: COLORS.immediate, display: "inline-block" }} />
            <span className="text-[14px] font-medium text-foreground/80">Grouped Blockages</span>
          </div>
          <div className="flex items-center gap-2.5">
            <span style={{ width: 18, height: 18, borderRadius: 3, background: COLORS.isolated, display: "inline-block" }} />
            <span className="text-[14px] font-medium text-foreground/80">Standalone Blockage</span>
          </div>
          <div className="flex items-center gap-2.5">
            <span style={{ width: 18, height: 18, borderRadius: 3, background: FLAGGED_COLOR, display: "inline-block" }} />
            <span className="text-[14px] font-medium text-foreground/80">Need Review</span>
          </div>
        </div>
      )}
    </div>
  );
});
