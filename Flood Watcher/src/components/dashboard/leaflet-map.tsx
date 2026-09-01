import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { Asset } from "@/lib/dashboard-data";

delete (L.Icon.Default.prototype as unknown as Record<string, unknown>)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl:       "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl:     "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

const RISK_COLOR: Record<string, string> = {
  high:    "#c0463a",
  medium:  "#b07828",
  low:     "#3a7a48",
  offline: "#888",
};

function makeIcon(risk: string, selected: boolean, inRoute: boolean): L.CircleMarkerOptions {
  const color = RISK_COLOR[risk] ?? RISK_COLOR.low;
  return {
    radius:      selected ? 10 : inRoute ? 9 : 7,
    fillColor:   inRoute ? "#c47c1a" : color,
    color:       selected ? "#fff" : inRoute ? "#fff" : color,
    weight:      selected ? 2.5 : inRoute ? 2 : 1.5,
    opacity:     1,
    fillOpacity: selected ? 1 : 0.88,
  };
}

interface Props {
  assets: Asset[];
  selectedId: string;
  onSelect: (id: string) => void;
  /** Ordered list of cameras forming the suggested inspection route */
  suggestedRoute?: Asset[];
}

export function LeafletMap({ assets, selectedId, onSelect, suggestedRoute = [] }: Props) {
  const containerRef  = useRef<HTMLDivElement>(null);
  const mapRef        = useRef<L.Map | null>(null);
  const markersRef    = useRef<Map<string, L.CircleMarker>>(new Map());
  const routesRef     = useRef<L.Polyline[]>([]);
  const suggRouteRef  = useRef<L.Polyline | null>(null);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = L.map(containerRef.current, {
      center: [50.65, -4.1],
      zoom: 9,
      zoomControl: false,
      attributionControl: false,
    });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
    }).addTo(map);
    L.control.attribution({ position: "bottomright", prefix: false })
      .addTo(map).setPrefix("© OSM");
    L.control.zoom({ position: "topright" }).addTo(map);
    mapRef.current = map;
    return () => { map.remove(); mapRef.current = null; };
  }, []);

  // Update markers + background HIGH-risk lines
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const routeIds = new Set(suggestedRoute.map(a => a.id));
    const assetMap = new Map(assets.map(a => [a.id, a]));

    markersRef.current.forEach((m, id) => {
      if (!assetMap.has(id)) { m.remove(); markersRef.current.delete(id); }
    });

    assets.forEach(a => {
      const isSelected = a.id === selectedId;
      const inRoute    = routeIds.has(a.id);
      const opts       = makeIcon(a.risk, isSelected, inRoute);
      const existing   = markersRef.current.get(a.id);
      if (existing) {
        existing.setStyle(opts);
        existing.setRadius(opts.radius!);
      } else {
        const m = L.circleMarker([a.lat, a.lng], opts).addTo(map);
        m.bindTooltip(`<b>${a.name}</b><br/>${a.region} · ${a.risk.toUpperCase()}`,
          { direction: "top", offset: [0, -6] });
        m.on("click", () => onSelect(a.id));
        markersRef.current.set(a.id, m);
      }
    });

    if (selectedId) markersRef.current.get(selectedId)?.bringToFront();

    // Background dashed HIGH-risk lines
    routesRef.current.forEach(p => p.remove());
    routesRef.current = [];
    for (const region of ["Cornwall", "Devon"] as const) {
      const high = assets
        .filter(a => a.region === region && a.risk === "high" && a.status !== "offline")
        .sort((a, b) => a.lat - b.lat);
      if (high.length > 1) {
        routesRef.current.push(
          L.polyline(high.map(a => [a.lat, a.lng] as L.LatLngTuple), {
            color: "#c0463a", weight: 1.5, opacity: 0.35, dashArray: "5 5",
          }).addTo(map)
        );
      }
    }
  }, [assets, selectedId, onSelect, suggestedRoute]);

  // Draw suggested route as a prominent amber solid line
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (suggRouteRef.current) { suggRouteRef.current.remove(); suggRouteRef.current = null; }

    if (suggestedRoute.length > 1) {
      const latlngs = suggestedRoute.map(a => [a.lat, a.lng] as L.LatLngTuple);
      suggRouteRef.current = L.polyline(latlngs, {
        color:   "#c47c1a",
        weight:  3.5,
        opacity: 0.9,
        dashArray: undefined,
      }).addTo(map);
      suggRouteRef.current.bringToFront();

      // Number labels along the route
      suggestedRoute.forEach((a, i) => {
        L.marker([a.lat, a.lng], {
          icon: L.divIcon({
            className: "",
            html: `<div style="background:#c47c1a;color:#fff;font-size:10px;font-weight:700;
                          width:16px;height:16px;border-radius:50%;display:flex;
                          align-items:center;justify-content:center;
                          border:1.5px solid #fff;box-shadow:0 1px 3px rgba(0,0,0,.3)">${i + 1}</div>`,
            iconSize:   [16, 16],
            iconAnchor: [8, 8],
          }),
        }).addTo(map);
      });

      // Fit bounds to the route
      map.fitBounds(L.latLngBounds(latlngs), { padding: [30, 30], maxZoom: 11 });
    }
  }, [suggestedRoute]);

  // Pan to selected
  useEffect(() => {
    const map = mapRef.current;
    if (!map || suggestedRoute.length > 1) return;
    const sel = assets.find(a => a.id === selectedId);
    if (sel) map.panTo([sel.lat, sel.lng], { animate: true, duration: 0.4 });
  }, [selectedId, assets, suggestedRoute]);

  const showRouteInLegend = suggestedRoute.length > 1;

  return (
    <div className="relative h-full w-full">
      <div ref={containerRef} className="h-full w-full" />
      <div className="pointer-events-none absolute bottom-6 left-3 z-[1000] rounded-[3px] border border-border bg-surface/95 px-3 py-2 backdrop-blur-sm">
        <div className="label-eyebrow mb-1.5">Legend</div>
        <ul className="space-y-1 text-[11px]">
          {([ ["high","Blocked"], ["low","Clear"], ["offline","Offline"] ] as [string,string][]).map(([r,l]) => (
            <li key={r} className="flex items-center gap-2">
              <span className="inline-block size-2.5 rounded-full" style={{ background: RISK_COLOR[r] }} />
              <span>{l}</span>
            </li>
          ))}
          {showRouteInLegend && (
            <li className="flex items-center gap-2">
              <span className="inline-block h-[3px] w-5 rounded-sm" style={{ background: "#c47c1a" }} />
              <span className="font-medium" style={{ color: "#c47c1a" }}>Suggested route</span>
            </li>
          )}
        </ul>
      </div>
      <div className="pointer-events-none absolute right-3 top-3 z-[1000]">
        <div className="rounded-[3px] border border-border bg-surface/95 px-2 py-1 text-[11px] mono text-muted-foreground backdrop-blur-sm">
          Devon & Cornwall · SW Region
        </div>
      </div>
    </div>
  );
}
