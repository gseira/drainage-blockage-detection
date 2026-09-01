import type { Asset } from "@/lib/dashboard-data";

interface Props {
  assets: Asset[];
  selectedId: string;
  onSelect: (id: string) => void;
}

const riskColor = {
  high:   "var(--color-risk-high)",
  medium: "var(--color-risk-medium)",
  low:    "var(--color-risk-low)",
} as const;

// Stylised SVG map of Devon & Cornwall — engineered GIS feel, not literal cartography.
export function OperationsMap({ assets, selectedId, onSelect }: Props) {
  const selected = assets.find((a) => a.id === selectedId);

  // Highlight route through blocked (high-risk) assets
  const highRisk = assets.filter((a) => a.risk === "high").slice(0, 3);
  const routeD = highRisk
    .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`)
    .join(" ");

  return (
    <div className="relative h-full w-full overflow-hidden bg-[var(--color-map-land)]">
      <svg viewBox="0 0 1000 640" className="h-full w-full block" preserveAspectRatio="xMidYMid slice">
        {/* Hairline grid */}
        <defs>
          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="var(--color-map-grid)" strokeWidth="0.5" />
          </pattern>
          <pattern id="gridMajor" width="200" height="200" patternUnits="userSpaceOnUse">
            <path d="M 200 0 L 0 0 0 200" fill="none" stroke="var(--color-border)" strokeWidth="0.6" />
          </pattern>
        </defs>
        <rect width="1000" height="640" fill="url(#grid)" />
        <rect width="1000" height="640" fill="url(#gridMajor)" />

        {/* Coastline / land mass — abstracted Devon & Cornwall peninsula */}
        <path
          d="M 90 540 C 60 500 70 460 110 430 C 150 400 160 380 210 380 C 250 380 280 360 320 360 C 360 360 380 340 430 330 C 480 320 520 300 560 280 C 600 260 640 240 690 235 C 740 230 780 230 820 245 C 860 260 880 285 870 320 C 860 360 820 380 770 385 C 720 390 680 405 650 425 C 620 445 580 460 540 470 C 500 480 460 490 420 500 C 380 510 340 520 300 530 C 260 540 220 555 180 560 C 140 565 110 560 90 540 Z"
          fill="var(--color-surface)"
          stroke="var(--color-border-strong)"
          strokeWidth="1"
        />

        {/* Rivers — thin blue lines */}
        <g stroke="var(--color-map-water)" strokeWidth="1.8" fill="none" strokeLinecap="round">
          <path d="M 612 248 C 605 280 600 310 595 340 C 590 370 575 400 555 392" />
          <path d="M 690 286 C 685 320 670 360 640 400" />
          <path d="M 478 310 C 490 340 500 360 502 340" />
          <path d="M 360 412 C 340 440 310 460 250 460 C 220 460 200 450 188 408" />
        </g>

        {/* Roads — pale */}
        <g stroke="var(--color-map-road)" strokeWidth="1.2" fill="none" strokeDasharray="0">
          <path d="M 120 460 L 300 440 L 480 420 L 660 380 L 840 330" />
          <path d="M 300 530 L 380 480 L 480 420 L 600 360 L 720 300" />
        </g>

        {/* Active route — connects high-risk sites */}
        {highRisk.length > 1 && (
          <g>
            <path
              d={routeD}
              fill="none"
              stroke="var(--color-map-route)"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d={routeD}
              fill="none"
              stroke="var(--color-map-route)"
              strokeWidth="6"
              strokeOpacity="0.12"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </g>
        )}

        {/* Region labels */}
        <g fill="var(--color-muted-foreground)" fontFamily="var(--font-sans)" fontSize="10" letterSpacing="2">
          <text x="540" y="220" opacity="0.7">DEVON</text>
          <text x="220" y="360" opacity="0.7">CORNWALL</text>
        </g>

        {/* Assets */}
        {assets.map((a) => {
          const isSelected = a.campath === selectedId;
          const c = a.status === "offline" ? "var(--color-muted-foreground)" : riskColor[a.risk];
          return (
            <g
              key={a.campath}
              transform={`translate(${a.x} ${a.y})`}
              onClick={() => onSelect(a.campath)}
              className="cursor-pointer"
            >
              {isSelected && (
                <circle r="14" fill="none" stroke="var(--color-accent)" strokeWidth="1.2" />
              )}
              <circle r="5.5" fill="var(--color-surface)" stroke={c} strokeWidth="1.5" />
              <circle r="2.5" fill={c} />
              {(isSelected || a.risk === "high") && (
                <text
                  x="10"
                  y="3"
                  fontFamily="var(--font-mono)"
                  fontSize="9.5"
                  fill="var(--color-foreground)"
                >
                  {a.name.split(" ").slice(0, 2).join(" ")}
                </text>
              )}
            </g>
          );
        })}

        {/* Selected crosshair */}
        {selected && (
          <g stroke="var(--color-accent)" strokeWidth="0.6" opacity="0.45">
            <line x1={selected.x} y1="0"  x2={selected.x} y2="640" />
            <line x1="0" y1={selected.y} x2="1000" y2={selected.y} />
          </g>
        )}
      </svg>

      {/* Map chrome */}
      <div className="pointer-events-none absolute inset-0 flex flex-col">
        <div className="flex items-start justify-between p-3">
          <div className="pointer-events-auto flex items-center gap-1 rounded-[3px] border border-border bg-surface/95 px-2 py-1 backdrop-blur-sm">
            <button className="label-eyebrow px-1.5 py-0.5 text-foreground">Operational</button>
            <span className="h-3 w-px bg-border" />
            <button className="label-eyebrow px-1.5 py-0.5 hover:text-foreground">Hydrology</button>
            <span className="h-3 w-px bg-border" />
            <button className="label-eyebrow px-1.5 py-0.5 hover:text-foreground">Satellite</button>
          </div>
          <div className="pointer-events-auto flex flex-col gap-1.5">
            <div className="flex items-center gap-1 rounded-[3px] border border-border bg-surface/95 px-2 py-1 text-[11px] mono tnum text-muted-foreground backdrop-blur-sm">
              50.621 N · 03.987 W
            </div>
            <div className="flex flex-col items-end gap-1 rounded-[3px] border border-border bg-surface/95 backdrop-blur-sm">
              <button className="size-7 text-foreground hover:bg-muted">+</button>
              <div className="h-px w-5 self-center bg-border" />
              <button className="size-7 text-foreground hover:bg-muted">−</button>
            </div>
          </div>
        </div>

        <div className="mt-auto flex items-end justify-between p-3">
          <div className="pointer-events-auto rounded-[3px] border border-border bg-surface/95 px-3 py-2 backdrop-blur-sm">
            <div className="label-eyebrow mb-1">HIGH-risk sites highlighted</div>
            <div className="flex items-center gap-3 text-[12px] text-foreground">
              <span className="mono tnum">{highRisk.length} blocked</span>
              <span className="h-3 w-px bg-border" />
              <span className="mono tnum">15-min scan cycle</span>
            </div>
          </div>
          <div className="pointer-events-auto flex items-center gap-3 rounded-[3px] border border-border bg-surface/95 px-3 py-1.5 text-[11px] mono tnum text-muted-foreground backdrop-blur-sm">
            <span>1 : 250 000</span>
            <div className="flex items-center gap-1">
              <span className="block h-[3px] w-8 bg-foreground" />
              <span>10 km</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
