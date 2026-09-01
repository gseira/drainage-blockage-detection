import type { Risk, AssetStatus } from "@/lib/dashboard-data";

const riskTone: Record<Risk, { dot: string; text: string }> = {
  high:   { dot: "bg-[var(--color-risk-high)]",   text: "text-[var(--color-risk-high)]" },
  medium: { dot: "bg-[var(--color-risk-medium)]", text: "text-[var(--color-risk-medium)]" },
  low:    { dot: "bg-[var(--color-risk-low)]",    text: "text-[var(--color-risk-low)]" },
};

export function RiskBadge({ risk, compact = false }: { risk: Risk; compact?: boolean }) {
  const t = riskTone[risk];
  const label = risk[0].toUpperCase() + risk.slice(1);
  return (
    <span className="inline-flex items-center gap-1.5 tnum">
      <span className={`size-1.5 rounded-full ${t.dot}`} aria-hidden />
      <span className={compact ? "text-[12px] font-medium" : `text-[12px] font-medium ${t.text}`}>
        {label}
      </span>
    </span>
  );
}

const statusTone: Record<AssetStatus, string> = {
  blocked:    "text-[var(--color-risk-high)]",
  monitoring: "text-[var(--color-risk-medium)]",
  clear:      "text-[var(--color-ok)]",
  offline:    "text-muted-foreground",
};

export function StatusDot({ status, small = false }: { status: AssetStatus; small?: boolean }) {
  const dot = (
    <span
      className={`rounded-full shrink-0 ${small ? "size-1.5" : "size-1.5"} ${
        status === "blocked"    ? "bg-[var(--color-risk-high)]" :
        status === "monitoring" ? "bg-[var(--color-risk-medium)]" :
        status === "clear"      ? "bg-[var(--color-ok)]" :
                                  "bg-muted-foreground/50"
      }`}
      aria-hidden
    />
  );
  if (small) return dot;
  return (
    <span className="inline-flex items-center gap-1.5">
      {dot}
      <span className={`text-[12px] capitalize ${statusTone[status]}`}>{status}</span>
    </span>
  );
}

export function KeyValue({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1.5">
      <span className="label-eyebrow">{label}</span>
      <span className="text-[13px] text-foreground tnum">{children}</span>
    </div>
  );
}

export function Meter({ value, tone = "neutral" }: { value: number; tone?: "neutral" | Risk }) {
  const colour =
    tone === "high"   ? "var(--color-risk-high)" :
    tone === "medium" ? "var(--color-risk-medium)" :
    tone === "low"    ? "var(--color-risk-low)" :
                        "var(--color-foreground)";
  return (
    <div className="h-[3px] w-full bg-border overflow-hidden rounded-[1px]">
      <div
        className="h-full transition-[width] duration-200"
        style={{ width: `${Math.max(0, Math.min(100, value))}%`, background: colour }}
      />
    </div>
  );
}
