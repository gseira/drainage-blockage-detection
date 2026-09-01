import { Link, useRouterState } from "@tanstack/react-router";
import type { ReactNode } from "react";

const NAV = [
  {
    to: "/",
    label: "Live Monitor",
    icon: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
        <rect x="1" y="1" width="14" height="10" rx="1.5" stroke="currentColor" strokeWidth="1.4"/>
        <path d="M5 14h6M8 11v3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
        <circle cx="8" cy="6" r="2" stroke="currentColor" strokeWidth="1.3"/>
      </svg>
    ),
  },
  {
    to: "/upload",
    label: "Analyse Image",
    icon: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
        <rect x="1" y="3" width="14" height="10" rx="1.5" stroke="currentColor" strokeWidth="1.4"/>
        <path d="M8 7v4M6 9l2-2 2 2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
        <path d="M5 3V2.5a1 1 0 011-1h4a1 1 0 011 1V3" stroke="currentColor" strokeWidth="1.3"/>
      </svg>
    ),
  },
] as const;

export function AppShell({ children, camCount = 0, state = "loading" }: {
  children: ReactNode;
  camCount?: number;
  state?: "loading" | "live" | "partial" | "offline";
}) {
  return (
    <div className="flex h-screen min-h-screen flex-col bg-background text-foreground">
      <TopBar camCount={camCount} state={state} />
      <div className="flex min-h-0 flex-1 flex-col">{children}</div>
    </div>
  );
}

function TopBar({ camCount, state }: { camCount: number; state: string }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const isActive = (to: string) => (to === "/" ? pathname === "/" : pathname.startsWith(to));

  const now     = new Date();
  const timeStr = now.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
  const dateStr = now.toLocaleDateString("en-GB", { day: "2-digit", month: "short" });

  return (
    <header
      className="flex shrink-0 items-stretch justify-between bg-surface"
      style={{ borderBottom: "1px solid var(--color-border)", height: 62 }}
    >
      {/* Nav tabs — centred */}
      <nav style={{ display: "flex", alignItems: "stretch", flex: 1, justifyContent: "center", gap: 48 }}>
        {NAV.map((item) => {
          const active = isActive(item.to);
          return (
            <Link
              key={item.to}
              to={item.to}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "0 44px",
                fontSize: 15,
                fontWeight: active ? 700 : 500,
                letterSpacing: "0.01em",
                color: active ? "var(--color-foreground)" : "var(--color-muted-foreground)",
                background: active
                  ? "color-mix(in oklch, var(--color-foreground) 7%, transparent)"
                  : "transparent",
                borderBottom: active
                  ? "3px solid var(--color-foreground)"
                  : "3px solid transparent",
                transition: "all 0.15s",
                textDecoration: "none",
                whiteSpace: "nowrap",
              }}
            >
              <span style={{ opacity: active ? 1 : 0.5 }}>{item.icon}</span>
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Right: status + time */}
      <div
        className="flex items-center gap-3 px-5"
        style={{ borderLeft: "1px solid var(--color-border)", fontSize: 12, color: "var(--color-muted-foreground)" }}
      >
        <span className="flex items-center gap-1.5">
          <span
            className={state === "loading" ? "animate-pulse" : ""}
            style={{
              width: 7, height: 7, borderRadius: "50%",
              background:
                state === "live"    ? "var(--color-ok)" :
                state === "partial" ? "var(--color-risk-medium)" :
                state === "offline" ? "var(--color-risk-high)" :
                                      "var(--color-risk-medium)",
              display: "inline-block",
            }}
          />
          {state === "live"    ? `${camCount} cameras live` :
           state === "partial" ? `${camCount} cameras` :
           state === "offline" ? "Backend offline" : "Connecting…"}
        </span>
        <span style={{ width: 1, height: 14, background: "var(--color-border)", display: "inline-block" }} />
        <span>{dateStr} · {timeStr}</span>
      </div>
    </header>
  );
}

export function PageHeader({
  title, subtitle, right,
}: {
  title: string;
  subtitle?: string;
  right?: ReactNode;
}) {
  return (
    <div className="flex shrink-0 items-end justify-between border-b border-border bg-surface px-4 pb-3 pt-3.5">
      <div>
        <h1 className="text-[17px] font-semibold leading-tight tracking-tight">{title}</h1>
        {subtitle && <p className="mt-0.5 text-[12px] text-muted-foreground">{subtitle}</p>}
      </div>
      {right && <div className="flex items-center gap-2">{right}</div>}
    </div>
  );
}
