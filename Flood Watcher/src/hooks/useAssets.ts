/**
 * useAssets.ts
 * ------------
 * Fetches the camera list and latest results independently.
 * Cameras are ALWAYS shown (from /webcams, or static fallback).
 * Results from /results/latest are layered on top when available.
 * Polls every 60 seconds.
 *
 * state:
 *   "loading"   — first fetch in flight
 *   "live"      — /webcams reachable, cameras + results loaded
 *   "partial"   — /webcams reachable but /results/latest empty/failed
 *   "offline"   — backend completely unreachable, using static fallback
 */

import { useEffect, useRef, useState } from "react";
import { fetchWebcams, fetchLatestResults } from "@/lib/api";
import { toAsset, STATIC_ASSETS, type Asset } from "@/lib/dashboard-data";
import type { ResultRow, WebcamDef } from "@/lib/api";

export type DataState = "loading" | "live" | "partial" | "offline";

interface UseAssetsReturn {
  assets: Asset[];
  state: DataState;
  lastUpdated: Date | null;
  refresh: () => void;
}

const POLL_MS = 60_000;
const TIMEOUT_MS = 8_000;

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return Promise.race([
    promise,
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(`Request timed out after ${ms}ms`)), ms)
    ),
  ]);
}

export function useAssets(): UseAssetsReturn {
  const [assets, setAssets]           = useState<Asset[]>(STATIC_ASSETS);
  const [state, setState]             = useState<DataState>("loading");
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const timerRef                      = useRef<ReturnType<typeof setTimeout> | null>(null);

  const load = async () => {
    // 1. Try to get camera list (fast, always works if backend up)
    let cams: WebcamDef[] | null = null;
    try {
      cams = await withTimeout(fetchWebcams(), TIMEOUT_MS);
    } catch (err) {
      console.warn("[useAssets] /webcams unreachable:", err);
      setState("offline");
      // Keep static assets already in state
      return;
    }

    // 2. Try to get latest results (optional — may be empty on first run)
    let results: ResultRow[] = [];
    let hasResults = false;
    try {
      results = await withTimeout(fetchLatestResults(), TIMEOUT_MS);
      hasResults = results.length > 0;
    } catch (err) {
      console.warn("[useAssets] /results/latest failed:", err);
    }

    // 3. Merge
    const byPath = new Map<string, ResultRow>();
    for (const r of results) byPath.set(r.campath, r);

    const merged: Asset[] = cams.map((cam) =>
      toAsset(cam, byPath.get(cam.campath) ?? null)
    );

    setAssets(merged);
    setState(hasResults ? "live" : "partial");
    setLastUpdated(new Date());
  };

  const refresh = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    load().finally(() => {
      timerRef.current = setTimeout(refresh, POLL_MS);
    });
  };

  useEffect(() => {
    refresh();
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { assets, state, lastUpdated, refresh };
}
