"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { getApiBaseUrl } from "../utils/api";

export type BackendStatus = "checking" | "online" | "sleeping" | "waking";

export interface BackendHealthState {
  status: BackendStatus;
  latencyMs: number | null;
  attempts: number;
  lastChecked: Date | null;
  wakeBackend: () => void;
  checkHealth: () => Promise<boolean>;
}

export function useBackendHealth(): BackendHealthState {
  const [status, setStatus] = useState<BackendStatus>("checking");
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [attempts, setAttempts] = useState<number>(0);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);
  const wakingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const checkHealth = useCallback(async (): Promise<boolean> => {
    const startTime = performance.now();
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 4000);

      const res = await fetch(`${getApiBaseUrl()}/health`, {
        method: "GET",
        signal: controller.signal,
        headers: { "Cache-Control": "no-cache" },
      });
      clearTimeout(timeoutId);

      const elapsed = Math.round(performance.now() - startTime);
      setLatencyMs(elapsed);
      setLastChecked(new Date());

      if (res.ok) {
        setStatus("online");
        setAttempts(0);
        if (wakingIntervalRef.current) {
          clearInterval(wakingIntervalRef.current);
          wakingIntervalRef.current = null;
        }
        return true;
      } else {
        setStatus("sleeping");
        return false;
      }
    } catch {
      setLatencyMs(null);
      setLastChecked(new Date());
      setStatus((prev) => (prev === "waking" ? "waking" : "sleeping"));
      return false;
    }
  }, []);

  const wakeBackend = useCallback(() => {
    setStatus("waking");
    setAttempts(1);

    if (wakingIntervalRef.current) {
      clearInterval(wakingIntervalRef.current);
    }

    // Ping immediately
    checkHealth().then((isOk) => {
      if (isOk) return;

      // Repeat ping every 3 seconds until awake
      wakingIntervalRef.current = setInterval(async () => {
        setAttempts((prev) => prev + 1);
        const ok = await checkHealth();
        if (ok && wakingIntervalRef.current) {
          clearInterval(wakingIntervalRef.current);
          wakingIntervalRef.current = null;
        }
      }, 3000);
    });
  }, [checkHealth]);

  // Initial check on mount
  useEffect(() => {
    checkHealth().then((ok) => {
      if (!ok) {
        // Auto-attempt wake up in background if asleep on first visit
        wakeBackend();
      }
    });

    return () => {
      if (wakingIntervalRef.current) {
        clearInterval(wakingIntervalRef.current);
      }
    };
  }, [checkHealth, wakeBackend]);

  return {
    status,
    latencyMs,
    attempts,
    lastChecked,
    wakeBackend,
    checkHealth,
  };
}
