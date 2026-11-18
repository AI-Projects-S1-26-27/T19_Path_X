"use client";

import React from "react";
import { ShieldCheck, Cpu, Terminal, Sparkles, Network, Grid, Power, RefreshCw } from "lucide-react";

export type NavTab = "demo" | "graph" | "domains" | "benchmark" | "architecture";

interface NavbarProps {
  activeTab: NavTab;
  setActiveTab: (tab: NavTab) => void;
  isStreaming: boolean;
  onOpenDefenseModal?: () => void;
  backendHealth?: {
    status: "checking" | "online" | "sleeping" | "waking";
    latencyMs: number | null;
    attempts: number;
    wakeBackend: () => void;
  };
}

export function Navbar({ activeTab, setActiveTab, isStreaming, backendHealth }: NavbarProps) {
  return (
    <header className="sticky top-0 z-50 border-b border-slate-800 bg-slate-950/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-500 shadow-lg shadow-cyan-500/20">
            <ShieldCheck className="h-6 w-6 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-lg font-bold tracking-tight text-white">RE-PLAN-V</span>
              <span className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 font-mono text-[10px] font-semibold text-cyan-400">
                MULTI-DOMAIN CORE
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Generic Planning &bull; State-Space Graph Exploration &bull; Counterexample Repair
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1 rounded-xl border border-slate-800 bg-slate-900/60 p-1">
          <button
            onClick={() => setActiveTab("demo")}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
              activeTab === "demo"
                ? "bg-cyan-500/20 text-cyan-300 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Cpu className="h-3.5 w-3.5" />
            Interactive Planner
          </button>
          <button
            onClick={() => setActiveTab("graph")}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
              activeTab === "graph"
                ? "bg-cyan-500/20 text-cyan-300 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Network className="h-3.5 w-3.5" />
            State Space Explorer
          </button>
          <button
            onClick={() => setActiveTab("domains")}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
              activeTab === "domains"
                ? "bg-cyan-500/20 text-cyan-300 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Grid className="h-3.5 w-3.5" />
            Domain Studio
          </button>
          <button
            onClick={() => setActiveTab("benchmark")}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
              activeTab === "benchmark"
                ? "bg-cyan-500/20 text-cyan-300 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Terminal className="h-3.5 w-3.5" />
            Research Benchmark
          </button>
          <button
            onClick={() => setActiveTab("architecture")}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all ${
              activeTab === "architecture"
                ? "bg-cyan-500/20 text-cyan-300 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Sparkles className="h-3.5 w-3.5" />
            10 Inviolable Rules
          </button>
        </nav>

        {/* Live Status & Backend Wake Controller */}
        <div className="flex items-center gap-2">
          {backendHealth && (
            <>
              {backendHealth.status === "online" && (
                <div
                  title="Backend is awake and authoritative STRIPS core is ready"
                  className="flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-950/40 px-3 py-1 text-xs"
                >
                  <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="font-mono text-[11px] text-emerald-300">
                    CORE ONLINE {backendHealth.latencyMs ? `(${backendHealth.latencyMs}ms)` : ""}
                  </span>
                </div>
              )}

              {backendHealth.status === "waking" && (
                <div
                  title="Render free-tier service is spinning up. Stand by..."
                  className="flex items-center gap-1.5 rounded-full border border-amber-500/40 bg-amber-950/50 px-3 py-1 text-xs"
                >
                  <RefreshCw className="h-3 w-3 text-amber-400 animate-spin" />
                  <span className="font-mono text-[11px] text-amber-300">
                    WAKING SERVER {backendHealth.attempts > 0 ? `(#${backendHealth.attempts})` : ""}
                  </span>
                </div>
              )}

              {(backendHealth.status === "sleeping" || backendHealth.status === "checking") && (
                <button
                  type="button"
                  onClick={backendHealth.wakeBackend}
                  title="Backend is sleeping (Render Free Tier). Click to wake it up."
                  className="flex items-center gap-1.5 rounded-full border border-rose-500/40 bg-rose-950/50 px-3 py-1 text-xs font-semibold text-rose-300 hover:bg-rose-900/60 transition shadow-sm"
                >
                  <Power className="h-3 w-3 text-rose-400 animate-pulse" />
                  <span>WAKE BACKEND</span>
                </button>
              )}
            </>
          )}

          <div
            title="Authoritative STRIPS verifier, forward chaining, and CEGIS repair engine active"
            className="hidden sm:flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900 px-3 py-1 text-xs"
          >
            <span
              className={`h-2 w-2 rounded-full ${
                isStreaming ? "animate-ping bg-cyan-400" : "bg-cyan-500"
              }`}
            />
            <span className="font-mono text-[11px] text-slate-300">
              {isStreaming ? "STREAMING..." : "6 DOMAINS"}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
