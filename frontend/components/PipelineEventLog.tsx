"use client";

import React from "react";
import { PipelineStreamEvent } from "../types/pipeline";

interface PipelineEventLogProps {
  events: PipelineStreamEvent[];
  isStreaming: boolean;
}

export function PipelineEventLog({ events, isStreaming }: PipelineEventLogProps) {
  const visibleEvents = events.slice(-12).reverse();

  return (
    <section
      aria-live="polite"
      aria-label="Live pipeline events"
      className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4 shadow-xl"
    >
      <div className="mb-3 flex items-center justify-between border-b border-slate-800 pb-2">
        <h3 className="font-mono text-[11px] font-semibold uppercase tracking-wider text-slate-300">
          Live WebSocket Events
        </h3>
        <span className={`font-mono text-[10px] ${isStreaming ? "text-cyan-300" : "text-slate-500"}`}>
          {isStreaming ? "STREAMING" : `${events.length} events`}
        </span>
      </div>
      {visibleEvents.length === 0 ? (
        <p className="text-xs text-slate-500">Pipeline stage updates will appear here when a task runs.</p>
      ) : (
        <ol className="flex max-h-64 flex-col gap-2 overflow-y-auto">
          {visibleEvents.map((event, index) => (
            <li key={`${event.timestamp}-${event.stage}-${index}`} className="flex gap-2 text-xs">
              <span className="shrink-0 font-mono text-[10px] text-cyan-400">{event.stage}</span>
              <span className="text-slate-300">{event.message}</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
