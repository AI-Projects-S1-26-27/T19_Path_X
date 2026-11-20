"use client";

import { useState, useCallback, useRef } from "react";
import {
  PipelineStreamEvent,
  PipelineRunRequest,
  CounterexampleWitness,
  FaultAttribution,
  GroundActionSchema,
  RepairConstraint,
  GraphNode,
  GraphEdge,
  GraphStats,
} from "../types/pipeline";
import { getApiBaseUrl, getWebSocketUrl } from "../utils/api";

export interface PipelineAccumulatedState {
  prompt: string;
  domainId?: string;
  domainConfidence?: number;
  clarificationQuestion?: string;
  unsupportedReason?: string;
  entities: Record<string, string>;
  initialFacts: any[];
  goalFacts: any[];
  negativeConstraints: any[];
  isSchemaValid: boolean | null;
  candidatePlan: GroundActionSchema[];
  candidateCost: number;
  initialVerificationPassed: boolean | null;
  verificationFailureMessage?: string;
  counterexample?: CounterexampleWitness;
  attribution?: FaultAttribution;
  repairsApplied?: RepairConstraint[];
  repairTimeMs?: number;
  finalPlan: GroundActionSchema[];
  isFinalVerified: boolean | null;
  executionTimeMs: number;
  interpretationTimeMs?: number;
  searchTimeMs?: number;
  verificationTimeMs?: number;
  decisionTrace?: any;
  plannerAlgorithm?: string;
  requestedAlgorithm?: string;
  providerUsed?: string;
  cascadeCount?: number;
  backtrackCount?: number;
  searchBenchmark?: Array<{ algorithm: string; duration_ms: number; cost: number; nodes_expanded: number; is_winner: boolean; is_success: boolean }>;
  isFastestSearchMode?: boolean;
  planningFailed?: boolean;
  planningFailedMessage?: string;
  treeLog?: Array<{ stage: string; message: string; timestamp: number; type: "info" | "warning" | "recovery" }>;
  // Graph exploration state
  graphNodes: GraphNode[];
  graphEdges: GraphEdge[];
  graphStats?: GraphStats;
  graphStatus: "IDLE" | "ENUMERATING" | "COMPLETED" | "TRUNCATED";
}

const initialState: PipelineAccumulatedState = {
  prompt: "",
  entities: {},
  initialFacts: [],
  goalFacts: [],
  negativeConstraints: [],
  isSchemaValid: null,
  candidatePlan: [],
  candidateCost: 0,
  initialVerificationPassed: null,
  finalPlan: [],
  isFinalVerified: null,
  executionTimeMs: 0,
  plannerAlgorithm: "",
  requestedAlgorithm: "",
  providerUsed: "auto",
  cascadeCount: 0,
  backtrackCount: 0,
  planningFailed: false,
  planningFailedMessage: undefined,
  treeLog: [],
  graphNodes: [],
  graphEdges: [],
  graphStatus: "IDLE",
};

function normalizeGraphNode(n: any): GraphNode {
  const id = n.id || n.node_id || "";
  const is_boundary = Boolean(n.is_boundary || n.is_truncated_boundary || n.partially_expanded);
  return {
    id,
    node_id: id,
    depth: typeof n.depth === "number" ? n.depth : 0,
    is_initial: Boolean(n.is_initial),
    is_goal: Boolean(n.is_goal),
    is_dead_end: Boolean(n.is_dead_end),
    is_boundary,
    is_truncated_boundary: is_boundary,
    facts: Array.isArray(n.facts) ? n.facts : [],
    facts_summary: Array.isArray(n.facts_summary)
      ? n.facts_summary
      : (Array.isArray(n.facts)
          ? n.facts.slice(0, 6).map((f: any) => typeof f === "string" ? f : `${f.predicate}(${f.arguments?.join(", ")})`)
          : []),
    summary: n.summary || "",
    incoming_edges_count: n.incoming_edges_count,
    outgoing_edges_count: n.outgoing_edges_count,
  };
}

function normalizeGraphEdge(e: any): GraphEdge {
  const source = e.source || e.source_id || "";
  const target = e.target || e.target_id || "";
  const id = e.id || e.edge_id || `${source}->${target}`;
  return {
    id,
    source,
    source_id: source,
    target,
    target_id: target,
    action_name: e.action_name || "",
    action_args: Array.isArray(e.action_args) ? e.action_args : [],
    action_label: e.action_label || e.label || e.action_name || "",
    cost: typeof e.cost === "number" ? e.cost : 1.0,
    is_cycle: Boolean(e.is_cycle),
    is_plan_edge: Boolean(e.is_plan_edge),
    is_repaired_edge: Boolean(e.is_repaired_edge),
  };
}

function normalizeGraphStats(s: any): GraphStats | undefined {
  if (!s) return undefined;
  const max_depth = typeof s.max_depth === "number" ? s.max_depth : (typeof s.max_depth_reached === "number" ? s.max_depth_reached : 0);
  return {
    total_nodes: typeof s.total_nodes === "number" ? s.total_nodes : 0,
    total_edges: typeof s.total_edges === "number" ? s.total_edges : 0,
    max_depth,
    max_depth_reached: max_depth,
    goal_node_ids: Array.isArray(s.goal_node_ids) ? s.goal_node_ids : [],
    dead_end_node_ids: Array.isArray(s.dead_end_node_ids) ? s.dead_end_node_ids : [],
    is_complete: Boolean(s.is_complete),
    truncation_reason: s.truncation_reason ?? null,
    enumeration_time_ms: typeof s.enumeration_time_ms === "number" ? s.enumeration_time_ms : 0,
    duplicate_states_pruned: s.duplicate_states_pruned,
    cycles_detected: s.cycles_detected,
  };
}

export function usePipelineWebSocket(backendUrl: string = "http://127.0.0.1:8080") {
  const [isStreaming, setIsStreaming] = useState(false);
  const [events, setEvents] = useState<PipelineStreamEvent[]>([]);
  const [state, setState] = useState<PipelineAccumulatedState>(initialState);
  const [error, setError] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const receivedEventRef = useRef(false);
  const terminalEventRef = useRef(false);

  const reset = useCallback(() => {
    setEvents([]);
    setState(initialState);
    setError(null);
    setIsStreaming(false);
    receivedEventRef.current = false;
    terminalEventRef.current = false;
  }, []);

  const runPipeline = useCallback(
    (req: PipelineRunRequest) => {
      reset();
      setIsStreaming(true);
      receivedEventRef.current = false;
      terminalEventRef.current = false;

      const isFastest = !req.algorithm || req.algorithm.toLowerCase() === "fastest" || req.algorithm.toLowerCase() === "auto";
      const userAlgo = isFastest
        ? "Fastest measured of A*, BFS, and Greedy Best-First"
        : (req.algorithm.toUpperCase() === "ASTAR" || req.algorithm === "A*")
        ? "A*"
        : req.algorithm.toUpperCase() === "BFS"
        ? "BFS"
        : (req.algorithm.toUpperCase() === "BEST_FIRST" || req.algorithm.toLowerCase() === "best_first")
        ? "BEST_FIRST"
        : req.algorithm;

      setState({
        ...initialState,
        prompt: req.prompt,
        plannerAlgorithm: userAlgo,
        requestedAlgorithm: userAlgo,
        isFastestSearchMode: isFastest,
      });

      const wsUrl = getWebSocketUrl("/ws/pipeline");

      try {
        const ws = new WebSocket(wsUrl);
        socketRef.current = ws;

        ws.onopen = () => {
          ws.send(JSON.stringify(req));
        };

        ws.onmessage = (event) => {
          try {
            const data: PipelineStreamEvent = JSON.parse(event.data);
            receivedEventRef.current = true;
            setEvents((prev) => [...prev, data]);

            // Update accumulated state stage by stage
            if (data.stage === "DOMAIN_SELECTED") {
              setState((prev) => ({
                ...prev,
                domainId: data.data.domain_id,
                domainConfidence: data.data.confidence,
                treeLog: [
                  ...(prev.treeLog || []),
                  { stage: "DOMAIN", message: data.message, timestamp: data.timestamp || Date.now(), type: "info" },
                ],
              }));
            } else if (data.stage === "CLARIFICATION_REQUIRED") {
              setState((prev) => ({
                ...prev,
                clarificationQuestion: data.data.question || data.message,
                unsupportedReason: undefined,
                treeLog: [
                  ...(prev.treeLog || []),
                  { stage: "CLARIFICATION", message: data.message, timestamp: data.timestamp || Date.now(), type: "warning" },
                ],
              }));
              terminalEventRef.current = true;
              setIsStreaming(false);
              ws.close();
            } else if (data.stage === "TASK_UNSUPPORTED") {
              setState((prev) => ({
                ...prev,
                unsupportedReason: data.data.explanation || data.message,
                clarificationQuestion: undefined,
                treeLog: [
                  ...(prev.treeLog || []),
                  { stage: "UNSUPPORTED", message: data.message, timestamp: data.timestamp || Date.now(), type: "warning" },
                ],
              }));
              terminalEventRef.current = true;
              setIsStreaming(false);
              ws.close();
            } else if (data.stage === "GRAPH_ENUMERATION_STARTED") {
              setState((prev) => ({
                ...prev,
                graphStatus: "ENUMERATING",
                graphNodes: [],
                graphEdges: [],
                graphStats: undefined,
              }));
            } else if (data.stage === "GRAPH_NODE_BATCH") {
              const rawNodes = data.data.nodes || [];
              const normalized = rawNodes.map(normalizeGraphNode);
              setState((prev) => ({
                ...prev,
                graphNodes: [...prev.graphNodes, ...normalized],
              }));
            } else if (data.stage === "GRAPH_EDGE_BATCH") {
              const rawEdges = data.data.edges || [];
              const normalized = rawEdges.map(normalizeGraphEdge);
              setState((prev) => ({
                ...prev,
                graphEdges: [...prev.graphEdges, ...normalized],
              }));
            } else if (data.stage === "GRAPH_ENUMERATION_COMPLETED") {
              setState((prev) => ({
                ...prev,
                graphStats: normalizeGraphStats(data.data.stats),
                graphStatus: "COMPLETED",
              }));
            } else if (data.stage === "GRAPH_ENUMERATION_TRUNCATED") {
              setState((prev) => ({
                ...prev,
                graphStats: normalizeGraphStats(data.data.stats),
                graphStatus: "TRUNCATED",
              }));
            } else if (data.stage === "BACKTRACK") {
              setState((prev) => ({
                ...prev,
                backtrackCount: (prev.backtrackCount || 0) + 1,
                treeLog: [
                  ...(prev.treeLog || []),
                  { stage: "BACKTRACK", message: data.message, timestamp: data.timestamp || Date.now(), type: "recovery" },
                ],
              }));
            } else if (data.stage === "PLANNER_CASCADE") {
              setState((prev) => ({
                ...prev,
                plannerAlgorithm: data.data.algorithm || prev.plannerAlgorithm,
                cascadeCount: (prev.cascadeCount || 0) + 1,
                treeLog: [
                  ...(prev.treeLog || []),
                  { stage: "CASCADE", message: data.message, timestamp: data.timestamp || Date.now(), type: "recovery" },
                ],
              }));
            } else if (data.stage === "PLANNING_FAILED") {
              setState((prev) => ({
                ...prev,
                planningFailed: true,
                planningFailedMessage: data.message || "All planners exhausted. No valid plan found.",
                searchBenchmark: data.data?.search_benchmark || prev.searchBenchmark,
                treeLog: [
                  ...(prev.treeLog || []),
                  { stage: "PLANNING_FAILED", message: data.message, timestamp: data.timestamp || Date.now(), type: "warning" },
                ],
              }));
              if (data.data?.will_retry === false) {
                terminalEventRef.current = true;
                setIsStreaming(false);
              }
            } else if (data.stage === "VALIDATION_FAILED") {
              setState((prev) => ({
                ...prev,
                isSchemaValid: false,
                treeLog: [
                  ...(prev.treeLog || []),
                  { stage: "VALIDATION_FAILED", message: data.message, timestamp: data.timestamp || Date.now(), type: "warning" },
                ],
              }));
            } else if (data.stage === "INTERPRETATION_COMPLETED") {
              setState((prev) => ({
                ...prev,
                entities: data.data.entities || {},
                goalFacts: data.data.goal_facts || [],
                initialFacts: data.data.initial_facts || [],
                negativeConstraints: data.data.negative_constraints || [],
                isSchemaValid: data.data.is_valid,
                interpretationTimeMs: data.data.stage_duration_ms,
                providerUsed: data.data.provider_used || prev.providerUsed,
              }));
            } else if (data.stage === "SEARCH_COMPLETED") {
              setState((prev) => ({
                ...prev,
                candidatePlan: data.data.actions || [],
                candidateCost: data.data.cost || 0,
                searchTimeMs: data.data.stage_duration_ms,
                plannerAlgorithm: data.data.algorithm || prev.plannerAlgorithm,
                searchBenchmark: data.data.search_benchmark,
                isFastestSearchMode: Boolean(data.data.is_fastest_mode ?? prev.isFastestSearchMode),
              }));
            } else if (data.stage === "VERIFICATION_RESULT") {
              setState((prev) => ({
                ...prev,
                initialVerificationPassed: data.data.is_valid,
                verificationTimeMs: data.data.stage_duration_ms,
              }));
            } else if (data.stage === "VERIFICATION_FAILED") {
              setState((prev) => ({
                ...prev,
                initialVerificationPassed: false,
                verificationFailureMessage: data.data.explanation,
                verificationTimeMs: data.data.stage_duration_ms,
              }));
            } else if (data.stage === "COUNTEREXAMPLE_EXTRACTED") {
              setState((prev) => ({
                ...prev,
                counterexample: data.data.witness,
              }));
            } else if (data.stage === "ATTRIBUTION_CLASSIFIED") {
              setState((prev) => ({
                ...prev,
                attribution: {
                  fault_class: data.data.fault_class,
                  affected_stage: data.data.affected_stage,
                  confidence: data.data.confidence,
                },
              }));
            } else if (data.stage === "REPAIR_INJECTED") {
              setState((prev) => ({
                ...prev,
                repairsApplied: data.data.repairs_applied || [],
                repairTimeMs: data.data.stage_duration_ms,
              }));
            } else if (data.stage === "FINAL_VERIFICATION") {
              setState((prev) => ({
                ...prev,
                finalPlan: data.data.final_plan || [],
                isFinalVerified: data.data.is_valid,
                executionTimeMs: data.data.execution_time_ms || 0,
                decisionTrace: data.data.decision_trace || prev.decisionTrace,
              }));
            } else if (data.stage === "PIPELINE_FINISHED" || data.stage === "ERROR") {
              if (data.stage === "ERROR") {
                setError(data.message || "Pipeline execution error occurred.");
                setState((prev) => ({
                  ...prev,
                  planningFailed: prev.candidatePlan.length === 0 ? true : prev.planningFailed,
                  planningFailedMessage: prev.candidatePlan.length === 0
                    ? (data.message || "No valid path found to reach the goal under active constraints.")
                    : prev.planningFailedMessage,
                  treeLog: [
                    ...(prev.treeLog || []),
                    { stage: "ERROR", message: data.message, timestamp: data.timestamp || Date.now(), type: "warning" },
                  ],
                }));
              }
              terminalEventRef.current = true;
              setIsStreaming(false);
              ws.close();
            }
          } catch (e: any) {
            console.error("Failed to parse WebSocket message:", e);
          }
        };

        ws.onerror = async () => {
          console.warn("WebSocket error; falling back to REST endpoint /api/pipeline/run");
          if (receivedEventRef.current) {
            setError("The WebSocket stream disconnected before the pipeline finished. Check the live event log for the last completed stage.");
            setIsStreaming(false);
            ws.close();
            return;
          }
          // Fallback to REST API
          try {
            const resp = await fetch(`${getApiBaseUrl()}/api/pipeline/run`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                prompt: req.prompt,
                algorithm: req.algorithm,
                force_invalid_first_candidate: req.force_fault,
                provider_type: req.provider_type,
              }),
            });
            if (!resp.ok) throw new Error(await resp.text());
            const json = await resp.json();

            setState({
              prompt: req.prompt,
              entities: json.interpretation?.entities || {},
              initialFacts: json.interpretation?.initial_facts || [],
              goalFacts: json.interpretation?.goal_facts || [],
              negativeConstraints: json.interpretation?.negative_constraints || [],
              isSchemaValid: json.validation?.is_valid ?? true,
              candidatePlan: json.candidate_plan?.actions || [],
              candidateCost: json.candidate_plan?.total_cost || 0,
              initialVerificationPassed: json.initial_verification?.is_valid,
              verificationFailureMessage: json.initial_verification?.explanation,
              counterexample: json.counterexample,
              attribution: json.fault_attribution,
              repairsApplied: json.repairs_applied,
              finalPlan: json.final_plan?.actions || [],
              isFinalVerified: json.final_verification?.is_valid,
              executionTimeMs: json.total_execution_time_ms || 0,
              graphNodes: [],
              graphEdges: [],
              graphStatus: "IDLE",
            });
          } catch (err: any) {
            setError(err.message || "Failed to execute pipeline via REST fallback");
          } finally {
            setIsStreaming(false);
          }
        };

        ws.onclose = () => {
          if (receivedEventRef.current && !terminalEventRef.current) {
            setError("The WebSocket closed before the pipeline sent a terminal event.");
            setIsStreaming(false);
          }
        };
      } catch (err: any) {
        setError(err.message || "WebSocket initialization error");
        setIsStreaming(false);
      }
    },
    [backendUrl, reset]
  );

  return {
    isStreaming,
    events,
    state,
    error,
    runPipeline,
    reset,
  };
}
