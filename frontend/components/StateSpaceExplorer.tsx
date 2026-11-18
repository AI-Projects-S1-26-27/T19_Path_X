"use client";

import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { GraphNode, GraphEdge, GraphStats } from "../types/pipeline";

export type SearchAlgorithmType = "astar" | "bfs" | "greedy";

export interface SearchSimulationStep {
  stepIndex: number;
  currentNodeId: string;
  frontierNodeIds: string[];
  closedNodeIds: string[];
  solutionPathNodeIds: string[];
  solutionEdgeKeys: Set<string>;
  g: number;
  h: number;
  f: number;
  explanation: string;
  queueSnapshot: Array<{ id: string; g: number; h: number; f: number }>;
}

interface StateSpaceExplorerProps {
  streamedNodes?: GraphNode[];
  streamedEdges?: GraphEdge[];
  streamedStats?: GraphStats;
  streamedStatus?: "IDLE" | "ENUMERATING" | "COMPLETED" | "TRUNCATED";
}

export function StateSpaceExplorer({
  streamedNodes = [],
  streamedEdges = [],
  streamedStats,
  streamedStatus = "IDLE",
}: StateSpaceExplorerProps) {
  // ponytail: standalone explorer removed — stream-only

  // Filter & Search
  const [searchFilter, setSearchFilter] = useState<string>("");
  const [filterType, setFilterType] = useState<"all" | "goal" | "dead_end" | "boundary">("all");

  // Selection & Hover (no scale transform, stable tracking)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [hoveredEdgeKey, setHoveredEdgeKey] = useState<string | null>(null);

  // Zoom & Pan atomic viewport state with focal-point cursor zooming
  const MIN_ZOOM = 0.2;
  const MAX_ZOOM = 3.5;
  const [{ zoom, pan }, setViewport] = useState<{ zoom: number; pan: { x: number; y: number } }>({
    zoom: 0.9,
    pan: { x: 0, y: 0 },
  });
  const viewportRef = useRef<{ zoom: number; pan: { x: number; y: number } }>({
    zoom: 0.9,
    pan: { x: 0, y: 0 },
  });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const dragStartRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const hasUserPanned = useRef<boolean>(false);

  // Search Algorithm Simulation State
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<SearchAlgorithmType>("astar");
  const [currentSimStepIndex, setCurrentSimStepIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(800); // ms per step

  // Always use stream data directly
  const rawNodes = streamedNodes;
  const rawEdges = streamedEdges;
  const currentStats = streamedStats;
  const currentStatus = streamedStatus;

  // Unified Canonical State Normalization:
  // Guarantees that whether nodes and edges originate from WebSocket streaming (Mission Control)
  // or standalone REST enumeration, ALL state IDs are mapped to clean, human-readable labels:
  // s0 (Initial State), s1, s2, s3, etc. Zero hashes will ever appear in the UI.
  const { currentNodes, currentEdges } = useMemo(() => {
    if (rawNodes.length === 0) {
      return { currentNodes: [], currentEdges: [] };
    }

    // Sort: Initial state is guaranteed to be index 0 (s0), then by depth
    const sorted = [...rawNodes].sort((a, b) => {
      if (a.is_initial && !b.is_initial) return -1;
      if (!a.is_initial && b.is_initial) return 1;
      return (a.depth ?? 0) - (b.depth ?? 0);
    });

    // Create bidirectional ID lookup table
    const idMap = new Map<string, string>();
    sorted.forEach((node, idx) => {
      const cleanId = `s${idx}`;
      idMap.set(node.id, cleanId);
      if ((node as any).node_id) idMap.set((node as any).node_id, cleanId);
      if (node.state_hash) idMap.set(node.state_hash, cleanId);
      if (node.id.startsWith("state_")) {
        idMap.set(node.id.replace("state_", ""), cleanId);
      }
    });

    const normalizedNodes: GraphNode[] = sorted.map((node, idx) => {
      const cleanId = idMap.get(node.id) || `s${idx}`;
      return {
        ...node,
        id: cleanId,
        node_id: cleanId,
      };
    });

    const normalizedEdges: GraphEdge[] = rawEdges.map((edge) => {
      const srcRaw = edge.source || (edge as any).source_id || "";
      const tgtRaw = edge.target || (edge as any).target_id || "";
      const srcClean = idMap.get(srcRaw) || (srcRaw.startsWith("state_") ? idMap.get(srcRaw.replace("state_", "")) : undefined) || srcRaw.replace("state_", "s");
      const tgtClean = idMap.get(tgtRaw) || (tgtRaw.startsWith("state_") ? idMap.get(tgtRaw.replace("state_", "")) : undefined) || tgtRaw.replace("state_", "s");
      return {
        ...edge,
        source: srcClean,
        source_id: srcClean,
        target: tgtClean,
        target_id: tgtClean,
      };
    });

    return { currentNodes: normalizedNodes, currentEdges: normalizedEdges };
  }, [rawNodes, rawEdges]);




  // Node position calculation: Stratified Hierarchical Layout centered symmetrically
  const nodePositions = useMemo(() => {
    const positions: Record<string, { x: number; y: number }> = {};
    if (currentNodes.length === 0) return positions;

    // Group by depth
    const levels: Record<number, GraphNode[]> = {};
    let maxDepth = 0;
    for (const node of currentNodes) {
      const d = node.depth ?? 0;
      if (!levels[d]) levels[d] = [];
      levels[d].push(node);
      if (d > maxDepth) maxDepth = d;
    }

    const ySpacing = 120;
    const minXSpacing = 110;

    // Find the maximum node count across all depth levels to define a uniform layout width
    let maxLevelCount = 1;
    for (let d = 0; d <= maxDepth; d++) {
      const count = (levels[d] || []).length;
      if (count > maxLevelCount) maxLevelCount = count;
    }
    const layoutWidth = Math.max(800, (maxLevelCount + 1) * minXSpacing);

    for (let d = 0; d <= maxDepth; d++) {
      const nodesAtLevel = levels[d] || [];
      const count = nodesAtLevel.length;
      if (count === 0) continue;

      const step = layoutWidth / (count + 1);

      nodesAtLevel.forEach((node, idx) => {
        positions[node.id] = {
          x: Math.round((idx + 1) * step),
          y: Math.round(70 + d * ySpacing),
        };
      });
    }

    return positions;
  }, [currentNodes]);

  // Compute graph bounds and center
  const graphBounds = useMemo(() => {
    const nodeKeys = Object.keys(nodePositions);
    if (nodeKeys.length === 0) {
      return { minX: 0, maxX: 800, minY: 0, maxY: 600, centerX: 400, centerY: 300, width: 800, height: 600 };
    }
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    for (const key of nodeKeys) {
      const pos = nodePositions[key];
      if (pos.x < minX) minX = pos.x;
      if (pos.x > maxX) maxX = pos.x;
      if (pos.y < minY) minY = pos.y;
      if (pos.y > maxY) maxY = pos.y;
    }

    return {
      minX,
      maxX,
      minY,
      maxY,
      centerX: (minX + maxX) / 2,
      centerY: (minY + maxY) / 2,
      width: Math.max(10, maxX - minX),
      height: Math.max(10, maxY - minY),
    };
  }, [nodePositions]);

  // Centering pan calculation: positions the graph in the exact center of the SVG display
  const getCenteredPan = useCallback((targetZoom: number = 0.9) => {
    const el = svgRef.current || containerRef.current;
    const viewW = el && el.clientWidth > 0 ? el.clientWidth : 850;
    const viewH = el && el.clientHeight > 0 ? el.clientHeight : 580;

    const targetX = Math.round(viewW / 2 - graphBounds.centerX * targetZoom);
    const scaledGraphH = graphBounds.height * targetZoom;

    let targetY: number;
    if (scaledGraphH <= viewH - 140) {
      targetY = Math.round(viewH / 2 - graphBounds.centerY * targetZoom);
    } else {
      targetY = Math.round(65 - graphBounds.minY * targetZoom);
    }

    return { x: targetX, y: targetY };
  }, [graphBounds]);

  // Helper to format any state ID into a clean friendly identifier (e.g., s0, s1, etc.)
  // and guarantee that no raw hexadecimal hash strings are ever rendered.
  const formatStateId = (id: string | null | undefined): string => {
    if (!id) return "";
    if (/^s\d+$/i.test(id)) return id.toLowerCase();
    const clean = id.replace(/^(state_|node_|s_)/i, "");
    if (/^\d+$/.test(clean)) return `s${clean}`;
    const idx = currentNodes.findIndex((n) => n.id === id || n.id === clean || (n as any).node_id === id);
    if (idx >= 0) return `s${idx}`;
    if (clean.length > 6) {
      let hash = 0;
      for (let i = 0; i < clean.length; i++) hash = (hash * 31 + clean.charCodeAt(i)) >>> 0;
      return `s${hash % 30}`;
    }
    return `s${clean}`;
  };

  // Selected node details
  const selectedNode = useMemo(() => {
    return currentNodes.find((n) => n.id === selectedNodeId) || null;
  }, [currentNodes, selectedNodeId]);

  // Helper to format full action name with arguments (e.g., stack(blue_box, glass))
  const formatActionCall = useCallback((edge: GraphEdge | null | undefined): string => {
    if (!edge) return "";
    if (edge.label && edge.label.includes("(")) return edge.label;
    if (edge.action_args && edge.action_args.length > 0) {
      return `${edge.action_name}(${edge.action_args.join(", ")})`;
    }
    return edge.action_name;
  }, []);

  // Filter edges to strictly forward search tree transitions (target depth > source depth)
  const forwardEdges = useMemo(() => {
    return currentEdges.filter((edge) => {
      const sNode = currentNodes.find((n) => n.id === edge.source);
      const tNode = currentNodes.find((n) => n.id === edge.target);
      if (!sNode || !tNode) return false;
      return (tNode.depth ?? 0) > (sNode.depth ?? 0);
    });
  }, [currentEdges, currentNodes]);

  // Trace depth-first / search derivation trajectory from Root (s0) to Selected Node
  const selectedTrajectory = useMemo(() => {
    if (!selectedNodeId || !selectedNode) {
      return {
        pathEdges: [] as GraphEdge[],
        edgeKeyToStep: new Map<string, { stepNumber: number; actionText: string }>(),
        pathNodeIds: [] as string[],
      };
    }

    if ((selectedNode.depth ?? 0) === 0 || selectedNode.is_initial) {
      return {
        pathEdges: [] as GraphEdge[],
        edgeKeyToStep: new Map<string, { stepNumber: number; actionText: string }>(),
        pathNodeIds: [selectedNode.id],
      };
    }

    const pathEdges: GraphEdge[] = [];
    const pathNodeIds: string[] = [selectedNodeId];
    let currId = selectedNodeId;
    const visited = new Set<string>();

    while (currId && !visited.has(currId)) {
      visited.add(currId);
      const curr = currentNodes.find((n) => n.id === currId);
      if (!curr || (curr.depth ?? 0) === 0 || curr.is_initial) break;

      // Find arrival forward edge where source has depth < curr.depth
      // Prefer parent with depth === curr.depth - 1
      const candidates = forwardEdges.filter((e) => e.target === currId);
      let bestEdge: GraphEdge | undefined = candidates.find((e) => {
        const s = currentNodes.find((n) => n.id === e.source);
        return s && (s.depth ?? 0) === (curr.depth ?? 0) - 1;
      });

      if (!bestEdge && candidates.length > 0) {
        bestEdge = candidates[0];
      }

      if (!bestEdge) break;

      pathEdges.unshift(bestEdge);
      pathNodeIds.unshift(bestEdge.source);
      currId = bestEdge.source;
    }

    const edgeKeyToStep = new Map<string, { stepNumber: number; actionText: string }>();
    pathEdges.forEach((edge, idx) => {
      const key = `${edge.source}->${edge.target}`;
      edgeKeyToStep.set(key, {
        stepNumber: idx + 1,
        actionText: formatActionCall(edge),
      });
    });

    return { pathEdges, edgeKeyToStep, pathNodeIds };
  }, [selectedNodeId, selectedNode, currentNodes, forwardEdges, formatActionCall]);

  // Transition details for selected node
  const { parentTransition, parentSourceNode, forwardActions, reverseActions } = useMemo(() => {
    if (!selectedNodeId || !selectedNode) {
      return { parentTransition: null, parentSourceNode: null, forwardActions: [], reverseActions: [] };
    }

    // Find parent edge (edge that led to this node from a shallower depth)
    const pEdge = currentEdges.find((e) => {
      if (e.target !== selectedNodeId) return false;
      const sNode = currentNodes.find((n) => n.id === e.source);
      return sNode && (sNode.depth ?? 0) < (selectedNode.depth ?? 0);
    }) || null;

    const pSource = pEdge ? (currentNodes.find((n) => n.id === pEdge.source) || null) : null;

    // Outgoing edges partitioned into forward progress vs reverse cycles
    const outgoing = currentEdges.filter((e) => e.source === selectedNodeId);
    const fwd: GraphEdge[] = [];
    const rev: GraphEdge[] = [];

    outgoing.forEach((e) => {
      const tNode = currentNodes.find((n) => n.id === e.target);
      if (tNode && (tNode.depth ?? 0) > (selectedNode.depth ?? 0)) {
        fwd.push(e);
      } else {
        rev.push(e);
      }
    });

    return {
      parentTransition: pEdge,
      parentSourceNode: pSource,
      forwardActions: fwd,
      reverseActions: rev,
    };
  }, [selectedNodeId, selectedNode, currentEdges, currentNodes]);

  // Filtered nodes
  const filteredNodeIds = useMemo(() => {
    const ids = new Set<string>();
    const query = searchFilter.toLowerCase().trim();

    for (const node of currentNodes) {
      if (filterType === "goal" && !node.is_goal) continue;
      if (filterType === "dead_end" && !node.is_dead_end) continue;
      if (filterType === "boundary" && !node.is_truncated_boundary) continue;

      if (query) {
        const matchesId = node.id.toLowerCase().includes(query);
        const matchesFacts = (node.facts_summary || []).some((f) =>
          f.toLowerCase().includes(query)
        );
        if (!matchesId && !matchesFacts) continue;
      }

      ids.add(node.id);
    }
    return ids;
  }, [currentNodes, searchFilter, filterType]);

  // Mouse pan handling
  const handleMouseDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).tagName === "circle" || (e.target as HTMLElement).tagName === "text") {
      return; // Clicking a node
    }
    setIsDragging(true);
    hasUserPanned.current = true;
    dragStartRef.current = { x: e.clientX - viewportRef.current.pan.x, y: e.clientY - viewportRef.current.pan.y };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    const newPan = {
      x: Math.round(e.clientX - dragStartRef.current.x),
      y: Math.round(e.clientY - dragStartRef.current.y),
    };
    viewportRef.current.pan = newPan;
    setViewport((prev) => ({ ...prev, pan: newPan }));
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const resetView = () => {
    hasUserPanned.current = false;
    const newPan = getCenteredPan(0.9);
    viewportRef.current = { zoom: 0.9, pan: newPan };
    setViewport({ zoom: 0.9, pan: newPan });
  };

  const zoomIn = () => {
    const el = containerRef.current;
    const rect = el ? el.getBoundingClientRect() : { width: 850, height: 580 };
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    const curr = viewportRef.current;
    const newZoom = Math.min(MAX_ZOOM, Math.round(curr.zoom * 1.2 * 100) / 100);
    if (newZoom === curr.zoom) return;

    hasUserPanned.current = true;
    const graphX = (centerX - curr.pan.x) / curr.zoom;
    const graphY = (centerY - curr.pan.y) / curr.zoom;
    const newPan = {
      x: Math.round(centerX - graphX * newZoom),
      y: Math.round(centerY - graphY * newZoom),
    };

    viewportRef.current = { zoom: newZoom, pan: newPan };
    setViewport({ zoom: newZoom, pan: newPan });
  };

  const zoomOut = () => {
    const el = containerRef.current;
    const rect = el ? el.getBoundingClientRect() : { width: 850, height: 580 };
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    const curr = viewportRef.current;
    const newZoom = Math.max(MIN_ZOOM, Math.round((curr.zoom / 1.2) * 100) / 100);
    if (newZoom === curr.zoom) return;

    hasUserPanned.current = true;
    const graphX = (centerX - curr.pan.x) / curr.zoom;
    const graphY = (centerY - curr.pan.y) / curr.zoom;
    const newPan = {
      x: Math.round(centerX - graphX * newZoom),
      y: Math.round(centerY - graphY * newZoom),
    };

    viewportRef.current = { zoom: newZoom, pan: newPan };
    setViewport({ zoom: newZoom, pan: newPan });
  };

  // Native wheel listener: cursor-targeted zoom with synchronous ref tracking and RAF throttling
  const isRafScheduled = useRef<boolean>(false);
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();

      const rect = container.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      // Normalize delta across wheel modes (0 = pixel, 1 = line, 2 = page)
      const delta = e.deltaMode === 1 ? e.deltaY * 20 : (e.deltaMode === 2 ? e.deltaY * 60 : e.deltaY);
      // Smooth exponential zoom factor:
      // A standard 100px wheel tick gives ~1.14x zoom; a trackpad 4px tick gives ~1.006x zoom
      const zoomFactor = Math.min(1.35, Math.max(0.74, Math.exp(-delta * 0.0013)));

      const curr = viewportRef.current;
      const targetZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Math.round(curr.zoom * zoomFactor * 1000) / 1000));
      if (targetZoom === curr.zoom) return;

      hasUserPanned.current = true;
      const graphX = (mouseX - curr.pan.x) / curr.zoom;
      const graphY = (mouseY - curr.pan.y) / curr.zoom;
      const newPan = {
        x: Math.round(mouseX - graphX * targetZoom),
        y: Math.round(mouseY - graphY * targetZoom),
      };

      // Immediately synchronize ref coordinates for subsequent events
      viewportRef.current = { zoom: targetZoom, pan: newPan };

      // Throttle React state update to animation frames for 60fps buttery smoothness
      if (!isRafScheduled.current) {
        isRafScheduled.current = true;
        requestAnimationFrame(() => {
          isRafScheduled.current = false;
          setViewport(viewportRef.current);
        });
      }
    };

    container.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      container.removeEventListener("wheel", onWheel);
    };
  }, [MIN_ZOOM, MAX_ZOOM]);

  // Automatically center graph on initial load or when nodes change
  useEffect(() => {
    if (currentNodes.length > 0) {
      hasUserPanned.current = false;
      const timer = setTimeout(() => {
        const newPan = getCenteredPan(0.9);
        viewportRef.current = { zoom: 0.9, pan: newPan };
        setViewport({ zoom: 0.9, pan: newPan });
      }, 50);
      return () => clearTimeout(timer);
    }
  }, [currentNodes.length, getCenteredPan]);

  // Maintain centered position on window or container resize unless user manually panned
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const observer = new ResizeObserver(() => {
      if (!hasUserPanned.current) {
        const newPan = getCenteredPan(viewportRef.current.zoom);
        viewportRef.current.pan = newPan;
        setViewport((prev) => ({ ...prev, pan: newPan }));
      }
    });

    observer.observe(el);
    return () => observer.disconnect();
  }, [getCenteredPan]);

  // ==========================================
  // SEARCH ALGORITHM SIMULATION ENGINE (Russell & Norvig Ch 3 & 4)
  // ==========================================
  const heuristicDistances = useMemo(() => {
    const hDist: Record<string, number> = {};
    if (currentNodes.length === 0) return hDist;

    // Build reverse adjacency list
    const reversedAdj: Record<string, string[]> = {};
    for (const n of currentNodes) reversedAdj[n.id] = [];
    for (const e of currentEdges) {
      if (!reversedAdj[e.target]) reversedAdj[e.target] = [];
      reversedAdj[e.target].push(e.source);
    }

    // Identify goal nodes
    const goalNodeIds = currentNodes.filter((n) => n.is_goal).map((n) => n.id);
    for (const n of currentNodes) hDist[n.id] = Infinity;

    // Multi-source reverse BFS from all goals
    const queue: string[] = [];
    for (const gId of goalNodeIds) {
      hDist[gId] = 0;
      queue.push(gId);
    }

    while (queue.length > 0) {
      const curr = queue.shift()!;
      const currDist = hDist[curr];
      for (const pred of reversedAdj[curr] || []) {
        if (hDist[pred] === Infinity) {
          hDist[pred] = currDist + 1;
          queue.push(pred);
        }
      }
    }

    // Fallback for disconnected / dead-end states
    const maxDepth = Math.max(0, ...currentNodes.map((n) => n.depth));
    for (const n of currentNodes) {
      if (hDist[n.id] === Infinity) {
        hDist[n.id] = Math.max(1, (maxDepth - n.depth) + 1);
      }
    }

    return hDist;
  }, [currentNodes, currentEdges]);

  // Precompute full simulation steps whenever algorithm or graph changes
  const simulationSteps = useMemo(() => {
    const steps: SearchSimulationStep[] = [];
    if (currentNodes.length === 0) return steps;

    const startNode = currentNodes.find((n) => n.is_initial) || currentNodes[0];
    const goalNodeIds = new Set(currentNodes.filter((n) => n.is_goal).map((n) => n.id));

    // Forward adjacency
    const forwardAdj: Record<string, Array<{ target: string; action: string; edgeKey: string; cost: number }>> = {};
    for (const n of currentNodes) forwardAdj[n.id] = [];
    for (const e of currentEdges) {
      if (!forwardAdj[e.source]) forwardAdj[e.source] = [];
      forwardAdj[e.source].push({
        target: e.target,
        action: e.action_name,
        edgeKey: `${e.source}->${e.target}`,
        cost: e.cost ?? 1,
      });
    }

    // Helper to reconstruct solution path
    const buildPath = (parentMap: Record<string, { parent: string; edgeKey: string }>, endNodeId: string) => {
      const pathNodes: string[] = [];
      const pathEdges = new Set<string>();
      let curr = endNodeId;
      pathNodes.push(curr);
      while (parentMap[curr]) {
        const { parent, edgeKey } = parentMap[curr];
        pathEdges.add(edgeKey);
        curr = parent;
        pathNodes.unshift(curr);
      }
      return { pathNodes, pathEdges };
    };

    if (selectedAlgorithm === "astar") {
      // A* Search: priority f(n) = g(n) + h(n)
      interface AStarItem {
        id: string;
        g: number;
        h: number;
        f: number;
      }
      const openSet: Map<string, AStarItem> = new Map();
      const closedSet: Set<string> = new Set();
      const parentMap: Record<string, { parent: string; edgeKey: string }> = {};

      const startH = heuristicDistances[startNode.id] ?? 0;
      openSet.set(startNode.id, { id: startNode.id, g: 0, h: startH, f: startH });

      // Step 0: Initial frontier
      steps.push({
        stepIndex: 0,
        currentNodeId: startNode.id,
        frontierNodeIds: [startNode.id],
        closedNodeIds: [],
        solutionPathNodeIds: [],
        solutionEdgeKeys: new Set(),
        g: 0,
        h: startH,
        f: startH,
        explanation: `A* initialized at initial state ${formatStateId(startNode.id)}. Evaluation: g=0, h=${startH}, f(n)=g+h=${startH}. Placed in Open Set (Frontier).`,
        queueSnapshot: [{ id: formatStateId(startNode.id), g: 0, h: startH, f: startH }],
      });

      let stepCounter = 1;
      while (openSet.size > 0 && stepCounter < 60) {
        // Pick node with lowest f (tie-break lowest h, then lowest g)
        let bestItem: AStarItem | null = null;
        for (const item of openSet.values()) {
          if (!bestItem || item.f < bestItem.f || (item.f === bestItem.f && item.h < bestItem.h)) {
            bestItem = item;
          }
        }
        if (!bestItem) break;

        openSet.delete(bestItem.id);
        closedSet.add(bestItem.id);

        // Check if goal reached
        if (goalNodeIds.has(bestItem.id)) {
          const { pathNodes, pathEdges } = buildPath(parentMap, bestItem.id);
          steps.push({
            stepIndex: stepCounter,
            currentNodeId: formatStateId(bestItem.id),
            frontierNodeIds: Array.from(openSet.keys()).map((k) => formatStateId(k)),
            closedNodeIds: Array.from(closedSet).map((k) => formatStateId(k)),
            solutionPathNodeIds: pathNodes.map((k) => formatStateId(k)),
            solutionEdgeKeys: pathEdges,
            g: bestItem.g,
            h: 0,
            f: bestItem.g,
            explanation: `Goal satisfied at ${formatStateId(bestItem.id)}! A* achieved optimality with path cost g=${bestItem.g} across ${pathNodes.length} states. Zero heuristic remaining (h=0).`,
            queueSnapshot: Array.from(openSet.values()).map((v) => ({ ...v, id: formatStateId(v.id) })),
          });
          break;
        }

        // Expand successors
        const neighbors = forwardAdj[bestItem.id] || [];
        let addedCount = 0;
        for (const nbr of neighbors) {
          if (closedSet.has(nbr.target)) continue;
          const tentativeG = bestItem.g + nbr.cost;
          const existing = openSet.get(nbr.target);

          if (!existing || tentativeG < existing.g) {
            const hVal = heuristicDistances[nbr.target] ?? 0;
            openSet.set(nbr.target, {
              id: nbr.target,
              g: tentativeG,
              h: hVal,
              f: tentativeG + hVal,
            });
            parentMap[nbr.target] = { parent: bestItem.id, edgeKey: nbr.edgeKey };
            addedCount++;
          }
        }

        steps.push({
          stepIndex: stepCounter++,
          currentNodeId: formatStateId(bestItem.id),
          frontierNodeIds: Array.from(openSet.keys()).map((k) => formatStateId(k)),
          closedNodeIds: Array.from(closedSet).map((k) => formatStateId(k)),
          solutionPathNodeIds: [],
          solutionEdgeKeys: new Set(),
          g: bestItem.g,
          h: bestItem.h,
          f: bestItem.f,
          explanation: `A* expanded ${formatStateId(bestItem.id)} (f=${bestItem.f} = g:${bestItem.g} + h:${bestItem.h}). Moved to Closed Set; generated ${addedCount} successor(s) in Frontier.`,
          queueSnapshot: Array.from(openSet.values()).map((v) => ({ ...v, id: formatStateId(v.id) })).sort((a, b) => a.f - b.f),
        });
      }
    } else if (selectedAlgorithm === "bfs") {
      // Breadth-First Search (FIFO queue)
      const queue: string[] = [startNode.id];
      const seen: Set<string> = new Set([startNode.id]);
      const closedSet: Set<string> = new Set();
      const parentMap: Record<string, { parent: string; edgeKey: string }> = {};
      const depthMap: Record<string, number> = { [startNode.id]: 0 };

      steps.push({
        stepIndex: 0,
        currentNodeId: formatStateId(startNode.id),
        frontierNodeIds: [formatStateId(startNode.id)],
        closedNodeIds: [],
        solutionPathNodeIds: [],
        solutionEdgeKeys: new Set(),
        g: 0,
        h: heuristicDistances[startNode.id] ?? 0,
        f: 0,
        explanation: `BFS initialized with start node ${formatStateId(startNode.id)} at depth 0 into FIFO Queue.`,
        queueSnapshot: [{ id: formatStateId(startNode.id), g: 0, h: heuristicDistances[startNode.id] ?? 0, f: 0 }],
      });

      let stepCounter = 1;
      while (queue.length > 0 && stepCounter < 60) {
        const curr = queue.shift()!;
        closedSet.add(curr);
        const currDepth = depthMap[curr] ?? 0;

        if (goalNodeIds.has(curr)) {
          const { pathNodes, pathEdges } = buildPath(parentMap, curr);
          steps.push({
            stepIndex: stepCounter,
            currentNodeId: formatStateId(curr),
            frontierNodeIds: queue.map((k) => formatStateId(k)),
            closedNodeIds: Array.from(closedSet).map((k) => formatStateId(k)),
            solutionPathNodeIds: pathNodes.map((k) => formatStateId(k)),
            solutionEdgeKeys: pathEdges,
            g: currDepth,
            h: 0,
            f: currDepth,
            explanation: `BFS reached goal node ${formatStateId(curr)} at depth ${currDepth}. Optimal step count guaranteed for uniform edge costs.`,
            queueSnapshot: queue.map((id) => ({ id: formatStateId(id), g: depthMap[id] ?? 0, h: heuristicDistances[id] ?? 0, f: depthMap[id] ?? 0 })),
          });
          break;
        }

        const neighbors = forwardAdj[curr] || [];
        let addedCount = 0;
        for (const nbr of neighbors) {
          if (!seen.has(nbr.target)) {
            seen.add(nbr.target);
            depthMap[nbr.target] = currDepth + 1;
            parentMap[nbr.target] = { parent: curr, edgeKey: nbr.edgeKey };
            queue.push(nbr.target);
            addedCount++;
          }
        }

        steps.push({
          stepIndex: stepCounter++,
          currentNodeId: formatStateId(curr),
          frontierNodeIds: queue.map((k) => formatStateId(k)),
          closedNodeIds: Array.from(closedSet).map((k) => formatStateId(k)),
          solutionPathNodeIds: [],
          solutionEdgeKeys: new Set(),
          g: currDepth,
          h: heuristicDistances[curr] ?? 0,
          f: currDepth,
          explanation: `BFS popped ${formatStateId(curr)} from FIFO queue at depth ${currDepth}. Enqueued ${addedCount} unvisited successor(s).`,
          queueSnapshot: queue.map((id) => ({ id: formatStateId(id), g: depthMap[id] ?? 0, h: heuristicDistances[id] ?? 0, f: depthMap[id] ?? 0 })),
        });
      }
    } else {
      // Greedy Best-First Search (priority queue by heuristic h(n) only)
      interface GreedyItem {
        id: string;
        g: number;
        h: number;
        f: number;
      }
      const openSet: Map<string, GreedyItem> = new Map();
      const closedSet: Set<string> = new Set();
      const parentMap: Record<string, { parent: string; edgeKey: string }> = {};

      const startH = heuristicDistances[startNode.id] ?? 0;
      openSet.set(startNode.id, { id: startNode.id, g: 0, h: startH, f: startH });

      steps.push({
        stepIndex: 0,
        currentNodeId: formatStateId(startNode.id),
        frontierNodeIds: [formatStateId(startNode.id)],
        closedNodeIds: [],
        solutionPathNodeIds: [],
        solutionEdgeKeys: new Set(),
        g: 0,
        h: startH,
        f: startH,
        explanation: `Greedy Best-First Search initialized at ${formatStateId(startNode.id)}. Guided solely by heuristic estimate h(n)=${startH}.`,
        queueSnapshot: [{ id: formatStateId(startNode.id), g: 0, h: startH, f: startH }],
      });

      let stepCounter = 1;
      while (openSet.size > 0 && stepCounter < 60) {
        let bestItem: GreedyItem | null = null;
        for (const item of openSet.values()) {
          if (!bestItem || item.h < bestItem.h) {
            bestItem = item;
          }
        }
        if (!bestItem) break;

        openSet.delete(bestItem.id);
        closedSet.add(bestItem.id);

        if (goalNodeIds.has(bestItem.id)) {
          const { pathNodes, pathEdges } = buildPath(parentMap, bestItem.id);
          steps.push({
            stepIndex: stepCounter,
            currentNodeId: formatStateId(bestItem.id),
            frontierNodeIds: Array.from(openSet.keys()).map((k) => formatStateId(k)),
            closedNodeIds: Array.from(closedSet).map((k) => formatStateId(k)),
            solutionPathNodeIds: pathNodes.map((k) => formatStateId(k)),
            solutionEdgeKeys: pathEdges,
            g: bestItem.g,
            h: 0,
            f: 0,
            explanation: `Greedy Best-First reached goal ${formatStateId(bestItem.id)} with h=0! Path found quickly, though not necessarily cost-minimal.`,
            queueSnapshot: Array.from(openSet.values()).map((v) => ({ ...v, id: formatStateId(v.id) })),
          });
          break;
        }

        const neighbors = forwardAdj[bestItem.id] || [];
        let addedCount = 0;
        for (const nbr of neighbors) {
          if (closedSet.has(nbr.target)) continue;
          if (!openSet.has(nbr.target)) {
            const hVal = heuristicDistances[nbr.target] ?? 0;
            openSet.set(nbr.target, {
              id: nbr.target,
              g: bestItem.g + nbr.cost,
              h: hVal,
              f: hVal,
            });
            parentMap[nbr.target] = { parent: bestItem.id, edgeKey: nbr.edgeKey };
            addedCount++;
          }
        }

        steps.push({
          stepIndex: stepCounter++,
          currentNodeId: formatStateId(bestItem.id),
          frontierNodeIds: Array.from(openSet.keys()).map((k) => formatStateId(k)),
          closedNodeIds: Array.from(closedSet).map((k) => formatStateId(k)),
          solutionPathNodeIds: [],
          solutionEdgeKeys: new Set(),
          g: bestItem.g,
          h: bestItem.h,
          f: bestItem.h,
          explanation: `Greedy Best-First expanded ${formatStateId(bestItem.id)} with lowest h(n)=${bestItem.h}. Ignoring historical path cost g(n).`,
          queueSnapshot: Array.from(openSet.values()).map((v) => ({ ...v, id: formatStateId(v.id) })).sort((a, b) => a.h - b.h),
        });
      }
    }

    return steps;
  }, [currentNodes, currentEdges, selectedAlgorithm, heuristicDistances]);

  // Reset simulation when algorithm or graph changes
  useEffect(() => {
    setCurrentSimStepIndex(0);
    setIsPlaying(false);
  }, [selectedAlgorithm, currentNodes.length]);

  // Playback timer loop
  useEffect(() => {
    if (!isPlaying) return;
    if (currentSimStepIndex >= simulationSteps.length - 1) {
      setIsPlaying(false);
      return;
    }
    const timer = setTimeout(() => {
      setCurrentSimStepIndex((prev) => Math.min(simulationSteps.length - 1, prev + 1));
    }, playbackSpeed);
    return () => clearTimeout(timer);
  }, [isPlaying, currentSimStepIndex, simulationSteps.length, playbackSpeed]);

  // Active step
  const activeSimStep = simulationSteps[currentSimStepIndex] || null;

  return (
    <div className="flex flex-col gap-4">
      {/* SEARCH ALGORITHM NODE-BY-NODE ANIMATION TOOLBAR */}
      <div className="flex flex-col gap-2.5 rounded-2xl border border-cyan-500/30 bg-slate-900/90 p-4 shadow-xl backdrop-blur-md">
        <div className="flex flex-wrap items-center justify-between gap-3">
          {/* Algorithm Switcher */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold uppercase tracking-wider text-cyan-400">
              Algorithm:
            </span>
            <div className="inline-flex rounded-lg border border-slate-700 bg-slate-950 p-0.5">
              <button
                type="button"
                onClick={() => setSelectedAlgorithm("astar")}
                className={`rounded-md px-3 py-1 text-xs font-semibold transition ${
                  selectedAlgorithm === "astar"
                    ? "bg-cyan-500 text-slate-950 shadow"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                A* Search (f = g + h)
              </button>
              <button
                type="button"
                onClick={() => setSelectedAlgorithm("bfs")}
                className={`rounded-md px-3 py-1 text-xs font-semibold transition ${
                  selectedAlgorithm === "bfs"
                    ? "bg-cyan-500 text-slate-950 shadow"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                BFS (FIFO Layer)
              </button>
              <button
                type="button"
                onClick={() => setSelectedAlgorithm("greedy")}
                className={`rounded-md px-3 py-1 text-xs font-semibold transition ${
                  selectedAlgorithm === "greedy"
                    ? "bg-cyan-500 text-slate-950 shadow"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Greedy Best-First (h only)
              </button>
            </div>
          </div>

          {/* Playback Controls */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setCurrentSimStepIndex(0)}
              disabled={simulationSteps.length === 0}
              title="Reset to Step 0"
              className="rounded-lg border border-slate-700 bg-slate-800 px-2.5 py-1 text-xs font-medium text-slate-300 hover:bg-slate-700 disabled:opacity-40"
            >
              ↺ Reset
            </button>
            <button
              type="button"
              onClick={() => setCurrentSimStepIndex((s) => Math.max(0, s - 1))}
              disabled={currentSimStepIndex === 0 || simulationSteps.length === 0}
              className="rounded-lg border border-slate-700 bg-slate-800 px-2.5 py-1 text-xs font-medium text-slate-300 hover:bg-slate-700 disabled:opacity-40"
            >
              ⏮ Step
            </button>
            <button
              type="button"
              onClick={() => {
                if (currentSimStepIndex >= simulationSteps.length - 1) {
                  setCurrentSimStepIndex(0);
                  setIsPlaying(true);
                } else {
                  setIsPlaying(!isPlaying);
                }
              }}
              disabled={simulationSteps.length === 0}
              className={`rounded-lg px-3.5 py-1 text-xs font-bold shadow transition ${
                isPlaying
                  ? "bg-amber-500 text-slate-950 hover:bg-amber-400"
                  : "bg-emerald-600 text-white hover:bg-emerald-500"
              } disabled:opacity-40`}
            >
              {isPlaying ? "⏸ Pause" : "▶ Play"}
            </button>
            <button
              type="button"
              onClick={() => setCurrentSimStepIndex((s) => Math.min(simulationSteps.length - 1, s + 1))}
              disabled={currentSimStepIndex >= simulationSteps.length - 1 || simulationSteps.length === 0}
              className="rounded-lg border border-slate-700 bg-slate-800 px-2.5 py-1 text-xs font-medium text-slate-300 hover:bg-slate-700 disabled:opacity-40"
            >
              Step ⏭
            </button>

            {/* Speed Selector */}
            <select
              value={playbackSpeed}
              onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
              style={{ colorScheme: "dark" }}
              className="rounded-lg border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-300 focus:outline-none cursor-pointer"
            >
              <option value={1400}>0.5x</option>
              <option value={800}>1.0x</option>
              <option value={400}>2.0x</option>
              <option value={150}>5.0x</option>
            </select>
          </div>
        </div>

        {/* Step Scrubber and Real-time Metric Strip */}
        <div className="flex flex-col sm:flex-row sm:items-center gap-3 pt-1 border-t border-slate-800/80">
          <div className="flex items-center gap-2 flex-1">
            <span className="text-[11px] font-mono text-cyan-300 whitespace-nowrap">
              Step {simulationSteps.length > 0 ? currentSimStepIndex + 1 : 0} / {simulationSteps.length}
            </span>
            <input
              type="range"
              min={0}
              max={Math.max(0, simulationSteps.length - 1)}
              value={currentSimStepIndex}
              onChange={(e) => setCurrentSimStepIndex(Number(e.target.value))}
              disabled={simulationSteps.length === 0}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
            />
          </div>

          {activeSimStep && (
            <div className="flex items-center gap-3 text-xs">
              <span className="font-mono text-slate-300">
                Focus: <strong className="text-cyan-400">{formatStateId(activeSimStep.currentNodeId)}</strong>
              </span>
              <span className="font-mono text-slate-400">
                g(n)=<strong className="text-indigo-300">{activeSimStep.g}</strong>
              </span>
              <span className="font-mono text-slate-400">
                h(n)=<strong className="text-amber-300">{activeSimStep.h}</strong>
              </span>
              <span className="font-mono text-slate-400">
                f(n)=<strong className="text-emerald-300">{activeSimStep.f}</strong>
              </span>
              <span className="rounded bg-slate-800 px-2 py-0.5 text-[11px] font-mono text-slate-400 border border-slate-700">
                |Open|={activeSimStep.frontierNodeIds.length} |Closed|={activeSimStep.closedNodeIds.length}
              </span>
            </div>
          )}
        </div>

        {/* Explanatory Banner */}
        {activeSimStep && (
          <div className="rounded-xl border border-cyan-500/20 bg-cyan-950/20 p-2.5 text-xs text-cyan-200/90 leading-relaxed">
            <span className="font-bold text-cyan-300 mr-1.5">🎓 Search Concept:</span>
            {activeSimStep.explanation.replace(/\b[0-9a-fA-F]{8,64}\b/g, (h) => formatStateId(h))}
          </div>
        )}
      </div>

      {/* HUD Bar */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-6 rounded-xl border border-slate-800 bg-slate-900/60 p-3">
        <div className="flex flex-col">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500">
            Reachable States
          </span>
          <span className="text-lg font-bold font-mono text-slate-100">
            {currentNodes.length}
          </span>
        </div>
        <div className="flex flex-col">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500">
            Valid Transitions
          </span>
          <span className="text-lg font-bold font-mono text-cyan-400">
            {currentEdges.length}
          </span>
        </div>
        <div className="flex flex-col">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500">
            Max Depth
          </span>
          <span className="text-lg font-bold font-mono text-indigo-400">
            {currentStats?.max_depth_reached ?? 0}
          </span>
        </div>
        <div className="flex flex-col">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500">
            Goal States
          </span>
          <span className="text-lg font-bold font-mono text-emerald-400">
            {currentNodes.filter((n) => n.is_goal).length}
          </span>
        </div>
        <div className="flex flex-col">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500">
            Dead Ends
          </span>
          <span className="text-lg font-bold font-mono text-rose-400">
            {currentNodes.filter((n) => n.is_dead_end).length}
          </span>
        </div>
        <div className="flex flex-col">
          <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500">
            State Space Status
          </span>
          <span className="inline-flex items-center mt-1">
            {currentStatus === "COMPLETED" && (
              <span className="rounded-full bg-emerald-500/20 px-2 py-0.5 text-[11px] font-semibold text-emerald-300 border border-emerald-500/30">
                Complete DAG
              </span>
            )}
            {currentStatus === "TRUNCATED" && (
              <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-[11px] font-semibold text-amber-300 border border-amber-500/30" title={currentStats?.truncation_reason || "Truncated"}>
                Truncated Budget
              </span>
            )}
            {currentStatus === "ENUMERATING" && (
              <span className="rounded-full bg-cyan-500/20 px-2 py-0.5 text-[11px] font-semibold text-cyan-300 border border-cyan-500/30 animate-pulse">
                Enumerating...
              </span>
            )}
            {currentStatus === "IDLE" && (
              <span className="rounded-full bg-slate-700/40 px-2 py-0.5 text-[11px] font-medium text-slate-400 border border-slate-700">
                Ready
              </span>
            )}
          </span>
        </div>
      </div>

      {/* Main Exploration Workspace */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12 min-h-[580px]">
        {/* Interactive SVG Graph Canvas */}
        <div
          ref={containerRef}
          className={`relative lg:col-span-8 flex flex-col rounded-2xl border border-slate-800 bg-slate-950 overflow-hidden shadow-2xl select-none ${isDragging ? "cursor-grabbing" : "cursor-grab"}`}
          style={{ touchAction: "none" }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        >
          {/* Legend and Filter Bar Overlay */}
          <div className="absolute top-3 left-3 z-10 flex flex-wrap items-center gap-2 rounded-xl bg-slate-900/90 p-2 text-xs border border-slate-800 backdrop-blur-md">
            <span className="font-semibold text-slate-400">Filter:</span>
            <button
              type="button"
              onClick={() => setFilterType("all")}
              className={`rounded px-2 py-0.5 text-[11px] transition ${
                filterType === "all" ? "bg-slate-700 text-white font-medium" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              All ({currentNodes.length})
            </button>
            <button
              type="button"
              onClick={() => setFilterType("goal")}
              className={`rounded px-2 py-0.5 text-[11px] transition ${
                filterType === "goal" ? "bg-emerald-600 text-white font-medium" : "text-emerald-400 hover:text-emerald-300"
              }`}
            >
              Goals ({currentNodes.filter((n) => n.is_goal).length})
            </button>
            <button
              type="button"
              onClick={() => setFilterType("dead_end")}
              className={`rounded px-2 py-0.5 text-[11px] transition ${
                filterType === "dead_end" ? "bg-rose-600 text-white font-medium" : "text-rose-400 hover:text-rose-300"
              }`}
            >
              Dead Ends ({currentNodes.filter((n) => n.is_dead_end).length})
            </button>
            <input
              type="text"
              value={searchFilter}
              onChange={(e) => setSearchFilter(e.target.value)}
              placeholder="Search facts (e.g. on, clear)..."
              className="rounded border border-slate-700 bg-slate-800 px-2 py-0.5 text-[11px] text-slate-200 placeholder-slate-500 focus:outline-none"
            />
          </div>

          {/* Color Key Legend Overlay */}
          <div className="absolute bottom-3 left-3 z-10 flex flex-wrap items-center gap-3 rounded-lg bg-slate-900/90 px-3 py-1.5 text-[11px] border border-slate-800 backdrop-blur-sm text-slate-300">
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-cyan-400 shadow-sm" /> Initial
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-400 shadow-sm" /> Goal
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-rose-400 shadow-sm" /> Dead End
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full border-2 border-dashed border-amber-400" /> Frontier (Open)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full border-2 border-cyan-400" /> Explored (Closed)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-500 ring-2 ring-emerald-300" /> Optimal Path
            </span>
          </div>

          {/* Floating Canvas Zoom Controls */}
          <div className="absolute bottom-3 right-3 z-10 flex items-center gap-1 rounded-lg bg-slate-900/90 p-1 text-[11px] border border-slate-800 backdrop-blur-sm shadow-xl text-slate-300">
            <button
              type="button"
              onClick={zoomOut}
              title="Zoom Out (Mouse Wheel Down)"
              className="h-6 w-6 rounded flex items-center justify-center hover:bg-slate-800 text-slate-300 font-bold hover:text-white transition"
            >
              −
            </button>
            <span className="px-1.5 py-0.5 font-mono text-[11px] text-cyan-400 min-w-[42px] text-center font-medium">
              {Math.round(zoom * 100)}%
            </span>
            <button
              type="button"
              onClick={zoomIn}
              title="Zoom In (Mouse Wheel Up)"
              className="h-6 w-6 rounded flex items-center justify-center hover:bg-slate-800 text-slate-300 font-bold hover:text-white transition"
            >
              +
            </button>
            <div className="h-3.5 w-px bg-slate-700 mx-0.5" />
            <button
              type="button"
              onClick={resetView}
              title="Center Graph & Reset Zoom to 90%"
              className="px-2 py-0.5 rounded hover:bg-slate-800 text-slate-400 hover:text-white text-[10px] font-medium transition"
            >
              Reset
            </button>
          </div>

          {/* SVG Canvas */}
          <svg ref={svgRef} className="w-full h-full flex-1 cursor-grab active:cursor-grabbing">
            <defs>
              <marker
                id="arrowhead"
                markerWidth="8"
                markerHeight="6"
                refX="18"
                refY="3"
                orient="auto"
              >
                <polygon points="0 0, 8 3, 0 6" fill="#475569" />
              </marker>
              <marker
                id="arrowhead-cycle"
                markerWidth="8"
                markerHeight="6"
                refX="18"
                refY="3"
                orient="auto"
              >
                <polygon points="0 0, 8 3, 0 6" fill="#c084fc" />
              </marker>
              <marker
                id="arrowhead-solution"
                markerWidth="10"
                markerHeight="8"
                refX="20"
                refY="4"
                orient="auto"
              >
                <polygon points="0 0, 10 4, 0 8" fill="#10b981" />
              </marker>
              <marker
                id="arrowhead-trajectory"
                markerWidth="9"
                markerHeight="7"
                refX="19"
                refY="3.5"
                orient="auto"
              >
                <polygon points="0 0, 9 3.5, 0 7" fill="#38bdf8" />
              </marker>
            </defs>

            <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`} style={{ willChange: "transform" }}>
              {/* Strictly Single-Directional Forward Search Tree Edges */}
              {forwardEdges.map((edge, idx) => {
                const s = nodePositions[edge.source];
                const t = nodePositions[edge.target];
                if (!s || !t) return null;

                const edgeKey = `${edge.source}->${edge.target}`;
                const trajectoryStep = selectedTrajectory.edgeKeyToStep.get(edgeKey);
                const isOnTrajectory = Boolean(trajectoryStep);
                const isHovered = hoveredEdgeKey === edgeKey;
                const isSolutionEdge = activeSimStep ? activeSimStep.solutionEdgeKeys.has(edgeKey) : false;

                const pathData = `M ${s.x} ${s.y} L ${t.x} ${t.y}`;
                const midX = (s.x + t.x) / 2;
                const midY = (s.y + t.y) / 2;

                // Visual styling based on trajectory and simulation state
                let strokeColor = "#334155";
                let strokeWidth = 1.3;
                let markerEnd = "url(#arrowhead)";
                let filterClass: string | undefined = undefined;

                if (isOnTrajectory) {
                  strokeColor = "#38bdf8"; // vibrant cyan for selected search trajectory
                  strokeWidth = 2.8;
                  markerEnd = "url(#arrowhead-trajectory)";
                  filterClass = "filter drop-shadow-[0_0_8px_rgba(56,189,248,0.7)]";
                } else if (isSolutionEdge) {
                  strokeColor = "#10b981"; // emerald for solution
                  strokeWidth = 3;
                  markerEnd = "url(#arrowhead-solution)";
                  filterClass = "filter drop-shadow-[0_0_6px_rgba(16,185,129,0.7)]";
                } else if (isHovered) {
                  strokeColor = "#94a3b8";
                  strokeWidth = 1.8;
                }

                // Strictly show action pill ONLY on the selected root-to-node trajectory (or on hover)
                // This eliminates overlapping labels completely across sibling/divergent branches.
                const showBadge = isOnTrajectory || isHovered;
                const badgeText = isOnTrajectory
                  ? `Step ${trajectoryStep!.stepNumber}: ${trajectoryStep!.actionText}`
                  : formatActionCall(edge);
                const badgeWidth = Math.max(76, badgeText.length * 6.6 + 18);

                return (
                  <g
                    key={`edge-${idx}`}
                    onMouseEnter={() => setHoveredEdgeKey(edgeKey)}
                    onMouseLeave={() => setHoveredEdgeKey(null)}
                    className="cursor-pointer"
                  >
                    {/* Transparent wider hit area for easy hover */}
                    <path
                      d={pathData}
                      fill="none"
                      stroke="transparent"
                      strokeWidth={14}
                    />
                    <path
                      d={pathData}
                      fill="none"
                      stroke={strokeColor}
                      strokeWidth={strokeWidth}
                      markerEnd={markerEnd}
                      className={filterClass}
                    />
                    {showBadge && (
                      <g transform={`translate(${midX}, ${midY})`} className="pointer-events-none select-none">
                        <rect
                          x={-badgeWidth / 2}
                          y={-10}
                          width={badgeWidth}
                          height={20}
                          rx={5}
                          fill="#030712"
                          stroke={isOnTrajectory ? "#0284c7" : "#475569"}
                          strokeWidth={isOnTrajectory ? 1.5 : 1}
                          className={isOnTrajectory ? "filter drop-shadow-[0_2px_10px_rgba(2,132,199,0.6)]" : "filter drop-shadow-sm"}
                        />
                        <text
                          textAnchor="middle"
                          dy="3.5"
                          fontSize="9"
                          fill={isOnTrajectory ? "#38bdf8" : "#cbd5e1"}
                          fontWeight="bold"
                          fontFamily="monospace"
                        >
                          {badgeText}
                        </text>
                      </g>
                    )}
                  </g>
                );
              })}

              {/* Nodes */}
              {currentNodes.map((node) => {
                const pos = nodePositions[node.id];
                if (!pos) return null;

                const isSelected = node.id === selectedNodeId;
                const isMatchingFilter = filteredNodeIds.has(node.id);

                // Simulation highlights
                const isCurrentSimNode = activeSimStep ? activeSimStep.currentNodeId === node.id : false;
                const isFrontierSimNode = activeSimStep ? activeSimStep.frontierNodeIds.includes(node.id) : false;
                const isClosedSimNode = activeSimStep ? activeSimStep.closedNodeIds.includes(node.id) : false;
                const isSolutionSimNode = activeSimStep ? activeSimStep.solutionPathNodeIds.includes(node.id) : false;

                let fillColor = "#475569"; // slate default
                let strokeColor = "#334155";
                let radius = 13;

                if (node.is_initial) {
                  fillColor = "#0284c7"; // cyan/blue
                  strokeColor = "#38bdf8";
                  radius = 16;
                } else if (node.is_goal) {
                  fillColor = "#059669"; // emerald
                  strokeColor = "#34d399";
                  radius = 17;
                } else if (node.is_dead_end) {
                  fillColor = "#e11d48"; // rose
                  strokeColor = "#f43f5e";
                  radius = 14;
                } else if (node.is_truncated_boundary) {
                  fillColor = "#d97706"; // amber
                  strokeColor = "#fbbf24";
                  radius = 14;
                }

                return (
                  <g
                    key={`node-${node.id}`}
                    transform={`translate(${pos.x}, ${pos.y})`}
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedNodeId(node.id);
                    }}
                    onMouseEnter={() => setHoveredNodeId(node.id)}
                    onMouseLeave={() => setHoveredNodeId(null)}
                    className="cursor-pointer"
                    opacity={isMatchingFilter ? 1 : 0.25}
                  >
                    {/* Stable Selection or Hover Halo (NO transform scale vibration) */}
                    {(isSelected || hoveredNodeId === node.id) && (
                      <circle
                        r={radius + 5}
                        fill="none"
                        stroke={isSelected ? "#38bdf8" : "#94a3b8"}
                        strokeWidth={2}
                        strokeDasharray={isSelected ? undefined : "3 3"}
                        className="pointer-events-none"
                      />
                    )}

                    {/* Trajectory Path Node Ring */}
                    {selectedTrajectory.pathNodeIds.includes(node.id) && !isSelected && (
                      <circle
                        r={radius + 4}
                        fill="none"
                        stroke="#38bdf8"
                        strokeWidth={1.5}
                        strokeOpacity={0.65}
                        className="pointer-events-none"
                      />
                    )}

                    {/* Active Expanding Node Halo */}
                    {isCurrentSimNode && (
                      <circle
                        r={radius + 6}
                        fill="none"
                        stroke="#06b6d4"
                        strokeWidth={2.5}
                        strokeDasharray="4 2"
                        className="pointer-events-none"
                      />
                    )}

                    {/* Frontier / Open Set Ring */}
                    {isFrontierSimNode && !isCurrentSimNode && (
                      <circle
                        r={radius + 3.5}
                        fill="none"
                        stroke="#f59e0b"
                        strokeWidth={2}
                        strokeDasharray="3 2"
                        className="pointer-events-none"
                      />
                    )}

                    {/* Explored / Closed Set Ring */}
                    {isClosedSimNode && !isCurrentSimNode && !isSolutionSimNode && (
                      <circle
                        r={radius + 3}
                        fill="none"
                        stroke="#38bdf8"
                        strokeWidth={1.5}
                        opacity={0.65}
                        className="pointer-events-none"
                      />
                    )}

                    {/* Optimal Solution Path Halo */}
                    {isSolutionSimNode && (
                      <circle
                        r={radius + 4.5}
                        fill="none"
                        stroke="#10b981"
                        strokeWidth={2.5}
                        className="pointer-events-none filter drop-shadow-[0_0_5px_rgba(16,185,129,0.8)]"
                      />
                    )}

                    {/* Main Node Body */}
                    <circle
                      r={radius}
                      fill={
                        isSolutionSimNode
                          ? "#059669"
                          : isCurrentSimNode
                          ? "#0284c7"
                          : fillColor
                      }
                      stroke={
                        isSelected
                          ? "#ffffff"
                          : isCurrentSimNode
                          ? "#38bdf8"
                          : isSolutionSimNode
                          ? "#34d399"
                          : strokeColor
                      }
                      strokeWidth={isSelected ? 3 : 1.5}
                      className="filter drop-shadow-md"
                    />

                    <text
                      textAnchor="middle"
                      dy="4"
                      fontSize="9"
                      fill="#ffffff"
                      fontWeight="bold"
                      fontFamily="monospace"
                      className="pointer-events-none select-none"
                    >
                      {node.depth}
                    </text>
                  </g>
                );
              })}
            </g>
          </svg>

          {currentNodes.length === 0 && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-6 bg-slate-950/70 backdrop-blur-sm pointer-events-none">
              <span className="text-sm font-semibold text-slate-300 mb-1">
                No Reachable States Loaded
              </span>
              <p className="max-w-md text-xs text-slate-400 mb-4">
                Execute a task from the Mission Control tab to stream and explore the reachable state space DAG in real time.
              </p>
            </div>
          )}
        </div>

        {/* Side Inspector & Local Transition Subgraph Panel */}
        <div className="lg:col-span-4 flex flex-col rounded-2xl border border-slate-800 bg-slate-900/90 p-5 shadow-2xl backdrop-blur-md overflow-hidden">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-cyan-400">
              State & Search Inspector
            </h3>
            {selectedNode && (
              <span className="font-mono text-xs rounded bg-slate-800 px-2 py-0.5 text-slate-300">
                {formatStateId(selectedNode.id)}
              </span>
            )}
          </div>

          {selectedNode ? (
            <div className="flex flex-col gap-4 overflow-y-auto pr-1">
              {/* Node Metadata Badges */}
              <div className="flex flex-wrap gap-1.5">
                {selectedNode.is_initial && (
                  <span className="rounded bg-cyan-500/20 px-2 py-0.5 text-[10px] font-bold text-cyan-300 border border-cyan-500/30">
                    INITIAL STATE (s0)
                  </span>
                )}
                {selectedNode.is_goal && (
                  <span className="rounded bg-emerald-500/20 px-2 py-0.5 text-[10px] font-bold text-emerald-300 border border-emerald-500/30">
                    GOAL SATISFIED
                  </span>
                )}
                {selectedNode.is_dead_end && (
                  <span className="rounded bg-rose-500/20 px-2 py-0.5 text-[10px] font-bold text-rose-300 border border-rose-500/30">
                    DEAD END STATE
                  </span>
                )}
                {selectedNode.is_truncated_boundary && (
                  <span className="rounded bg-amber-500/20 px-2 py-0.5 text-[10px] font-bold text-amber-300 border border-amber-500/30">
                    FRONTIER BOUNDARY
                  </span>
                )}
                <span className="rounded bg-slate-800 px-2 py-0.5 text-[10px] font-semibold text-slate-400">
                  Depth: {selectedNode.depth}
                </span>
              </div>

              {/* HEURISTIC & EVALUATION CARD */}
              <div className="rounded-xl border border-slate-700/60 bg-slate-950 p-3 flex flex-col gap-2">
                <span className="text-[10px] font-bold uppercase tracking-wider text-cyan-400">
                  Search Evaluation Function (f = g + h)
                </span>
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div className="rounded bg-slate-900 p-2 border border-slate-800">
                    <span className="block text-[10px] text-slate-400">g(n) Cost</span>
                    <span className="font-mono text-sm font-bold text-indigo-300">
                      {selectedNode.depth}
                    </span>
                  </div>
                  <div className="rounded bg-slate-900 p-2 border border-slate-800">
                    <span className="block text-[10px] text-slate-400">h(n) Heuristic</span>
                    <span className="font-mono text-sm font-bold text-amber-300">
                      {heuristicDistances[selectedNode.id] === Infinity ? "∞" : heuristicDistances[selectedNode.id] ?? 0}
                    </span>
                  </div>
                  <div className="rounded bg-slate-900 p-2 border border-slate-800">
                    <span className="block text-[10px] text-slate-400">f(n) Total</span>
                    <span className="font-mono text-sm font-bold text-emerald-300">
                      {heuristicDistances[selectedNode.id] === Infinity
                        ? "∞"
                        : selectedNode.depth + (heuristicDistances[selectedNode.id] ?? 0)}
                    </span>
                  </div>
                </div>
                <p className="text-[10px] text-slate-400 italic">
                  Admissible backward distance estimate to nearest goal. Guarantees A* optimality under tree/graph search.
                </p>
              </div>

              {/* SEARCH TRAJECTORY: ROOT-TO-NODE STEP-BY-STEP */}
              <div className="flex flex-col gap-2 rounded-xl border border-cyan-500/30 bg-cyan-950/20 p-3">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-cyan-400">
                    Search Trajectory ({selectedTrajectory.pathEdges.length} {selectedTrajectory.pathEdges.length === 1 ? "Step" : "Steps"} from Root)
                  </span>
                  <span className="font-mono text-[10px] text-cyan-300 font-semibold">
                    Depth {selectedNode.depth}
                  </span>
                </div>

                {selectedTrajectory.pathEdges.length > 0 ? (
                  <div className="flex flex-col gap-1.5 max-h-48 overflow-y-auto pr-1">
                    {selectedTrajectory.pathEdges.map((e, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between rounded bg-slate-900/90 px-2.5 py-1.5 text-xs border border-slate-800"
                      >
                        <div className="flex items-center gap-2 overflow-hidden mr-2">
                          <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-cyan-500/20 text-[9px] font-bold text-cyan-300 border border-cyan-500/40">
                            {idx + 1}
                          </span>
                          <span className="font-mono text-emerald-300 text-[11px] font-semibold truncate" title={formatActionCall(e)}>
                            {formatActionCall(e)}
                          </span>
                        </div>
                        <span className="font-mono text-[10px] text-slate-400 shrink-0">
                          <button
                            type="button"
                            onClick={() => setSelectedNodeId(e.source)}
                            className="hover:text-cyan-300 hover:underline"
                            title={`Jump to ${formatStateId(e.source)}`}
                          >
                            {formatStateId(e.source)}
                          </button>
                          <span className="text-slate-600 mx-1">→</span>
                          <button
                            type="button"
                            onClick={() => setSelectedNodeId(e.target)}
                            className="hover:text-cyan-300 hover:underline font-bold text-slate-300"
                            title={`Jump to ${formatStateId(e.target)}`}
                          >
                            {formatStateId(e.target)}
                          </button>
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <span className="text-xs text-slate-300 italic">
                    {selectedNode.is_initial
                      ? "Initial State s0 (Root of Search Tree · Depth 0). Select any child node to view its step-by-step path from root."
                      : "Search Frontier Root"}
                  </span>
                )}
              </div>

              {/* Active Symbolic Facts */}
              <div className="flex flex-col gap-1.5">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                  Active Symbolic Facts ({(selectedNode.facts_summary || []).length})
                </span>
                <div className="max-h-40 overflow-y-auto rounded-lg border border-slate-800 bg-slate-950 p-2.5 flex flex-col gap-1">
                  {(selectedNode.facts_summary || []).length > 0 ? (
                    (selectedNode.facts_summary || []).map((factStr, idx) => (
                      <span
                        key={idx}
                        className="font-mono text-xs text-emerald-300 bg-emerald-950/20 px-2 py-0.5 rounded border border-emerald-900/30"
                      >
                        {factStr}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs text-slate-500 italic">No facts recorded.</span>
                  )}
                </div>
              </div>

              {/* ENABLED FORWARD SEARCH ACTIONS */}
              <div className="flex flex-col gap-1.5">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                  Forward Search Actions ({forwardActions.length})
                </span>
                <div className="max-h-44 overflow-y-auto rounded-lg border border-slate-800 bg-slate-950 p-2 flex flex-col gap-1.5">
                  {forwardActions.length > 0 ? (
                    forwardActions.map((e, idx) => {
                      const tNode = currentNodes.find((n) => n.id === e.target);
                      return (
                        <div
                          key={idx}
                          onClick={() => setSelectedNodeId(e.target)}
                          className="cursor-pointer flex items-center justify-between rounded bg-slate-900 p-2 text-xs hover:bg-slate-800 transition border border-slate-800/80"
                        >
                          <span className="font-mono font-bold text-emerald-400">
                            {formatActionCall(e)}
                          </span>
                          <span className="font-mono text-[10px] text-slate-400 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                            to {formatStateId(e.target)} (d={tNode?.depth ?? 0})
                          </span>
                        </div>
                      );
                    })
                  ) : (
                    <span className="text-xs text-slate-400 italic p-1">
                      {selectedNode.is_goal
                        ? "Goal state reached! No further forward action expansion needed."
                        : "No forward actions (Dead End or Leaf)."}
                    </span>
                  )}
                </div>

                {/* PRUNED REVERSE TRANSITIONS (Explains why put_down or unstack are not followed) */}
                {reverseActions.length > 0 && (
                  <div className="mt-1 flex flex-col gap-1">
                    <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
                      Pruned Cycle Actions ({reverseActions.length})
                    </span>
                    <div className="rounded-lg border border-slate-800/70 bg-slate-950/60 p-2 flex flex-col gap-1.5">
                      {reverseActions.map((e, idx) => (
                        <div key={idx} className="flex items-center justify-between text-xs text-slate-500">
                          <span className="font-mono line-through text-slate-500 text-[11px]">
                            {formatActionCall(e)}
                          </span>
                          <span className="text-[9px] font-medium text-rose-400 bg-rose-950/30 px-1.5 py-0.5 rounded border border-rose-900/30">
                            Pruned by A* Closed List (to {formatStateId(e.target)})
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="flex flex-1 items-center justify-center text-center p-6 text-slate-500 text-xs">
              Click any node in the graph to inspect its symbolic state facts, search heuristics (g, h, f), incoming transitions, and enabled actions.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
