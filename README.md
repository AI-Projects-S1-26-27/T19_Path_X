# RE-PLAN-V

**RE-PLAN-V** is a text-first planning demo for converting natural-language tasks into symbolic problems, searching for action plans, and checking those plans against declared domain rules. When a candidate plan fails verification, the system can use the first failure witness to constrain replanning.The LLM formalizes the task. Symbolic search and verification determine whether a plan is valid or not.

## Project information

| Field         | Details                                                                                   |
| ------------- | ----------------------------------------------------------------------------------------- |
| Course        | CS F407 / U407 Artificial Intelligence, BITS Pilani                                       |
| Team          | Path_X                                                                                    |
| Track         | Demo                                                                                      |
| Project topic | Search & Problem Solving                                                                  |
| Repository    | [github.com/saicharanyadavalli/RE-PLAN-V](https://github.com/saicharanyadavalli/RE-PLAN-V) |

## Team

| No. | Name                           | BITS ID       | Role                           |
| --: | ------------------------------ | ------------- | ------------------------------ |
|   1 | Sai Gautam Reddy               | 2024A3PS0429H | Planning Lead                  |
|   2 | Deeptimahanti Aashrith         | 2024AAPS0189H | KR&R Lead                      |
|   3 | Yadavalli Sai Charan           | 2024A8PS0460H | Verification Lead              |
|   4 | Chandhireddy Sathwik Reddy     | 2024AAPS0265H | Repair Lead                    |
|   5 | Sana Nitchel Kumar             | 2024A7PS0056H | NLP/LLM Lead                   |
|   6 | M. Yaswanth Pranav             | 2024A4PS0651H | Dataset/Research Lead          |
|   7 | Maudela Bhargava Krishna Reddy | 2024A8PS0349H | Integration/Visualization Lead |

## How it works

```text
Natural-language task
        ↓
Domain routing and symbolic formalization
        ↓
Validation and symbolic problem construction
        ↓
BFS, Greedy Best-First, or A* planning
        ↓
Formal plan verification
        ↓ on failure
Failure witness → rule-based attribution → constrained replanning → re-verification
```

### Symbolic representation

Each task is represented with typed entities and predicate facts. A domain declares its predicates, action parameters, preconditions, effects, and invariants. The planner grounds those action schemas with the task's entities, applies valid actions to generate successor states, and searches for a state satisfying all goal facts. The LLM proposes the task representation; validation checks it before it enters search.

## Features

- Six registered symbolic domains: Blocks World, Grid World, Household Robotics, Warehouse Logistics, Job Scheduling, and Software Workflow.
- LLM-backed task formalization, with a deterministic mock provider for offline demonstrations.
- BFS, Greedy Best-First, and A* search over declarative domain actions.
- State-space graph enumeration and interactive frontend exploration.
- Verification of action preconditions, hard constraints, invariants, and final goals.
- Optional simulated-fault demonstration of counterexample-guided repair and replanning.
- Next.js frontend with live pipeline events streamed from the FastAPI backend over WebSockets.

### Registered domains

| Domain                           | Example planning problem                                                |
| -------------------------------- | ----------------------------------------------------------------------- |
| Blocks World                     | Pick up, put down, stack, and unstack blocks.                           |
| Grid World                       | Navigate connected cells and move an item.                              |
| Household Robotics (Abstract)    | Navigate rooms and move items to or from surfaces and containers.       |
| Warehouse Operations             | Move a carrier between zones and retrieve or deposit inventory.         |
| Scheduling & Resource Allocation | Schedule independent or precedence-constrained jobs on resources.       |
| Software Automation / CI/CD      | Plan tests, security scans, artifact builds, approvals, and deployment. |

These are symbolic planning domains; their names do not imply connection to physical robots, warehouse hardware, or external deployment systems.

### What the demo displays

- The routed domain and formalized entities, initial facts, goals, and constraints.
- The candidate action sequence, selected planner, and execution trace.
- The verification result, including the first failed action and violated condition when verification fails.
- When fault injection is enabled, the counterexample witness, rule-based attribution, repair constraint, replanned actions, and final verification.
- When graph enumeration is requested, reachable states and transitions within the configured exploration budget.

The state-space graph is limited by its enumeration budget and may be marked truncated. It represents the configured symbolic model, not a live physical environment.

### Interfaces and providers

- `POST /api/pipeline/run` runs the pipeline and returns a structured result.
- `WS /ws/pipeline` streams pipeline stages and events to the frontend.
- `GET /domains` lists registered domains; `POST /graph/enumerate` enumerates a reachable state-space graph.
- The frontend offers OpenRouter, NVIDIA NIM Cloud, Gemini, and a deterministic mock provider. Cloud providers require their own configured API keys; exact model selection is configured separately.

### Planner notes

- Greedy Best-First uses the number of unsatisfied goal facts as its heuristic.
- A* currently defaults to a zero heuristic and therefore orders states by accumulated action cost. The `h_max` and `h_add` Relaxed Planning Graph heuristic is implemented but is not connected to the default planning pipeline.
- The State Space Explorer computes a separate heuristic for its visual search simulation from the graph it receives.
- A counterexample records the verifier's first failure; it is not minimized. Fault attribution uses deterministic rules rather than an LLM classifier.

## Scope

RE-PLAN-V plans over symbolic world models. Verification claims apply to those models and their declared rules. The project does not include image perception, live physical-world observation, or robot actuation. Fault injection simulates an invalid candidate plan to demonstrate the recovery workflow.

The repository includes benchmark code and saved experiment results. Report measurements with their specific experiment and instance count; they are not general performance guarantees.

## Getting started

Requirements: Python 3.10 or newer, Node.js, and npm.

### Backend

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

For cloud formalization, add the relevant API key to `.env`. The symbolic pipeline can also be run with the deterministic mock provider.

Start the API server:

```powershell
python -m app.main --mode server --host 127.0.0.1 --port 8080
```

Health endpoint: [http://127.0.0.1:8080/health](http://127.0.0.1:8080/health)

### Frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). By default, the frontend connects to the backend on port `8080`. Set `NEXT_PUBLIC_API_URL` or `NEXT_PUBLIC_WS_URL` to override the API or WebSocket endpoint.

## Demonstrate recovery

1. Enter a Blocks World task with a clear initial state and goal.
2. Select a planner and formalization provider.
3. Enable **Simulate Initial Planning Fault** and run the pipeline.
4. Follow the verification failure, witness, attribution, repair, and final verification in the event trace.

With fault injection off, a valid candidate may pass verification directly, so the counterexample-repair stages may not appear.

## Repository structure

Selected source folders; generated files, caches, and detailed test files are omitted.

```text
RE-PLAN-V/
├── app/
│   ├── api/                 # REST and WebSocket endpoints
│   ├── services/            # Pipeline orchestration and problem building
│   └── static/              # Static web interface
├── benchmark/               # Benchmark datasets and mutation utilities
├── configs/                 # Runtime defaults
├── core/
│   ├── actions/             # Action schemas, grounding, and transitions
│   ├── attribution/         # Fault classification
│   ├── counterexamples/     # Verification failure witnesses
│   ├── domains/             # Registered planning domains
│   ├── graph/               # Reachable state-space enumeration
│   ├── krr/                 # Knowledge representation and reasoning
│   ├── repair/              # Repair constraint generation
│   ├── replanning/          # Counterexample-guided replanning loop
│   ├── search/              # BFS, Greedy Best-First, A*, and heuristics
│   ├── verification/        # Formal plan verifier
│   └── world/               # Symbolic states, problems, and constraints
├── evaluation/              # Benchmarks, metrics, and saved results
├── dynamic/                 # Symbolic world monitoring and replanning
├── frontend/
│   ├── app/                 # Next.js routes and pages
│   ├── components/          # Mission Control and graph UI
│   ├── hooks/               # WebSocket and frontend state hooks
│   └── types/               # Frontend data contracts
├── interpretation/
│   ├── llm/                 # Provider integrations and mock provider
│   ├── schemas/             # Formalization schemas
│   └── validator/           # Proposal consistency validation
├── scripts/                 # Benchmark and provider utilities
├── tests/                   # Unit, integration, scenario, and other tests
├── .env.example             # Example provider configuration
├── app/main.py              # API, CLI, and benchmark entry point
├── pyproject.toml           # Python project metadata
├── requirements.txt         # Python dependencies
└── README.md
```
