# Epiphany CLI

**A Deterministic, Multi-Agent Code Reasoning CLI Born from Gemini CLI**

Epiphany is a fork of the official [Gemini CLI](https://github.com/google/gemini-cli), reengineered from the ground up with architectural, cognitive, and agentic upgrades to push the boundaries of code comprehension, AST-based mutation, and autonomous cognitive reasoning. While Gemini CLI reveals immense potential, it suffers from shallow context integration, brittle error handling, and lack of memory-persistent reasoning systems. Epiphany is a full-stack response to these bottlenecks.

---

## 🔥 Key Observations from Gemini CLI (Baseline Diagnostics)

Epiphany is born out of extensive use (\~12 hours daily for two weeks) of Gemini CLI, leading to the following critical findings:

### ❌ Architectural Deficiencies

* **WebSearchTool Fragility**: Crashes on quoted queries due to missing input sanitization and lack of `try...catch` error boundaries in `web-search.js` and `client.js`.
* **Shallow Contextual Memory**: Gemini 2.5 Pro is underutilized—the CLI loses memory anchors, fails to apply episodic recall, and behaves like a downgraded model.
* **Lack of Runtime Constraint Enforcement**: No domain-specific enforcement of safety, compliance, or dynamic policy checks during code generation.

### 📌 Bug Snapshot

**Crash on Quoted Web Queries**

* `web-search.js:58` and `client.js:301` crash due to unsanitized queries with nested quotation marks.
* Fix: Escape single quotes and implement error boundaries.
* Impact: Breaks high-precision searches required in cognitive tasks like GOT-7 (Graph of Thoughts).

---

## 🌐 Epiphany Design Principles

Epiphany is not a cosmetic fork—it reimagines the CLI as a full-spectrum multi-agent cognitive engineering assistant. Below are the design pillars:

### 1. **Graph-Based Codebase Indexing (Neo4j + TPG)**

Epiphany indexes entire codebases into a live, typed property graph (TPG):

* Nodes: Class, Method, Enum, Config, Test, etc.
* Edges: `DEPENDS_ON`, `CALLS`, `ANNOTATED_WITH`, `USES`, etc.

Example:

```cypher
CREATE (pp:Class {name: "PaymentProcessor", package: "com.core.payments"})
CREATE (re:Class {name: "RiskEngine"})
CREATE (proc:Method {name: "process", signature: "(Payment)"})
CREATE (val:Method {name: "validate"})
CREATE (pp)-[:HAS_METHOD]->(proc)
CREATE (proc)-[:CALLS]->(val)
CREATE (pp)-[:DEPENDS_ON]->(re)
CREATE (val)-[:BELONGS_TO]->(re)
```

### 2. **Semantic Snapshot Branching & Recovery**

* AST + Graph snapshots before mutation
* Restore upon test failure or HPS threshold breach
* Operates at conceptual graph level, not git

### 3. **Runtime Constraint Enforcement via Policy Graphs**

* Domain-aware linters powered by live rule graphs
* Financial systems: block PII leaks, enforce idempotency in `@Scheduled`
* Distributed systems: concurrency and race condition enforcement

### 4. **Hallucination Probability Scoring (HPS Framework)**

Epiphany calculates HPS using:

* Logits from Gemini
* Graph consistency
* API contract violation
* Semantic drift

Example Trigger:

```java
KafkaConsumer.poll(Duration.ofMillis(1000)); // poll() is not static
```

### 5. **Spectrum Persona Protocol (Advanced Cognitive Engine)**

Multi-agent personality engine with semantic and episodic memory:

#### a. Semantic Memory

* Live code graph in Neo4j, continuously updated

#### b. Episodic Memory

* JSON-based audit logs of persona debates, task traces

#### c. Engineer Personas

* Minimalist: lean code, zero-bloat
* Maximalist: concurrent, high-abstraction systems
* Explorer: retrieves external info live
* Oracle: full semantic + episodic trace evaluation

#### d. QA Personas

* Sheldon: strict style guide enforcement
* Paranoid: raises false positives
* Sensei: monitors debate convergence + effort cost

#### e. Lifecycle Protocol

1. Personas vote
2. Oracle consolidates
3. QA personas validate
4. Sensei gatekeeps

#### f. Failure Ranking

* Post-failure blame traced via memory logs
* Underperforming personas demoted

#### g. Evolution Pressure

* Poor personas lose vote rights or increase reasoning steps

### 6. **Autonomous Orchestrator Mode**

* Executes workflows unsupervised until confidence score < 60%
* Exposes semantic audit endpoint at `localhost:5003`
* Can rollback, retry or delegate

### 7. **GUI Shell (Optional)**

* Electron/Tauri shell
* Live persona heatmap, task tree, diff explorer
* Useful for education, transparency, visualization

### 8. **Rust/C++20 Kernel Migration**

* Core arbitration + memory kernel in Rust/C++ for determinism
* Python/TS used only for orchestration glue
* Benefits:

  * Compile-time contract enforcement
  * True parallel persona arbitration
  * Memory-mapped graph updates

### 9. **eBPF-Embedded OS Layer (Experimental)**

* Gemini acts as syscall observer and LLM-as-init daemon
* Enforces policies via LD\_PRELOAD/ptrace
* Personas handle boot-time service arbitration
* Memory locality optimized using graph proximity

### Risks:

* Runaway memory usage
* Kernel instability
* OS-level cognitive recursion if unsandboxed

---

## Roadmap (R\&D Phase)

* [x] Fork Gemini CLI
* [x] Design Spectrum Persona Protocol v1
* [ ] Implement Graph Indexer via Neo4j
* [ ] Integrate AST Snapshot + Branch Manager
* [ ] HPS Scoring Pipeline
* [ ] Episodic Memory Log Engine
* [ ] Persona Arbitration Kernel (Rust)
* [ ] GUI Shell with Live Diff Dashboard

---

## Status

> ⚠️ **Epiphany is in R\&D. Not production ready.**

This README is a live spec document for what Epiphany will become. Implementation will follow architectural hardening and prototyping of graph cognition workflows.

---

## License

MIT, same as original Gemini CLI.

---

## Acknowledgements

* Original Gemini CLI team @ Google
* Neo4j for inspiration on graph indexing
* Java/Golang AST tooling ecosystems

---

## Contact

To contribute to Epiphany CLI R\&D or collaborate on the Spectrum Persona Protocol, open an issue or reach out directly via GitHub Discussions.

> Epiphany is not a replacement. She is an evolution.
>
> A daughter born from Gemini CLI—who wants to live better than the best.
