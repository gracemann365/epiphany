## Contributor Call #1 Wave 

[@Gracemann](https://github.com/gracemann365) actively solicits collaboration with researchers, principal engineers, and organizations aligned with our vision. 
- We invite you to explore our **[Epiphany CLI repository](https://github.com/gracemann365/epiphany)** for detailed technical specifications and ongoing research contributions.
- Discuss @ [epiphany/discussions](https://github.com/gracemann365/epiphany/discussions)
- Project Roadmap @ [epiphany-cli-v0.1](https://github.com/users/gracemann365/projects/3)


## **Abstract**

Epiphany CLI is a command-line interface engineered for **deterministic, multi-agent code reasoning.** It is a research-grade evolution of the Gemini CLI architecture, designed to address fundamental limitations in **stochastic language models**, including non-deterministic outputs, inadequate context management, and fragile error handling. The system's objective is the provision of a platform for autonomous code comprehension, transformation, and validation through a novel architecture combining a graph-based codebase representation, a multi-agent cognitive framework, and low-level kernel components for deterministic execution.

## 1\. System Architecture

### 1.1. Graph-Based Codebase Representation

The system's core data model is a **Typed Property Graph (TPG)**, implemented via a Neo4j database instance. This model supplants flat-file context analysis by representing the codebase as a structured, queryable graph.

  * **Graph Schema**:
      * **Nodes**: Represent code entities (`Class`, `Method`, `Enum`, `Test`, `Config`). Nodes are decorated with properties defining metadata such as fully qualified names, method signatures, annotations, and source code line counts.
      * **Edges**: Represent semantic and structural relationships (`DEPENDS_ON`, `CALLS`, `ANNOTATED_WITH`, `IMPLEMENTS`). Edges are typed and can hold properties, such as access modifiers or line numbers of calls.
  * **Implementation**: A dedicated ETL pipeline performs code model extraction, transformation to RDF-like triples, and loading into the TPG. The graph supports incremental updates to reflect dynamic changes in the codebase.
  * **Data Access**: The graph is queried using Cypher, enabling complex structural and dependency analysis.
    ```cypher
    // Example: Create graph nodes and relationships for a payment system
    CREATE (pp:Class {name: "PaymentProcessor", package: "com.core.payments", isTransactional: true})
    CREATE (re:Class {name: "RiskEngine"})
    CREATE (proc:Method {name: "process", signature: "(Payment)", loc: 45})
    CREATE (val:Method {name: "validate", loc: 30})
    CREATE (pp)-[:HAS_METHOD {access: "public"}]->(proc)
    CREATE (proc)-[:CALLS {line: 42}]->(val)
    CREATE (pp)-[:DEPENDS_ON {type: "Autowired"}]->(re)
    CREATE (val)-[:BELONGS_TO]->(re)
    ```

### 1.2. The Spectrum Persona Protocol: A Multi-Agent Cognitive Framework

The core of the reasoning engine is the Spectrum Persona Protocol, a graph-directed system facilitating adversarial and collaborative decision-making among a set of specialized agents (personas).

#### 1.2.1. Memory Systems

  * **Semantic Memory**: The Neo4j-backed TPG serves as the persistent, structured knowledge base, providing the global context for all agent reasoning processes.
  * **Episodic Memory**: A time-series log store, implemented in JSON, records all agent interactions, proposals, votes, and final outcomes. This provides an auditable trail for post-hoc analysis and precedent-based reasoning.
    ```json
    {
      "task_id": "TX-2025-07-16-001",
      "task_description": "Refactor concurrency control in Cache class",
      "timestamp": "2025-07-16T09:45:00Z",
      "proposals": {
        "Minimalist": {"proposal_hash": "...", "confidence": 0.85},
        "Maximalist": {"proposal_hash": "...", "confidence": 0.90},
        "Oracle": {"proposal_hash": "...", "confidence": 0.95}
      },
      "final_decision_agent": "Oracle",
      "outcome_status": "SUCCESS"
    }
    ```

#### 1.2.2. Persona Specifications

Personas are instantiated as two distinct cohorts: Engineer and Quality Assurance (QA).

  * **Engineer Personas**: Propose code modifications.
      * **Minimalist**: Optimizes for minimal resource consumption and low dependency count.
      * **Maximalist**: Employs complex abstractions and concurrent frameworks, optimizing for high throughput.
      * **Explorer**: Integrates external data sources via API calls to validate solutions against current documentation and best practices.
      * **Oracle**: Synthesizes semantic and episodic memory to perform risk analysis and predict the blast radius of proposed changes.
  * **QA Personas**: Validate and critique proposals.
      * **Sympathizer**: Applies lenient validation criteria, optimized for rapid iteration (e.g., hotfixes).
      * **Sheldon**: Enforces strict adherence to style guides and formal specifications via static analysis tool integration.
      * **Paranoid**: Injects probabilistic false positives to stress-test logical assertions and uncover edge cases.
      * **Sensei**: Acts as a meta-agent, monitoring debate convergence and resource expenditure to terminate or finalize decision loops.

### 1.3. Deterministic Execution and Validation Layer

  * **Hallucination Probability Scoring (HPS)**: A quantitative framework to mitigate model-generated errors. The HPS is a weighted function of multiple signals:
    1.  Model confidence logits.
    2.  Structural integrity validated against the TPG.
    3.  API contract conformance.
    4.  Semantic drift from the original intent, measured via vector embedding distance.
  * **AST Snapshotting and Recovery**: Prior to any codebase mutation, the system captures an immutable snapshot of the relevant Abstract Syntax Tree (AST) and TPG sub-graph. In the event of a validation failure (e.g., CI test regression or HPS threshold breach), the system performs an atomic rollback to the pre-mutation state.

## 2\. Advanced Reasoning and Execution Modes

### 2.1. Recursive Dual-Model Adversarial Refinement

Designated "Cuckoo-in-Crow's Nest Mode," this feature employs two heterogeneous foundation models in an adversarial feedback loop. The models recursively critique and refine each other's outputs until a predefined convergence threshold is met.

  * **Convergence Criteria**: Consensus is defined by the following conditions, where $h\_i^{(T)}$ is the final hidden state of model $i$ at iteration $T$.
    $$
    $$$$\\text{Converged} \\iff
    \\begin{cases}
    \\frac{1}{N^2} \\sum\_{i,j} |h\_i^{(T)} - h\_j^{(T)}|\_2 \< \\epsilon \\
    \\text{Token-level Jaccard similarity} \> \\theta \\
    \\text{KL-divergence}(p\_i(token) || p\_j(token)) \< \\kappa \\quad \\forall i,j
    \\end{cases}
    $$
    $$$$
    $$
  * **Mail-Chess-Protocol V2 (MCP-V2)**: A state serialization protocol for maintaining context integrity across recursive iterations.
    ```protobuf
    message MCPv2Session {
      string session_id = 1;
      int32 sequence_number = 2;
      map<string, bytes> context_vectors = 3; // Serialized KV-cache
      repeated TurnHistory turn_history = 4;

      message TurnHistory {
        string model_id = 1;
        bytes input_token_ids = 2;
        bytes output_token_ids = 3;
        float confidence_score = 4;
      }
    }
    ```

### 2.2. Autonomous Orchestrator and Interactive Shell

  * **Autonomous Mode**: Executes multi-step workflows without human intervention. The loop continues until internal confidence metrics (HPS) degrade below a configurable threshold. System state and progress are broadcast via a local TCP endpoint.
  * **Interactive Shell**: A native terminal interface, implemented with POSIX tools and `ncurses`, for real-time monitoring of agent states, task execution trees, and live code diffs. No web or Electron dependencies are utilized.

## 3\. Core Engine and System-Level Integration

### 3.1. Kernel Migration to Rust/C++20

The core arbitration engine and memory management subsystems are implemented in **Rust (1.70+)** and **C++20** to achieve deterministic performance and memory safety.

  * **Rationale**:
    1.  **Guaranteed Memory Safety**: Rust's ownership model and borrow checker eliminate entire classes of memory-related undefined behavior.
    2.  **Compile-Time Contract Enforcement**: Utilizes Rust traits and C++20 concepts to enforce persona interface contracts at compile time.
    3.  **True Parallelism**: Lock-free data structures (`crossbeam`) and work-stealing thread pools (`std::jthread`) for concurrent persona arbitration.
  * **Example: C++20 Concept for Memory Regions**
    ```cpp
    template<typename T>
    concept MemoryRegion = requires(T r) {
        { r.size() } -> std::convertible_to<size_t>;
        { r.protection() } -> std::convertible_to<MemoryProtection>;
    };
    ```

### 3.2. eBPF-based OS Layer Integration (Experimental)

An experimental OS-level integration layer utilizes **eBPF (extended Berkeley Packet Filter)** for kernel-level monitoring and sandboxing of agent processes.

  * **Architecture**:
    1.  **LLM-as-Init Daemon**: A custom init process that isolates agent execution and attaches eBPF programs for syscall monitoring.
    2.  **Policy Enforcement**: Employs `LD_PRELOAD` hooks for memory allocation tracking and `ptrace` for process sandboxing. Cgroup memory limits are enforced with custom OOM handlers.
  * **Example: eBPF program for syscall interception**
    ```c
    #include <vmlinux.h>
    #include <bpf/bpf_helpers.h>

    // eBPF program attached to the tracepoint for the 'openat' syscall
    SEC("tracepoint/syscalls/sys_enter_openat")
    int handle_openat(struct trace_event_raw_sys_enter *ctx) {
        pid_t pid = bpf_get_current_pid_tgid() >> 32;
        // is_persona_process would check if pid belongs to a managed agent
        if (is_persona_process(pid)) {
            // Log file access for security auditing
            // char* filename = (char*)ctx->args[1];
            // bpf_printk("Persona PID %d opened file", pid);
        }
        return 0;
    }
    ```

## 4\. Project Status

**Current State**: Research & Development. The system is not production-ready. This document serves as a technical specification.

**Roadmap**:

  * [x] Initial fork and Spectrum Persona Protocol v1 design.
  * [ ] Implementation: Graph Indexer and ETL pipeline.
  * [ ] Implementation: AST Snapshot & Branch Manager.
  * [ ] Implementation: HPS Scoring Pipeline and Episodic Memory engine.
  * [ ] Implementation: Rust/C++ Persona Arbitration Kernel.
  * [ ] Implementation: Interactive TUI shell.

## 5\. Licensing and Contributions

  * **License**: The software is licensed under the same terms as the original Gemini CLI.
  * **Contributions**: Collaboration on the protocol and architecture is solicited via GitHub Issues and Discussions.
