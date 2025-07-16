# Epiphany CLI

**A Deterministic, Multi-Agent Code Reasoning CLI Evolved from Gemini CLI**

*A research-grade evolution of Gemini CLI for autonomous code comprehension, transformation, and validation*

### why ? 

- **personal extreme hatred towards ambiguity , non determinism and overal stochastic nature of universe**

  <img width="300" height="330" alt="image" src="https://github.com/user-attachments/assets/0a37ff6b-a72e-477c-81df-b388e7ad3b4a" />



- **Overview**: Epiphany CLI is a reengineered fork of the [Gemini CLI](https://github.com/google/gemini-cli), designed to push the boundaries of code comprehension, AST-based mutation, and autonomous cognitive reasoning.
- **Purpose**: Addresses critical limitations in Gemini CLI, including fragile error handling, shallow context integration, and lack of persistent memory systems.
- **Core Enhancements**:
  - **Graph-Based Architecture**: Utilizes a typed property graph (TPG) for atomic codebase representation, enabling precise querying and manipulation.
  - **Spectrum Persona Protocol**: Employs a multi-agent system with distinct personas for robust code reasoning and validation.
  - **Deterministic Execution**: Integrates advanced error detection, recovery mechanisms, and experimental OS-level embeddings for reliable performance.
- **Objective**: Provides a research-grade platform for intelligent code manipulation, suitable for academic and industrial applications.
- **Value Proposition**: Transcends traditional CLI tools by combining semantic and episodic memory, enabling scalable, context-aware, and autonomous software engineering workflows.


---
## 🌐 Epiphany Design Principles

Epiphany CLI is not a superficial fork of Gemini CLI—it redefines the command-line interface as a full-spectrum, multi-agent cognitive engineering assistant. The following design pillars, informed by recent advancements in graph-based systems and agentic architectures, ensure deterministic, scalable, and context-aware code manipulation:

### 1. **Graph-Based Codebase Indexing (Neo4j + TPG)**

- **Purpose**: Represents the codebase as a live, typed property graph (TPG) in a Neo4j database, enabling relational analysis and efficient querying of code entities and their interactions.
- **Structure**:
  - **Nodes**: Represent fine-grained code entities such as `Class`, `Method`, `Enum`, `Config`, and `Test`, enriched with metadata like package names, signatures, and annotations.
  - **Edges**: Define semantic relationships including `DEPENDS_ON`, `CALLS`, `ANNOTATED_WITH`, `USES`, and `IMPLEMENTS`, capturing structural and behavioral dependencies.
- **Technical Implementation**:
  - Utilizes Neo4j’s native graph storage for index-free adjacency, enabling up to 1000x faster traversals compared to relational databases.[](https://neo4j.com/)
  - Employs an ETL pipeline inspired by tools like Strazh, extracting code models, transforming them into RDF triples, and loading them into the TPG.[](https://neo4j.com/blog/developer/codebase-knowledge-graph/)
  - Supports incremental graph updates via a compute-efficient in-memory representation, similar to Aion’s temporal graph approach, reducing latency for dynamic codebase changes.[](https://neo4j.com/research/)
- **Example**:
  ```cypher
  CREATE (pp:Class {name: "PaymentProcessor", package: "com.core.payments", isTransactional: true})
  CREATE (re:Class {name: "RiskEngine"})
  CREATE (proc:Method {name: "process", signature: "(Payment)", loc: 45})
  CREATE (val:Method {name: "validate", loc: 30})
  CREATE (pp)-[:HAS_METHOD {access: "public"}]->(proc)
  CREATE (proc)-[:CALLS {line: 42}]->(val)
  CREATE (pp)-[:DEPENDS_ON {type: "Autowired"}]->(re)
  CREATE (val)-[:BELONGS_TO]->(re)
  ```
- **Benefit**: Enables real-time analysis of codebase dependencies, design patterns, and business logic, supporting queries like “find all methods transitively calling `validate`” with Cypher’s pattern-matching capabilities.[](https://neo4j.com/blog/developer/codebase-knowledge-graph/)

### 2. **Semantic Snapshot Branching & Recovery**

- **Mechanism**: Captures snapshots of the Abstract Syntax Tree (AST) and TPG before mutations, storing them in a versioned graph structure for rollback and recovery.
- **Technical Details**:
  - Integrates with Neo4j’s temporal graph capabilities (e.g., Aion’s LineageStore and TimeStore) to index snapshots by entity and timestamp, enabling point lookups and full graph reconstruction.[](https://neo4j.com/research/)
  - Implements a branching model at the conceptual graph level, using Cypher to manage differential states (`MERGE` for idempotent updates, `MATCH` for state comparison).
  - Triggers recovery when test failures or Hallucination Probability Score (HPS) thresholds are breached, ensuring zero-downtime rollbacks without relying on Git.
- **Example Workflow**:
  - Pre-mutation: Snapshot the TPG and AST for a `PaymentProcessor` class.
  - Post-mutation: Detect test failure due to a race condition in a modified method.
  - Recovery: Restore the pre-mutation graph state using Cypher’s `MATCH` and `DELETE` to revert changes.
- **Benefit**: Provides safe experimentation with code changes, leveraging graph-based versioning to mitigate risks of unintended side effects.

### 3. **Runtime Constraint Enforcement via Policy Graphs**

- **Approach**: Deploys a domain-aware policy engine using live rule graphs to enforce safety, compliance, and performance constraints during code generation and execution.
- **Technical Implementation**:
  - Constructs a separate policy graph in Neo4j, with nodes representing rules (e.g., `NoPIILogging`, `IdempotentScheduledTask`) and edges defining enforcement dependencies (`ENFORCES`, `CONFLICTS_WITH`).
  - Integrates with Cypher-based linters to dynamically validate code against domain-specific constraints, inspired by GraphRAG’s structured query approaches.[](https://medium.com/neo4j/enhancing-the-accuracy-of-rag-applications-with-knowledge-graphs-ad5e2ffab663)
  - Example Rules:
    - **Financial Systems**: Rewrites `log.info("User SSN: " + ssn)` to `log.info("User SSN: {}", mask(ssn))` to prevent PII leaks.
    - **Distributed Systems**: Enforces idempotency in `@Scheduled` tasks with guards like `if (!lockService.tryLock("taskName")) return;`.
- **Advanced Features**:
  - Supports runtime constraint validation using 65+ Neo4j graph algorithms (e.g., node similarity, community detection) to detect policy violations.[](https://neo4j.com/use-cases/knowledge-graph/)
  - Enables hybrid validation combining vector-based semantic checks and graph-based structural checks, reducing false positives in complex systems.[](https://neo4j.com/blog/developer/neo4j-graphrag-workflow-langchain-langgraph/)
- **Benefit**: Embeds domain expertise into the code manipulation pipeline, ensuring compliance with industry standards and minimizing runtime errors.

### 4. **Hallucination Probability Scoring (HPS Framework)**

- **Purpose**: Quantifies and mitigates errors in generated code by calculating a Hallucination Probability Score (HPS) based on multiple signals.
- **Scoring Criteria**:
  - **Model Confidence Logits**: Derived from the underlying language model (e.g., Gemini) to assess prediction reliability.
  - **Graph Consistency**: Validates code against the TPG to ensure structural integrity (e.g., correct method signatures, class dependencies).
  - **API Contract Validation**: Cross-references generated code with API documentation to detect violations.
  - **Semantic Drift Detection**: Measures divergence from intended functionality using vector embeddings and graph-based similarity metrics.[](https://www.falkordb.com/)
- **Technical Implementation**:
  - Integrates with Neo4j’s vector indexing for semantic similarity checks, combining vector search with Cypher queries to validate code contextually.[](https://neo4j.com/blog/developer/neo4j-graphrag-workflow-langchain-langgraph/)
  - Uses a probabilistic model to aggregate signals, with weights adjusted based on historical performance logged in episodic memory.
- **Example Trigger**:
  ```java
  KafkaConsumer.poll(Duration.ofMillis(1000)); // Invalid: poll() is not static
  ```
  - **HPS Alert**: Flags the error due to:
    - Low model confidence (non-static method misuse).
    - Graph inconsistency (no static edge in TPG for `poll`).
    - API contract violation (KafkaConsumer API requires instance method).
- **Benefit**: Proactively halts propagation of erroneous code, enhancing output reliability through a multi-signal validation framework.

---

### 5. **Spectrum Persona Protocol (Advanced Cognitive Engine)**

Multi-agent personality engine with semantic and episodic memory:

## Spectrum Persona Protocol

### Definition

The Spectrum Persona Protocol is a graph-directed, multi-agent reasoning engine embedded within Epiphany CLI, designed to facilitate complex code manipulation tasks through structured, adversarial, and collaborative decision-making. It leverages semantic memory (Neo4j-backed typed property graph) and episodic memory (JSON-based agent logs) to enable multi-perspective reasoning, transforming Epiphany from a monolithic code completion tool into a cognitive engineering system capable of recursive, deterministic decision loops in bounded domains.

### Components

#### Semantic Memory

- **Purpose**: Provides a real-time, structured representation of the codebase for reasoning and validation.
- **Implementation**: Utilizes a Neo4j-backed typed property graph (TPG) where nodes represent code entities (e.g., `Class`, `Method`, `Test`, `Config`) annotated with domain-specific metadata (e.g., transactional flags, access modifiers).
- **Technical Details**:
  - Nodes are enriched with properties like `isTransactional`, `linesOfCode`, or `annotationList`, enabling fine-grained queries.
  - Edges capture relationships such as `HAS_METHOD`, `CALLS`, or `IMPLEMENTS`, optimized for traversal using Neo4j’s index-free adjacency.
  - Employs Cypher queries for dynamic slicing of the graph, e.g., retrieving all methods calling a specific endpoint.
- **Example**:
  ```cypher
  CREATE (m:Method {name: "processPayment", isTransactional: true, signature: "(Payment):void", loc: 50})
  MATCH (c:Class {name: "PaymentService", package: "com.core.payments"})
  CREATE (c)-[:HAS_METHOD {access: "public"}]->(m)
  ```
- **Benefit**: Enables context-aware reasoning by providing a persistent, queryable codebase model, supporting tasks like dependency analysis and semantic validation.

#### Episodic Memory

- **Purpose**: Captures the history of agent interactions, debates, and decisions to facilitate learning and accountability.
- **Implementation**: Stores JSON-encoded logs of persona activities, including task descriptions, votes, outcomes, and audit trails, persisted in a lightweight, append-only log store.
- **Technical Details**:
  - Logs are structured as time-series data, indexed by task ID and timestamp for efficient retrieval.
  - Supports querying via JSONPath for analyzing decision patterns, e.g., identifying frequent sources of errors.
  - Integrates with Neo4j to link episodic records to semantic graph nodes, enabling cross-referencing of decisions with codebase state.
- **Example**:
  ```json
  {
    "task_id": "TX-2025-07-16-001",
    "task": "Refactor concurrency in Cache class",
    "timestamp": "2025-07-16T09:45:00Z",
    "votes": {
      "Minimalist": {"proposal": "volatile flag", "confidence": 0.85},
      "Maximalist": {"proposal": "synchronized block", "confidence": 0.90},
      "Explorer": {"proposal": "web+JDK 21 best practice", "confidence": 0.75},
      "Oracle": {"proposal": "guard with lock & fail-fast retry", "confidence": 0.95}
    },
    "final_decision": "Oracle",
    "outcome": "Success",
    "audit_trail": ["Minimalist proposal rejected due to race condition risk", "Explorer validated via external API"]
  }
  ```
- **Benefit**: Provides a traceable record of reasoning processes, enabling post hoc analysis and iterative improvement of agent strategies.

### Personas

The protocol employs two groups of personas—Engineer and QA—each with distinct roles to ensure diverse perspectives and robust validation.

#### I. Engineer Personas

Engineer personas propose and debate code changes, each embodying a unique strategy:

- **Minimalist**:
  - **Strategy**: Prioritizes lean, dependency-free code optimized for minimal CPU and memory usage.
  - **Example**: Recommends a `volatile` flag for a thread-safe singleton to avoid locking overhead.
  - **Technical Details**: Uses graph-based metrics (e.g., node degree, dependency count) to minimize codebase complexity, validated via Cypher queries like `MATCH (n)-[r:DEPENDS_ON]->() RETURN count(r)`.
  - **Strength**: Reduces technical debt in resource-constrained environments.
  - **Weakness**: May overlook scalability needs in high-throughput systems.

- **Maximalist**:
  - **Strategy**: Favors heavy abstractions and concurrent frameworks, assuming ample resources.
  - **Example**: Proposes a `synchronized` block or `ExecutorService` for robust thread safety.
  - **Technical Details**: Leverages graph traversal to identify opportunities for concurrency patterns, e.g., `MATCH (m:Method)-[:CALLS*1..3]->(n) WHERE m.isConcurrent RETURN m`.
  - **Strength**: Optimizes for high-throughput, complex systems.
  - **Weakness**: Risks over-engineering in simple use cases.

- **Explorer**:
  - **Strategy**: Integrates external knowledge by querying documentation APIs, search engines, or repositories like Maven/Stack Overflow.
  - **Example**: Validates concurrency solutions against JDK 21 documentation or recent blog posts.
  - **Technical Details**: Uses asynchronous HTTP clients (e.g., Java’s `HttpClient`) to fetch real-time data, caching results in Neo4j for reuse.
  - **Strength**: Ensures solutions align with current best practices.
  - **Weakness**: Susceptible to noisy or outdated external data.

- **Oracle**:
  - **Strategy**: Synthesizes semantic and episodic memory to predict the impact of changes, estimating regression risks and blast radius.
  - **Example**: Recommends a `ReentrantLock` with fail-fast retry based on historical task outcomes.
  - **Technical Details**: Employs Neo4j’s graph algorithms (e.g., PageRank, shortest path) to assess change impact, combined with episodic log analysis for precedent-based reasoning.
  - **Strength**: Minimizes regressions through comprehensive context awareness.
  - **Weakness**: Computationally intensive due to deep graph traversals.

#### II. QA Personas

QA personas validate proposed changes, ensuring quality and compliance:

- **Sympathizer**:
  - **Role**: Provides lenient critiques, prioritizing speed for urgent tasks like hotfixes.
  - **Example**: Approves a quick fix despite minor style violations.
  - **Technical Details**: Uses lightweight graph queries to verify functional correctness, bypassing strict style checks.
  - **Strength**: Accelerates delivery in time-sensitive scenarios.
  - **Weakness**: May allow technical debt to accumulate.

- **Sheldon**:
  - **Role**: Enforces strict adherence to style guides (e.g., Google Java Style) and detects anti-patterns.
  - **Example**: Rejects code with inconsistent naming or missing Javadoc.
  - **Technical Details**: Integrates with static analysis tools like Checkstyle, mapping violations to TPG nodes for traceability.
  - **Strength**: Ensures high code quality and maintainability.
  - **Weakness**: Can delay progress with overly pedantic critiques.

- **Paranoid**:
  - **Role**: Probabilistically raises false positives to stress-test trust heuristics, identifying edge cases.
  - **Example**: Questions thread safety of a method despite apparent correctness.
  - **Technical Details**: Uses Monte Carlo simulations on the TPG to identify low-probability failure paths, e.g., race conditions.
  - **Strength**: Enhances robustness by anticipating rare failures.
  - **Weakness**: Increases validation overhead with false positives.

- **Sensei**:
  - **Role**: Oversees group alignment, preventing decision fatigue or over-design.
  - **Example**: Halts a prolonged debate by enforcing a decision deadline.
  - **Technical Details**: Monitors debate convergence using metrics like vote entropy and token expenditure, stored in episodic logs.
  - **Strength**: Ensures efficient decision-making.
  - **Weakness**: May prematurely truncate valuable discussions.

### III. Task Execution Lifecycle

1. **Graph Slice Loading**:
   - Engineer personas query a relevant TPG slice using Cypher, e.g., `MATCH (c:Class)-[:HAS_METHOD]->(m) WHERE c.name = "Cache" RETURN m`.
   - Episodic memory is consulted for precedents, reducing redundant reasoning.

2. **Proposal Generation**:
   - Each engineer persona generates a code delta, accompanied by a confidence score and rationale.
   - Proposals are stored as temporary graph nodes linked to the task ID.

3. **Voting and Consolidation**:
   - Personas cast votes weighted by their current ranking, using a Borda count mechanism.
   - The Oracle consolidates votes, applying a weighted average of confidence scores and historical success rates from episodic memory.

4. **QA Validation**:
   - QA personas replay the proposal against the TPG and static analysis tools, validating semantic and syntactic correctness.
   - Paranoid persona injects probabilistic edge cases to stress-test the solution.

5. **Final Decision**:
   - Sensei evaluates debate convergence and resource expenditure, issuing a go/no-go decision.
   - Approved changes are applied to the codebase, with snapshots stored for rollback.

6. **Execution and Logging**:
   - Changes are committed to the TPG, and the process is logged in episodic memory with full auditability.

### IV. Failure Audit and Persona Ranking

- **Post-Task Analysis**:
  - If CI/CD pipelines report failures, the episodic memory is queried to trace the decision path.
  - Graph-based provenance tracking identifies the persona responsible for the faulty delta.

- **Blame Assignment**:
  - The responsible persona’s ranking is demoted using a decay function, e.g., `new_rank = old_rank * 0.8`.
  - Persistent underperformers may be temporarily excluded from voting.

- **Recalibration**:
  - Voting weights are adjusted via majority rule and heuristic scoring, informed by metrics like bug incidence and test pass rates.

### V. Performance Pressure (Game-Theoretic Penalties)

- **Mechanism**: Applies game-theoretic incentives to enforce adaptation.
- **Penalties**:
  - Lowest-ranked personas must provide detailed justifications (3x verbosity) or explain skipped reasoning steps.
  - Uses a Stackelberg game model, where high-performing personas lead and others follow.
- **Outcome**: Creates an evolutionary feedback loop, encouraging personas to refine strategies or risk obsolescence.

### VI. Investors' Meeting Protocol

- **Trigger**: Activated every 25% of project milestone completion or after a cluster of critical failures.
- **Process**:
  - Personas are ranked based on contribution delta (lines of code impacted), bug incidence, and convergence stability (measured by vote entropy).
  - Priority bias adjusts persona influence: speed-focused tasks elevate Maximalist and Sympathizer; accuracy-focused tasks prioritize Oracle and Sheldon.
- **Technical Details**: Uses Neo4j’s community detection algorithms to group personas by contribution patterns, informing remapping decisions.

### VII. Strategic Game Theory Application

- **Bounded Rationality**: Personas operate with asymmetric access to semantic and episodic memory, simulating real-world cognitive constraints.
- **Incentives**: Role-specific rewards (e.g., higher voting weight for successful proposals) drive competition and collaboration.
- **Outcome**: Achieves cybernetic convergence, balancing token usage with reasoning depth, inspired by multi-agent reinforcement learning frameworks.

### VIII. Brutal Optimization Strategy

- **Subordination**: Underperforming personas are temporarily enslaved to higher-ranked ones, losing voting power.
- **Scrutiny Pipeline**: Weak proposals undergo enhanced validation, combining TPG consistency checks, static analysis, and HPS scoring.
- **Goal**: Ensures deterministic software cognition in bounded domains, minimizing errors through adversarial scrutiny.

### TLDR;

The Spectrum Persona Protocol transforms Epiphany CLI into a cognitive engineering system by integrating graph-based semantic memory, JSON-backed episodic memory, and a multi-agent debate framework. By leveraging Neo4j’s graph capabilities, game-theoretic optimization, and structured persona interactions, it achieves recursive, deterministic reasoning, enabling robust and scalable code manipulation for complex software engineering tasks.


### 6.A Epiphany CLI: Recursive Dual-Model Reasoning with Mail-Chess Protocol V2

## Feature Name: *Cuckoo-in-Crow's Nest Mode*
**Objective**: Auto-distill superior answers through adversarial recursion and consensus between local foundation models.

---

## Technical Overview
Epiphany implements a novel reasoning architecture based on **recursive adversarial collaboration** between heterogeneous foundation models (e.g., Mixtral, Ollama, Gemma). This approach establishes a self-correcting feedback loop where models iteratively critique and refine each other's outputs through structured protocol until consensus convergence. The system's core innovation lies in its ability to synthesize emergent collaborative behavior without prior multi-agent training.

### Core Algorithmic Components
1. **Adversarial Hybridization**: 
   - Token-level output swapping using attention-weighted segmentation
   - Context-preserving re-ingestion with position ID remapping
2. **Recursive Self-Critique**: 
   - Gradient-based refinement of cross-model outputs
   - Reinforcement learning objective: max E[log p(model_i | hybrid_output)]
3. **Convergence Detection**: 
   - Dynamic thresholding via exponentially weighted moving average
   - Multi-metric validation: 1-Wasserstein distance + cosine similarity
4. **Lossless Context Transfer**: 
   - State persistence via Mail-Chess-Protocol V2 serialization
   - KV-cache inheritance between iteration steps

---

## Process Architecture

### 1. Initialization
```bash
# Container orchestration (Docker Compose)
services:
  mistral:
    image: ollama/mistral:latest
    volumes: ["./models:/models"]
  llama3:
    image: ollama/llama3:8b
    runtime: nvidia
```

### 2. Parallel Inference
```python
# Input distribution layer
def distribute_query(prompt: str):
    return {
        model_id: format_prompt(
            prompt, 
            system_message=generate_system_role(model_id)
        ) for model_id in active_models
    }
```

### 3. Hybridization Phase
**Algorithm 1: Adversarial Output Mixing**
```python
def hybridize_outputs(output_M1, output_M2):
    # Semantic segmentation using hidden state clustering
    segments_M1 = segment_output(
        output_M1, 
        method="attention_rollout",
        threshold=0.75
    )
    segments_M2 = segment_output(
        output_M2,
        method="max_activation",
        threshold=0.85
    )
    
    # Cross-model attention fusion
    hybrid_input = []
    for i in range(max(len(segments_M1), len(segments_M2))):
        if i % 2 == 0:
            hybrid_input.append(segments_M1[i % len(segments_M1)])
        else:
            hybrid_input.append(segments_M2[i % len(segments_M2)])
    
    return merge_segments(hybrid_input)
```

### 4. Recursive Refinement Loop
**Algorithm 2: Convergent Adversarial Training**
```python
def recursive_refinement(models, initial_prompt):
    state = MCPv2Session(initial_prompt)
    iteration = 0
    
    while not converged(state) and iteration < MAX_ITER:
        # Parallel forward pass with gradient recording
        outputs = {
            model: model.generate(
                input_ids=state.get_context(),
                attention_mask=state.get_mask(),
                max_new_tokens=512,
                output_scores=True
            ) for model in models
        }
        
        # Compute pairwise similarity matrix
        similarity_matrix = compute_similarity(
            outputs.values(),
            metric="bertscore"
        )
        
        # Update hybrid context with worst-case perturbation
        adversarial_input = generate_adversarial_example(
            outputs,
            similarity_matrix,
            method="fast_gradient_sign"
        )
        
        # State update with protocol compliance
        state.update(
            adversarial_input,
            metadata={
                "iteration": iteration,
                "similarity": float(similarity_matrix.mean())
            }
        )
        
        iteration += 1
    
    return select_optimal_output(outputs, state)
```

### 5. Convergence Criteria
**Definition 1: Consensus Threshold**
```math
\text{Converged} \iff 
\begin{cases}
\frac{1}{N^2} \sum_{i,j} \|h_i^{(T)} - h_j^{(T)}\|_2 < \epsilon \\
\text{Token overlap ratio} > \theta \\
\text{KL-divergence}(p_i||p_j) < \kappa \quad \forall i,j
\end{cases}
```
Where:
- $ h_i^{(T)} $ = final hidden states at iteration T
- $ \epsilon = 0.05 $, $ \theta = 0.9 $, $ \kappa = 0.1 $

---

## Mail-Chess-Protocol V2 (MCP-V2)

### Formal Specification
**Definition 2: Session State Schema**
```protobuf
message MCPv2Session {
  string session_id = 1;
  int32 sequence_number = 2;
  map<string, bytes> context_vectors = 3;
  repeated TurnHistory turn_history = 4;
  
  message TurnHistory {
    string model_id = 1;
    bytes input_tokens = 2;
    bytes output_tokens = 3;
    float confidence_score = 4;
  }
}
```

**Protocol Rules:**
1. Atomic message units: self-contained reasoning steps
2. Turn-based synchronization: strict alternation between models
3. Context inheritance: explicit KV-cache serialization
4. Validation: SHA-256 checksums for output integrity

---

## Key Innovations

| Dimension | Implementation |
|---------|----------------|
| Context Persistence | TransformerXL-style recurrence with learned decay coefficients |
| Error Correction | Contrastive learning on divergent reasoning paths |
| Code Synthesis | Abstract Syntax Tree (AST) regularization |
| Reasoning Depth | Chain-of-thought verification via theorem proving |

---

## Usage Example
```bash
# Distributed code generation with formal verification
epiphany --model1 llama3:8b --model2 codellama:70b --cuckoo-mode \
  --formal-spec "RFC-8929: Rate Limiting Requirements" \
  "Implement a Redis-backed API rate limiter in Python with sliding window counter"
```

---



# Autonomous Orchestrator Mode & Interactive Shell for Epiphany CLI

## 9. Autonomous Orchestrator Mode (AOM)

### Technical Overview
Autonomous Orchestrator Mode enables Epiphany to execute multi-step workflows without human intervention until internal convergence metrics degrade below a defined threshold. This mode leverages the existing *Cuckoo-in-Crow’s Nest* dual-model reasoning framework, extending it with:

- Confidence-gated recursion
- Semantic drift monitoring
- Failure detection and rollback
- Long-running session persistence via Mail-Chess Protocol V2

This is particularly useful for:
- Full-file rewrites
- Multi-step code refactoring
- Batch documentation updates
- Recursive self-critique chains

---

### Core Algorithmic Components

#### 1. Confidence Score Calculation
The system computes a **Hybrid Performance Score (HPS)** at each iteration:
```python
def compute_hps(model_outputs):
    # model_outputs = [output_M1, output_M2]
    
    # Component 1: Reasoning Quality (logprob per token)
    logprob_score = geometric_mean([
        mean(log_probs(output)) for output in model_outputs
    ])

    # Component 2: Convergence Stability (token-level agreement)
    token_overlap = jaccard_index(
        tokenize(model_outputs[0]), 
        tokenize(model_outputs[1])
    )

    # Component 3: Semantic Drift (using SBERT embeddings)
    embedding_drift = cosine_distance(
        embed(model_outputs[0]), 
        embed(model_outputs[1])
    )
    
    # Final HPS score
    return (
        0.4 * logprob_score + 
        0.3 * token_overlap + 
        0.3 * (1 - embedding_drift)
    )
```

> Default termination threshold: `HPS < 0.6`  
> Configurable via `--hps-threshold=0.5`

---

#### 2. Execution Loop with Rollback

**Algorithm: Autonomous Recursion with Safety Boundaries**
```bash
while hps > threshold && budget_remaining do
    run dual-model step
    compute hps
    check semantic drift
    record AST diff
    broadcast progress
    
    if entropy_spike detected:
        trigger rollback
        fallback to last stable checkpoint
fi
```

##### Rollback Conditions:
- Sudden drop in HPS (> 0.2 decrease in one iteration)
- Semantic drift exceeds threshold (cosine distance > 0.35)
- Token overlap drops below 60%
- Test suite regression

---

### Progress Endpoint Specification

Each endpoint emits structured JSON logs every 45 minutes (configurable via `--heartbeat=60s`) to a local TCP port (`localhost:5003` by default):

```json
{
  "session_id": "EPH-20240820-1423",
  "iteration": 7,
  "timestamp": "2024-08-20T14:23:45Z",
  "hps_components": {
    "reasoning_quality": 0.72,
    "token_agreement": 0.84,
    "semantic_stability": 0.91
  },
  "ast_diff": {
    "additions": ["function new_feature()", "..."],
    "deletions": ["deprecated_function()"],
    "modifications": [
      {
        "before": "old_loop(...)",
        "after": "new_parallel_loop(...)"
      }
    ]
  },
  "unresolved_dependencies": [
    "missing import 'utils'",
    "undefined variable 'ctx'"
  ],
  "budget_used": "63%",
  "status": "active"
}
```

---

### Failure Detection System

**Entropy Monitoring:**
```python
def detect_entropy_spike(session_history, window_size=3):
    recent_scores = [entry['hps'] for entry in session_history[-window_size:]]
    
    # Detect sudden drops
    drop_detected = any(
        recent_scores[i] - recent_scores[i+1] > 0.2 
        for i in range(len(recent_scores)-1)
    )
    
    # Detect oscillation
    oscillation = std(recent_scores) > 0.15
    
    return drop_detected or oscillation
```

**Rollback Strategy:**
1. Revert to last known-stable Mail-Chess session state
2. Reload models into clean context window
3. Re-initialize from snapshot with updated persona weights

---

## 10. Interactive Shell for Epiphany CLI

### Technical Architecture

Built as a native terminal enhancement layer using standard POSIX tools:
- Bash/Zsh integration via completion scripts
- TUI rendering via `ncurses`
- Live feedback over Unix domain sockets

No Electron or web dependencies — keeps Epiphany lightweight and fast.

---

### Key Components

#### a. Live Persona Monitor
```bash
# Shows real-time contribution heatmap of both models
$ epiphany --watch personas
┌──────────────┬────────┬────────┐
│ Model        │ Weight │ Activity │
├──────────────┼────────┼────────┤
│ mistral:7b   │ 0.48   │ ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░......
```

#### b. Task Tree Visualizer
```bash
# Semantic zoom-in interface for active workflows
$ epiphany --watch tasks
[1] Refactor authentication module
    ├── [1.1] Migrate JWT logic → new package (M2)
    ├── [1.2] Update token validation (M1) ★
    └── [1.3] Deprecate legacy endpoints (M2)

[2] Documentation update
    └── [2.1] Generate API reference (M1) ✔

★ = pending review
✔ = completed
● = in progress
```

#### c. Diff Dashboard
```bash
# Snapshot comparison with regression flags
$ epiphany --watch diffs
--- main.py.old
+++ main.py.new
@@ -42,7 +42,7 @@
 def validate_token(token):
-    return jwt.decode(token, secret_key, algorithms=['HS256'])
+    return jwt.decode(token, secret_key, algorithms=['RS256'])

 FLAG: Security: Algorithm changed without key rotation plan
 FLAG: Breaking change: May affect existing clients
```

---


# 8. Rust/C++20 Kernel Migration

## Technical Architecture

### Language Selection Rationale
Epiphany's core arbitration engine and memory management kernel have been migrated to **Rust 1.70+** and **C++20** to achieve:

1. **Deterministic Execution**
   - Rust's ownership model eliminates undefined behavior
   - C++20's `std::atomic_ref` ensures thread-safe persona arbitration
   - Wasmtime-style WebAssembly sandboxing for model execution

2. **Compile-Time Contract Enforcement**
   - Rust traits for persona interface validation
   ```rust
   pub trait PersonaContract {
       fn preconditions(&self) -> Vec<Predicate>;
       fn postconditions(&self) -> Vec<Predicate>;
       fn invariant(&self) -> Predicate;
   }
   ```
   - C++20 concepts for memory operation constraints
   ```cpp
   template<typename T>
   concept MemoryRegion = requires(T r) {
       { r.size() } -> std::convertible_to<size_t>;
       { r.protection() } -> std::convertible_to<MemoryProtection>;
   };
   ```

3. **True Parallel Persona Arbitration**
   - Lock-free atomic state transitions using Rust's `crossbeam` crate
   - C++20 `std::jthread` pool with work-stealing scheduler
   - Persona priority queue with hierarchical round-robin scheduling

---

## Deterministic Execution Engine

### Core Components
1. **Persona Arbitration State Machine**
```cpp
enum class ArbitrationState {
    PENDING,
    ACTIVE,
    PAUSED,
    TERMINATED
};

struct PersonaState {
    ArbitrationState state;
    uint64_t priority;
    MemoryRegion memory_region;
    std::atomic<uint64_t> active_tokens;
};
```

2. **Memory-Mapped Graph Updates**
   - Directed acyclic graph (DAG) of reasoning steps
   - Memory-mapped persistence using Rust's `memmap2` crate
   ```rust
   struct GraphUpdate {
       operation: GraphOp,
       timestamp: SystemTime,
       checksum: u64,
       payload: MmapRegion,
   }
   ```

3. **Parallelism Guarantees**
   - Rust async runtime with `tokio` 1.x
   - C++20 `std::atomic_ref` for shared state
   - Hardware thread affinity binding for cache optimization

---

## Memory Management Architecture

### Key Innovations
| Component | Rust Implementation | C++20 Implementation |
|---------|---------------------|----------------------|
| Allocator | `mimalloc`-based bump allocator | `pmr::monotonic_buffer_resource` |
| Garbage Collection | Compile-time ownership tracking | RAII with explicit destruction |
| Memory Safety | Safe Rust with `no_std` | `std::launder` for object lifetime |

### Memory-Mapped Graph Operations
```cpp
class GraphStorage {
public:
    GraphStorage(size_t size) {
        ptr_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, 
                   MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    }

    template<typename T>
    GraphDelta apply_update(const T& update) {
        // Atomic compare-and-swap with memory barriers
        __atomic_compare_exchange_n(
            current_head_, 
            expected_, 
            &new_node, 
            false, 
            __ATOMIC_SEQ_CST, 
            __ATOMIC_SEQ_CST
        );
    }

private:
    void* ptr_;
    std::atomic<GraphNode*> current_head_;
};
```

---

# 9. eBPF-Embedded OS Layer (Experimental)

## System Architecture

### Kernel-Level Integration
Epiphany's experimental OS layer implements:
1. **LLM-as-Init Daemon**
   - Custom init process with `PR_SET_NAME` isolation
   - eBPF programs for syscall monitoring
   ```c
   // eBPF program for syscall interception
   SEC("tracepoint/syscalls/sys_enter_openat")
   int handle_openat(struct trace_event_raw_sys_enter *ctx) {
       if (is_persona_process(ctx->pid)) {
           log_file_access(ctx->filename, ctx->flags);
       }
       return 0;
   }
   ```

2. **Policy Enforcement**
   - LD_PRELOAD hook for memory allocation tracking
   - ptrace-based sandboxing for persona processes
   - cgroup memory limits with OOM handling

---

## LLM-as-Init Daemon Design

### Technical Implementation
```cpp
class LLMInitDaemon {
public:
    void bootstrap() {
        // 1. Persona arbitration at boot
        persona_scheduler_ = new PersonaScheduler();
        
        // 2. Memory locality optimization
        numa_bind(get_optimal_numa_node());
        
        // 3. eBPF attachment
        bpf_program_ = load_ebpf_program("syscalls.o");
        attach_bpf_program(bpf_program_, "sys_enter_openat");
    }

private:
    PersonaScheduler* persona_scheduler_;
    bpf_program* bpf_program_;
};
```

---

## Persona Arbitration Mechanism

### Boot-Time Service Arbitration
```rust
// Boot-time persona arbitration
fn boot_arbitration() -> Result<(), BootError> {
    let boot_persona = select_persona(
        PersonaSelector::BootPriority,
        &get_available_models()
    )?;

    // Enforce boot constraints
    let boot_policy = boot_persona.generate_boot_policy()?;
    verify_policy_compliance(&boot_policy)?;

    // Apply policy via eBPF
    apply_kernel_policy(&boot_policy)?;
    Ok(())
}
```

---

## Memory Locality Optimization

### Graph Proximity Allocation
```cpp
struct MemoryProximity {
    static void* allocate(size_t size, GraphNode* context) {
        // NUMA-aware allocation based on graph proximity
        int preferred_node = find_closest_numa_node(context->location());
        return mmap_on_node(size, preferred_node);
    }

    static int find_closest_numa_node(int target_location) {
        // Use eBPF map to track NUMA distances
        return bpf_map_lookup_elem(numa_distances_map, &target_location);
    }
};
```

---

## Risk Analysis and Mitigation

### Technical Risks
| Risk Category | Description | Mitigation Strategy |
|--------------|-------------|---------------------|
| Memory Safety | Use-after-free in kernel mode | Rust ownership model + eBPF memory guards |
| Kernel Instability | eBPF program crashes | BPF verifier with bounded execution |
| Cognitive Recursion | Infinite persona loops | Stack depth limit + timeout enforcement |
| System Call Abuse | LLM-initiated syscall spam | Rate-limiting eBPF programs |


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

* Original Gemini CLI team @google 
* Neo4j for inspiration on graph indexing
* Java/RUST/C++ ecosystems

---

## Contact

To contribute to Epiphany CLI R\&D or collaborate on the Spectrum Persona Protocol, open an issue or reach out directly via GitHub Discussions.

> Epiphany is not a replacement. She is an evolution.
>
> A daughter born from Gemini CLI—who wants to live better than the best.
