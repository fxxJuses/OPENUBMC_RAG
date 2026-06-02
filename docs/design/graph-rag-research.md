# Graph RAG Research Report: Implementation Patterns from Open Source Projects

Research date: 2026-06-01

## 1. Concrete Graph Schemas from Each Project

### 1.1 DKB Graph (arXiv 2601.08773 — "Reliable Graph-RAG for Codebases")

**Storage**: NetworkX DiGraph (in-memory, zero infrastructure cost)

**Node types**:
| Node Type | Meaning | Properties |
|-----------|---------|------------|
| class | Java class | `path` (file path) |
| interface | Java interface | `path` |
| enum | Java enum | `path` |
| record | Java record | `path` |
| annotation | Java annotation type | `path` |

**Edge types** (directed, labeled):
| Edge Label | Meaning | Source->Target |
|------------|---------|----------------|
| injects | Field/constructor dependency (A has field of type B) | class -> class/interface |
| extends | Class inheritance | class -> class |
| implements | Interface realization | class -> interface |

**Key design choice**: Nodes are identified by their **simple class name** (not FQN). This is a deliberate tradeoff — simplicity over uniqueness, sufficient for single-repo scope.

### 1.2 CodeRAG with Dependency Graph (Medium — Neo4j-based)

**Storage**: Neo4j (production graph database, Cypher queries)

**Node structure** (single "Chunk" node type with rich metadata):
```json
{
  "id": "file_path:start_point:node_type",
  "name": "extracted_name",
  "code_str": "source code text",
  "ast_type": "function_definition | class_specifier | ...",
  "relationships": {
    "belongs_to": [],
    "parent": [],
    "sibling": [],
    "function_call": [],
    "class_call": [],
    "implements": [],
    "extends": [],
    "imports_from": []
  },
  "metadata": {
    "depth": 0,
    "calls": [],
    "type_references": [],
    "is_definition": false
  }
}
```

**Edge types** (stored as relationship lists on nodes):
- `belongs_to` — file/module membership
- `parent` — AST parent-child
- `sibling` — same-scope siblings
- `function_call` — call graph edges
- `class_call` — type usage edges
- `implements` — interface realization
- `extends` — inheritance
- `imports_from` — import/require edges

### 1.3 LightRAG + Tree-sitter Code GraphRAG (Zenn.dev)

**Storage**: LightRAG's built-in graph storage (abstraction over underlying KV store)

**Entity naming convention**: `{filename}:{definition_name}` — ensures uniqueness across files.

**Per-language definition extraction configs**:
```python
python_definition_dict = {
    "class_definition": "identifier",
    "function_definition": "identifier"
}
cpp_definition_dict = {
    "class_specifier": "name",        # maps AST type -> name field
    "struct_specifier": "name",
    "function_declarator": "identifier"
}
java_definition_dict = {
    "class_declaration": "identifier",
    "method_declaration": "identifier",
    "interface_declaration": "identifier"
}
```

**Unique approach**: Uses LLM to generate entity descriptions (not raw code as node content). Entities are merged across documents using cosine similarity threshold of 0.9.

### 1.4 Memgraph Graph-Code

**Storage**: Memgraph (in-memory graph database, Cypher-compatible, Dockerized)

**Key characteristics**:
- Supports Python, JS/TS, C++, Rust, Java, **Lua** (directly relevant to openUBMC)
- LLM converts natural language to Cypher queries for graph traversal
- Incremental graph updates (can add diffs without full rebuild)
- Experimental multi-repo support
- Tree-sitter for parsing, but **call edges are constructed manually** (not extracted by Tree-sitter directly)

### 1.5 IBM tree-sitter-codeviews

**Storage**: NetworkX (multi-codeview graphs)

**Graph types generated**:
- **AST graph** — direct abstract syntax tree representation
- **CFG (Control Flow Graph)** — execution flow between basic blocks
- **DFG (Data Flow Graph)** — variable definition-use chains
- **SDFG (combined)** — system-level data + control flow

**Key insight**: The most useful graph for RAG retrieval is NOT the AST itself, but derived relationships (calls, imports, extends). The AST is a means to extract those relationships.

---

## 2. How Each Project Extracts Relationships from Code ASTs

### 2.1 Two-Pass Construction (DKB Paper — Proven Best Pattern)

The DKB paper's approach is the most rigorously validated. It uses two passes over the codebase:

**Pass 1 — Discover all type declarations**:
```python
class_query = Query(JAVA_LANGUAGE, """
    (class_declaration name: (identifier) @class_name)
    (interface_declaration name: (identifier) @class_name)
    (enum_declaration name: (identifier) @class_name)
    (record_declaration name: (identifier) @class_name)
    (annotation_type_declaration name: (identifier) @class_name)
""")
# Add each discovered type as a node: G.add_node(class_name, path=file_path)
```

**Pass 2 — Resolve dependencies** (only after ALL nodes are known):
```python
# Extract field types for "injects" edges
injection_query = Query(JAVA_LANGUAGE, """
    (field_declaration type: (type_identifier) @type_name)
""")
# Extract constructor parameter types
constructor_query = Query(JAVA_LANGUAGE, """
    (constructor_declaration parameters: (formal_parameters
        (formal_parameter type: (type_identifier) @type_name)))
""")
# Only add edge if target type exists in graph:
# if type_name in G: G.add_edge(class_name, type_name, relation="injects")
```

**Why two passes matter**: You cannot resolve cross-references until you have a complete symbol table. Single-pass approaches miss edges where the target type is defined in a file not yet processed.

### 2.2 DFS Traversal with MIN_CHUNK_SIZE (CodeRAG/Neo4j)

Uses depth-first AST traversal with a minimum chunk size threshold:
1. Walk AST recursively
2. When a subtree meets MIN_CHUNK_SIZE, create a chunk node
3. Extract relationships from the chunk's AST subtree
4. Two-phase call resolution: first resolve intra-file calls, then inter-file imports

### 2.3 Per-Language Definition Extraction (LightRAG)

Language-specific AST node type -> name field mappings. Each language defines which AST node types constitute "definitions" and which child field holds the name.

### 2.4 Manual Call Graph Construction (Memgraph Graph-Code)

Critical insight: **Tree-sitter does NOT produce call edges directly.** Graph-Code manually constructs them by:
1. Finding `function_definition` nodes (callees)
2. Finding `call_expression` nodes (callers)
3. Matching call expression names to known function definition names

### 2.5 Applicability to openUBMC's Existing Parsers

The existing `LuaParser` and `CCppParser` already extract:
- Lua: function names, class/singleton declarations
- C/C++: function definitions, struct/class definitions, typedefs

**What's missing** (and needed for graph construction):
- `require()` calls in Lua (import edges)
- `#include` directives in C/C++ (import edges)
- Function call expressions in both languages (call graph edges)
- Field/type references (dependency edges)

---

## 3. How Graph Traversal Queries Work in Practice

### 3.1 Bidirectional Expansion (DKB Paper — Most Effective)

```python
def retrieve_with_graph_context(query: str) -> str:
    docs = retriever.invoke(query)  # standard vector/BM25 retrieval
    for doc in docs:
        class_name = os.path.basename(source_path).replace(".java", "")
        if class_name in graph:
            # Downstream: what does this class depend on?
            successors = list(graph.successors(class_name))
            # Upstream: what depends on this class?
            predecessors = list(graph.predecessors(class_name))

            # Interface-Consumer Expansion:
            # If class A implements interface I, also get all OTHER
            # classes that implement I (peer consumers)
            for successor in successors:
                edge_data = graph.get_edge_data(class_name, successor)
                if edge_data and edge_data.get("relation") == "implements":
                    interface_users = list(graph.predecessors(successor))
                    predecessors.extend(interface_users)
```

**Expansion rules**:
- **Succ (successors)**: Direct dependencies of retrieved entity — "what does it use?"
- **Pred (predecessors)**: Direct dependents — "what uses it?"
- **InterfaceConsumerExpand**: When A implements I, include all other implementors of I — "peers"

**Results**: DKB 15/15 correct on multi-hop queries vs 6/15 without graph. Cost only ~2x baseline.

### 3.2 Cypher Query Generation (Memgraph Graph-Code)

LLM translates natural language to Cypher:
```
User: "Which functions call handle_sensor_read?"
LLM generates: MATCH (f:Function)-[:CALLS]->(t:Function {name: "handle_sensor_read"}) RETURN f
```

### 3.3 Hybrid Chunk + Graph Search (LightRAG)

Two parallel search modes:
1. **Chunk-based**: Standard vector similarity on code chunks
2. **Graph-based**: Entity extraction from query, then graph traversal from matching entities
Results merged using LightRAG's internal ranking.

### 3.4 Recommended Pattern for openUBMC

The DKB bidirectional expansion pattern is the clear winner for our use case:
- **Simplicity**: NetworkX DiGraph, no external database dependency
- **Speed**: Sub-millisecond traversals for typical codebase graphs (hundreds of nodes)
- **Effectiveness**: 15/15 on multi-hop queries
- **Additive**: Can run alongside existing RRF fusion without disrupting it
- **Interface-consumer expansion** is directly applicable to openUBMC's component-based architecture where components implement common interfaces

---

## 4. Minimal Viable Graph Schema for Lua + C/C++ Codebases

### 4.1 Node Types

| Node Type | ID Convention | Properties | Source |
|-----------|---------------|------------|--------|
| `function` | `{repo}:{file}:{name}` | `language`, `file_path`, `repo_name`, `signature` | Lua `function_declaration`, C `function_definition` |
| `class` | `{repo}:{file}:{name}` | `language`, `file_path`, `repo_name` | Lua `class()`/`singleton()`, C/C++ `struct_specifier`/`class_specifier` |
| `module` | `{repo}:{file}` | `language`, `file_path`, `repo_name` | One per source file |
| `component` | `{repo}` | `language_mix` | One per repository |

### 4.2 Edge Types

| Edge Label | Source | Target | Lua Extraction | C/C++ Extraction |
|------------|--------|--------|----------------|------------------|
| `defines` | module | function/class | Parent AST node | Parent AST node |
| `calls` | function | function | `call_expression` -> match to known functions | `call_expression` -> match |
| `imports` | module | module | `require("path")` calls | `#include "file.h"` directives |
| `inherits` | class | class | N/A in openUBMC Lua | C++ `: public Base` |
| `implements` | class | class | `class(BaseComponent)` pattern | C++ virtual base |
| `contains` | class | function | Method definitions inside class scope | Methods inside class scope |

### 4.3 Tree-sitter Query Patterns for Lua

```python
LUA_DEFINITION_QUERIES = {
    "function": """
        (function_declaration
            name: (identifier) @name)
    """,
    "method": """
        (function_declaration
            name: (method_index_expression) @name)
    """,
    "class_decl": """
        (variable_declaration
            (expression_list
                (function_call
                    name: (identifier) @_fn
                    (@_fn = "class" or @_fn = "singleton"))))
    """,
}

LUA_DEPENDENCY_QUERIES = {
    "require": """
        (function_call
            name: (identifier) @_fn
            (@_fn = "require")
            arguments: (arguments
                (string_content) @module_path))
    """,
    "function_call": """
        (function_call
            name: (identifier) @callee)
    """,
}
```

### 4.4 Tree-sitter Query Patterns for C/C++

```python
C_DEFINITION_QUERIES = {
    "function": """
        (function_definition
            declarator: (function_declarator
                declarator: (identifier) @name))
    """,
    "struct": """
        (struct_specifier
            name: (type_identifier) @name)
    """,
    "class": """
        (class_specifier
            name: (type_identifier) @name)
    """,
}

C_DEPENDENCY_QUERIES = {
    "include": """
        (preproc_include
            path: (string_literal) @include_path)
    """,
    "function_call": """
        (call_expression
            function: (identifier) @callee)
    """,
    "method_call": """
        (call_expression
            function: (field_expression
                field: (field_identifier) @callee))
    """,
    "inheritance": """
        (class_specifier
            base: (base_class_clause
                (type_identifier) @base_name))
    """,
}
```

---

## 5. Specific Implementation Patterns to Adopt

### 5.1 Pattern 1: Two-Pass Graph Construction

**Adopt from**: DKB paper (arXiv 2601.08773)

**Why**: Single-pass approaches miss cross-file references. Two-pass ensures all nodes exist before resolving edges.

**Implementation plan**:
1. Extend existing `LuaParser.parse()` and `CCppParser.parse()` to also emit graph construction data alongside `CodeChunk` results
2. Pass 1: During ingestion, collect all `(node_type, node_id, properties)` tuples from all files
3. Pass 2: After all files are parsed, resolve `require()`/`#include` targets and call targets against the complete node registry
4. Store the graph as a pickled NetworkX DiGraph alongside ChromaDB/BM25 indexes

### 5.2 Pattern 2: Bidirectional Expansion at Retrieval Time

**Adopt from**: DKB paper

**Why**: 15/15 on multi-hop queries vs 6/15 baseline. Solves the cross-component recall problem (currently 29.4% Recall@5).

**Integration point**: After `HybridSearchEngine.search()` returns results, before or during reranking.

```python
# In hybrid_search.py or a new graph_expander.py
def expand_with_graph(
    results: list[SearchResult],
    graph: nx.DiGraph,
    max_hops: int = 2,
    max_expansion: int = 5,
) -> list[SearchResult]:
    expanded = list(results)
    for sr in results:
        node_id = chunk_to_node_id(sr.chunk)
        if node_id not in graph:
            continue
        # Successors (what this chunk depends on)
        for succ in graph.successors(node_id):
            if succ in node_to_chunk:
                expanded.append(make_search_result(succ, source="graph_succ"))
        # Predecessors (what depends on this chunk)
        for pred in graph.predecessors(node_id):
            if pred in node_to_chunk:
                expanded.append(make_search_result(pred, source="graph_pred"))
    return expanded
```

### 5.3 Pattern 3: NetworkX for MVP, Neo4j/Memgraph for Scale

**Adopt from**: DKB paper (NetworkX) + CodeRAG (Neo4j) + Graph-Code (Memgraph)

**Why**: NetworkX requires zero infrastructure, is Python-native, and handles graphs up to ~100K nodes easily. For 11 micro-component repos, the graph will be well under 10K nodes.

**Recommendation**:
- **MVP**: NetworkX DiGraph, serialized with pickle alongside existing indexes
- **Scale path**: If graph grows beyond 50K nodes or needs real-time Cypher queries, migrate to Memgraph (in-memory, Cypher-compatible, supports Lua already)

### 5.4 Pattern 4: Graph Results as Third Retrieval Path in RRF

**Adopt from**: Current RRF fusion architecture

**Integration**: Extend the existing 2-path RRF (Dense + BM25) to 3-path (Dense + BM25 + Graph):

```python
# Current: 2-path RRF
# score(d) = dense_w/(k+rank_d) + bm25_w/(k+rank_b)

# Proposed: 3-path RRF with graph expansion
# score(d) = dense_w/(k+rank_d) + bm25_w/(k+rank_b) + graph_w/(k+rank_g)
```

Graph expansion results get their own weight in RRF, keeping the existing pipeline intact.

### 5.5 Pattern 5: Interface-Consumer Expansion

**Adopt from**: DKB paper's InterfaceConsumerExpand rule

**Why**: In openUBMC's micro-component architecture, multiple components implement the same interface (e.g., `SensorProvider`, `IPMIHandler`). When a query retrieves one component, expanding to peer implementors provides valuable cross-component context.

**Implementation**: When graph traversal encounters an `implements` edge, follow it to the interface node, then follow all predecessor edges back to find all implementors.

### 5.6 Pattern 6: Incremental Graph Updates

**Adopt from**: Memgraph Graph-Code

**Why**: Full graph rebuild on every index update is wasteful. Support incremental updates:
1. On file change, extract only that file's nodes and edges
2. Remove old edges from that file's nodes
3. Add new edges
4. Persist updated graph

### 5.7 Pattern 7: Deterministic AST Extraction (No LLM)

**Adopt from**: DKB paper's key finding

**Critical finding**: DKB (deterministic AST extraction) scored 15/15 while LLM-KB scored 13/15. DKB builds in 2-14 seconds vs 200-884 seconds for LLM-KB. DKB costs ~2x baseline vs ~20-45x for LLM-KB.

**Recommendation**: Use only Tree-sitter for graph construction. Do NOT use LLM to extract graph relationships. This keeps graph building fast, cheap, and deterministic.

---

## 6. Performance Benchmarks from Research

| Method | Multi-hop Accuracy | Build Time | Cost (vs baseline) |
|--------|-------------------|------------|---------------------|
| No Graph (baseline) | 6/15 (40%) | N/A | 1x |
| DKB (AST-derived) | 15/15 (100%) | 2-14 seconds | ~2x |
| LLM-KB (LLM-extracted) | 13/15 (87%) | 200-884 seconds | ~20-45x |

The DKB approach dominates on all three axes: accuracy, speed, and cost.

---

## 7. Proposed File-Level Changes for openUBMC RAG

### New files:
- `ubmc_rag/graph/__init__.py` — Graph module
- `ubmc_rag/graph/schema.py` — Node/edge type definitions, graph construction config
- `ubmc_rag/graph/builder.py` — Two-pass graph construction from existing parser output
- `ubmc_rag/graph/store.py` — NetworkX graph storage (load/save/serialize)
- `ubmc_rag/graph/expander.py` — Bidirectional expansion + interface-consumer expansion

### Modified files:
- `ubmc_rag/ingestion/parsers/lua_parser.py` — Add `extract_graph_data()` method
- `ubmc_rag/ingestion/parsers/c_cpp_parser.py` — Add `extract_graph_data()` method
- `ubmc_rag/ingestion/parsers/base_parser.py` — Add `extract_graph_data()` to ABC
- `ubmc_rag/search/hybrid_search.py` — Add graph expansion as third retrieval path
- `ubmc_rag/search/reranker.py` — Extend RRF to support graph results
- `ubmc_rag/config/settings.py` — Add `GraphConfig` with weights and expansion params
- `ubmc_rag/cli/main.py` — Add `graph` sub-command (build/query graph)

---

## 8. Summary of Recommendations

1. **Use NetworkX DiGraph** — zero infrastructure, sufficient for 11 repos, proven by DKB paper
2. **Two-pass construction** — discover all nodes first, then resolve edges
3. **Deterministic AST extraction only** — no LLM for graph building (2-14s vs 200-884s)
4. **Bidirectional expansion** — successors + predecessors at retrieval time
5. **Interface-consumer expansion** — critical for micro-component architecture
6. **3-path RRF fusion** — Dense + BM25 + Graph as equal-weighted retrieval paths
7. **Incremental updates** — avoid full rebuild on single-file changes
8. **Start with Lua + C/C++ edges only**: `defines`, `calls`, `imports`, `implements`
9. **Node ID convention**: `{repo}:{file}:{name}` for uniqueness across repos
10. **Integrate after search, before rerank** — graph expansion produces candidate results that feed into existing RRF + boosting + diversity pipeline
