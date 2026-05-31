#!/usr/bin/env python3
"""Scan BM25 k1/b parameters to find optimal combination for code retrieval.

Loads the existing index chunks and evaluation dataset, then evaluates
File@5 across a grid of k1/b values using BM25-only search (no vector/hybrid).
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ubmc_rag.config.settings import AppConfig
from ubmc_rag.indexing.index_manager import IndexManager
from ubmc_rag.indexing.bm25_index import BM25Index
from ubmc_rag.models.code_chunk import CodeChunk
from evaluation.datasets.loader import load_dataset
from evaluation.datasets.schema import ExpectedFile


def _get_relevant_set(expected: list[ExpectedFile]) -> set[str]:
    """Get set of expected file keys: 'repo_name:file_path'."""
    return {f"{e.repo_name}:{e.file_path}" for e in expected}


def _is_match(result_repo: str, result_path: str, expected_keys: set[str]) -> bool:
    """Check if a result (repo_name, file_path) matches any expected file."""
    result_key = f"{result_repo}:{result_path}"
    if result_key in expected_keys:
        return True
    # suffix match to handle path prefix differences
    for ek in expected_keys:
        exp_repo, exp_path = ek.split(":", 1)
        if result_repo == exp_repo and result_path.endswith(exp_path):
            return True
    return False


def evaluate_bm25_params(
    chunks: list[CodeChunk],
    dataset_path: str,
    k1_values: list[float],
    b_values: list[float],
    top_k: int = 5,
) -> dict[tuple[float, float], dict]:
    """Evaluate BM25 File@K across a grid of k1/b values.

    Args:
        chunks: All code chunks to index
        dataset_path: Path to regression dataset YAML
        k1_values: List of k1 values to test
        b_values: List of b values to test
        top_k: K value for File@K metric

    Returns:
        Dictionary mapping (k1, b) -> metrics dict
    """
    dataset = load_dataset(dataset_path)
    test_cases = dataset.test_cases
    print(f"Dataset: {dataset.name} v{dataset.version} - {len(test_cases)} test cases")
    print(f"Chunks loaded: {len(chunks)}")
    print(f"Grid: k1={k1_values}, b={b_values}")
    print(f"Total combinations: {len(k1_values) * len(b_values)}")
    print("=" * 70)

    # Build id -> chunk mapping for result lookups
    chunk_map: dict[str, CodeChunk] = {c.chunk_id: c for c in chunks}

    results = {}

    for k1 in k1_values:
        for b in b_values:
            start = time.time()

            # Build BM25 index with custom params
            bm25 = BM25Index(k1=k1, b=b)
            bm25.build(chunks)

            file_hits = 0
            total = len(test_cases)
            per_case = []

            for tc in test_cases:
                expected_keys = _get_relevant_set(tc.expected_files)
                bm25_results = bm25.search(tc.query, top_k=top_k)

                # Check if any result matches expected files
                hit = False
                for chunk_id, score in bm25_results:
                    chunk = chunk_map.get(chunk_id)
                    if chunk and _is_match(chunk.repo_name, chunk.file_path, expected_keys):
                        hit = True
                        break

                if hit:
                    file_hits += 1
                per_case.append((tc.id, hit))

            file_at_k_val = file_hits / total if total > 0 else 0.0
            elapsed = time.time() - start

            results[(k1, b)] = {
                "file_at_k": file_at_k_val,
                "hits": file_hits,
                "total": total,
                "time_s": round(elapsed, 2),
            }

            print(f"k1={k1:<5} b={b:<5}  File@{top_k}={file_at_k_val:.4f}  "
                  f"({file_hits}/{total})  [{elapsed:.1f}s]")

    return results


def main():
    # Load config and existing index to get chunks
    config = AppConfig.from_yaml("config/default_config.yaml")
    im = IndexManager(config)
    loaded = im.load_index()
    if not loaded:
        print("ERROR: No existing index found. Build the index first.")
        sys.exit(1)

    chunks = im.get_all_chunks()
    if not chunks:
        print("ERROR: No chunks loaded from index.")
        sys.exit(1)

    print(f"Index stats: {im.get_stats()}")
    print()

    # Parameter grid
    k1_values = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    b_values = [0.0, 0.15, 0.3, 0.5, 0.6, 0.75, 0.9, 1.0]

    results = evaluate_bm25_params(
        chunks=chunks,
        dataset_path="evaluation/datasets/regression_v1.yaml",
        k1_values=k1_values,
        b_values=b_values,
        top_k=5,
    )

    # Print sorted results
    print("\n" + "=" * 70)
    print("RESULTS SORTED BY File@5 (best first)")
    print("=" * 70)
    print(f"{'k1':<8} {'b':<8} {'File@5':<10} {'Hits':<8} {'Time'}")
    print("-" * 50)

    sorted_results = sorted(results.items(), key=lambda x: -x[1]["file_at_k"])
    for (k1, b), metrics in sorted_results:
        print(
            f"{k1:<8.2f} {b:<8.2f} {metrics['file_at_k']:<10.4f} "
            f"{metrics['hits']}/{metrics['total']:<5} {metrics['time_s']:.1f}s"
        )

    # Show top 3
    print("\n🏆 TOP 3 COMBINATIONS:")
    for i, ((k1, b), m) in enumerate(sorted_results[:3], 1):
        print(f"  #{i}: k1={k1}, b={b}  →  File@5={m['file_at_k']:.4f}  "
              f"({m['hits']}/{m['total']})")

    # Default comparison
    default_k1, default_b = 1.5, 0.75
    default = results.get((default_k1, default_b), {})
    if default:
        print(f"\n📊 Default (k1={default_k1}, b={default_b}): "
              f"File@5={default['file_at_k']:.4f}")
        best = sorted_results[0]
        print(f"   Best  (k1={best[0][0]}, b={best[0][1]}): "
              f"File@5={best[1]['file_at_k']:.4f}")
        delta = best[1]['file_at_k'] - default['file_at_k']
        print(f"   Δ = {delta:+.4f}")


if __name__ == "__main__":
    main()
