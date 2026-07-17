# Glass attribution experiments

## HNSW hierarchy isolation

The experiment targets current pyglass commit `d2296ec447d2374ee8f88c6d3b85be1b1e434ad3` (`glass==2.1.0`). Apply [pyglass-hierarchy.patch](pyglass-hierarchy.patch) at the pyglass repository root and rebuild its Python extension.

The patch adds three graph initialization modes without changing the serialized graph or layer-0 search:

- `hierarchy`: normal greedy HNSW upper-layer descent.
- `global_entry`: remove upper layers and seed layer 0 from the HNSW global entry.
- `fixed_entry`: remove upper layers and seed layer 0 from an explicitly selected vertex.

It also reports total and initializer distance comparisons for both single-query `SearchImpl1` and batch `SearchImpl2`. Layer-0 comparisons are total minus initializer comparisons.

[../benchmark_glass_attribution.py](../benchmark_glass_attribution.py) runs repeated single-query QPS and an untimed counter pass. [../analyze_glass_hierarchy.py](../analyze_glass_hierarchy.py) verifies graph hashes and creates same-ef and matched-recall tables.
