# Roadmap Notes

## Build Order

1. Scaffold the repository.
2. Write a notebook for the concept before hardening scripts.
3. Implement KV-cache memory math.
4. Generate memory comparison plots.
5. Create static diagrams for KV cache and attention variants.
6. Implement dense MHA.
7. Add decode cache tests.
8. Implement MQA and GQA.
9. Implement simplified MLA.
10. Implement sparse masks before sparse attention modules.
11. Add synthetic long-context retrieval tasks.
12. Add tiny LM training.
13. Benchmark decode.
14. Add one small Triton kernel.

## Guiding Principle

Prefer correctness and readability first. Optimization comes only after the
reference behavior is tested and explainable.

For learning-heavy milestones, notebooks come first. They should be
self-contained and avoid importing project package modules. Scripts and package
modules come after the concept has been explained and exercised interactively.

Videos come after storyboard approval. Rough animation prototypes should stay
out of release artifacts.
