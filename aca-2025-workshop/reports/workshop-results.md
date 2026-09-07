# Four-notebook verification — 2026-09-07

All four delivered notebook workflows executed in separate real local Jupyter kernels using the pinned pretrained Qwen model and real BGE embeddings. The adapter run used two optimizer steps to verify the workflow. No full-training improvement, final-test accuracy, Colab browser behavior or CUDA compatibility is claimed.

## Execution evidence

| Notebook | Code cells executed | Actual work verified |
| --- | ---: | --- |
| 01 Documents + default LLM | 7 | Source load/filter, pretrained inference, corpus export |
| 02 Hybrid RAG | 8 | Import notebook01 corpus; real semantic index; BM25/graph/fused search; pretrained inference; index export |
| 03 Fine-tune | 6 | Load reviewed splits; two real LoRA optimizer steps; save/reload adapter; export |
| 04 Compare | 8 | Import corpus, index and adapter; run two validation questions in all four conditions; save paired results and review worksheet |

Total: 29 code cells, zero cell errors. The saved runs explicitly record validation overrides: dependency installation skipped because the isolated environment already contained the pinned packages; model/training/comparison controls enabled; CPU selected; notebook04 used one question per dataset and imported earlier exported artifacts. Delivered notebooks keep generation/training opt-in. The smoke adapter is marked as such in training and comparison metadata.

Full training/validation token preflight passed: 326 training examples (maximum 2085 tokens), 211 validation examples (maximum 1933), all below 4096. The frozen workshop snapshot has 564 final-test questions. None were used in the model smoke comparison.

The final source corpus has 3,991 chunks: 617 enrollment and 3,374 plan chunks, connected by 6,484 explicit source/structure links. These include 1,760 distinct tobacco/non-tobacco rate observations. The full dense index was built with actual BGE-small-en-v1.5 embeddings. No fake embedding or injected generation fixture is counted as runtime evidence.

The upload path was also exercised with an actual insurer SBC PDF: 8 extracted pages, 24 chunks, corpus ZIP export/import, embedding/index generation, eight returned passages and zero wrong-state matches. This is an ingestion test; it does not certify every PDF's table extraction.

## Retrieval validation

Source recall at 8, measured against **partial existing citation judgments**, with no final-test tuning:

| Dataset | Resolvable cases | BM25 | Semantic | Graph with lexical seeds | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| Enrollment | 45 | 62.2% | 100.0% | 20.0% | 82.2% |
| Plan benefits | 129 | 75.7% | 54.4% | 81.3% | 92.4% |

These are source-retrieval diagnostics, not answer accuracy. Cases with absent/unresolvable reference IDs do not enter these denominators. The semantic-only result is stronger for enrollment; the combined result is stronger for the plan sample. This does not establish universal superiority. Full settings, query-set hash and timings are in `workshop/retrieval-validation.json`.

## Correctness checks and remaining limits

- 14 dedicated workshop tests passed, including exact scope/profile filtering, 1,760 rate observations, graph provenance/cycles, artifact integrity, source-upload rejection, context-budget behavior and reference-label isolation. Luna reviewed the final corrections and found no remaining blocking defect in this scope.
- The broader workspace suite passed 90 tests on the final development check. That suite also contains separately evolving plan-specific work; its count is not the workshop suite's size.
- Final notebook payloads are byte-identical across the four entry points and match the final RAG source and sample. Python/notebook syntax and limited credential-pattern checks passed. No dedicated security scanner or browser visual check was performed.
- Initial model smoke outputs included malformed JSON and inaccurate or insufficient answers. These failures were preserved. Improved format instructions were tested using development/validation examples, not the frozen final test. JSON and citation-ID validity remain separate from semantic correctness.
- The eight final model-comparison rows establish pipeline execution only. Review `workshop/predictions.jsonl`, the shuffled worksheet and the source evidence before grading answer quality. No overall semantic accuracy or fine-tuning improvement has been assigned.
- Document baseline defaults to explicit partial context in stable source order. Omitted chunks are reported. Strict mode refuses overflow. It is not a whole-corpus-in-context baseline.
- The graph consists of verified record joins and structural links. It does not implement Microsoft GraphRAG's LLM-generated community summaries or unrestricted entity extraction from uploads.
- Full one-epoch training, full held-out evaluation, independent domain review and Colab GPU execution remain unrun. No paid compute, public publishing or commits were performed.

Machine-readable evidence is in `reports/workshop/`; locally executed notebook copies are in `reports/workshop/executed/`. The primary entry points are the four clean notebooks under `notebooks/workshop/`.
