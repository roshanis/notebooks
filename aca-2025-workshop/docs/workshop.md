# Workshop methods and artifact contracts

## Four stages

Notebook 01 ingests source documents, displays scoped chunks and gives the unchanged LLM a stable-order document baseline. Notebook 02 builds real dense embeddings, BM25 statistics and a graph, then fuses their rankings. Notebook 03 trains a LoRA adapter from reviewed chat examples. Notebook 04 reloads the exported artifacts in a fresh runtime and performs paired comparisons.

Each notebook embeds the same shared Python runtime, source sample and separate demonstration datasets. Corpus ingestion reads only explicit source files; it never recursively indexes the project or training/evaluation records. The sample's original file hashes are preserved in its manifest. Source chunks retain IDs, original URLs, page/CSV locators, year, state and plan variant.

Uploads support PDF, CSV, TXT and MD. Archives and notebooks are not accepted as source documents. PDF extraction is text-based; image-only pages are reported as needing OCR. Table text extraction can lose column alignment, so the original document remains authoritative. Uploaded documents are data and never execute as code. Graph relationships for arbitrary uploads are structural links, not machine-inferred factual claims.

## Retrieval

- BM25 uses term-frequency saturation and document-length normalization, retaining exact plan IDs in tokenization.
- Semantic retrieval uses pinned `BAAI/bge-small-en-v1.5`, its query instruction and normalized dense vectors. Chunks are320 embedding tokens with40-token overlap. Embedding input overflow fails instead of silently truncating.
- Graph retrieval begins at three lexical seed passages and traverses explicit links up to depth2, with candidate/visit limits. Paths retain their source passage. This graph-only ablation therefore includes lexical seeding; it is not an independent entity-linking model.
- Plan benefit/rate records link to exact-plan cost rules. Enrollment observations link to metric definitions and reporting scope. Same-record chunks link for context continuity. Graph traversal cannot cross namespaces, plan variants or conflicting scoped states.
- Tobacco and non-tobacco rate observations are represented separately using the correct column of the original CMS row. `No Preference` rows would use the individual-rate column for both profiles. Profile filters are exact; source-unknown uploaded records cannot pass state/plan-specific filters.
- Reciprocal rank fusion merges the ranked lists. All methods apply the same explicit source filters. Scores are ranking signals, not calibrated confidence.

Embedding model documentation: https://huggingface.co/BAAI/bge-small-en-v1.5
Microsoft GraphRAG's distinct local-search implementation: https://microsoft.github.io/graphrag/query/local_search/

## Model and training

The base is `Qwen/Qwen2.5-1.5B-Instruct` at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Inference is deterministic decoding with a4096-token input ceiling and384-token generation ceiling. The complete prompt is measured with the actual model tokenizer. No exact-value lookup or numerical tool is given to any condition.

LoRA targets query/value projections with rank8 and alpha16. Completion-only loss masks prompt tokens; token-length checks reject overlong training examples. Full mode runs one epoch with validation. Two-step smoke mode tests mechanics only. Notebook 03 saves weights, base revision, source-corpus hash, dataset hashes, IDs, settings and training metrics. Notebook 04 validates those before loading the adapter.

The bundled dataset currently has326 training examples,211 validation questions and564 test questions across the two domains. This freezes the separate plan project's v3 snapshot (26 training,161 validation and500 test questions), alongside enrollment's300/50/64 split. Many cases share templates or source families; they are not564 independent scenarios. Example construction and source availability are separate from answer-quality evidence. Available source facts may be retrieved at evaluation time; held-out questions/reference answers may not enter training or retrieval.

## Artifacts and reproducibility

Corpus, index, adapter and comparison exports are ZIP files with a kind/schema manifest and per-file SHA-256 hashes. Import verifies contents and paths and refuses changed existing files. Export names are unique. Index fingerprints bind vectors to corpus and configuration. Notebook bootstrap verifies its embedded runtime and reuses only byte-identical files.

The comparison saves exact prompts, source IDs, source text, graph traces, input counts, omission counts, model/package settings, timing, errors and paired question IDs. Adapter/base prompts are checked for parity within each evidence mode. A missing adapter is shown as two unrun conditions, never as a copied base result.

Default prefix document context explicitly reports omissions. Strict mode refuses overflow. RAG uses ranked complete chunks. Different selection strategies may consume different numbers of tokens within the common maximum; the measured counts are preserved. This is a controlled context-budget comparison, not a claim of identical evidence content between document and RAG modes.

## Evaluation

Use validation while making changes, then freeze configuration before final testing. The retrieval report compares BM25, dense, graph and combined retrieval. Existing citation sets provide partial source-recall judgments, not exhaustive relevance labels; unresolved citation IDs are reported and excluded from denominators. Five enrollment validation definition cases lack resolvable record-level labels in the current mapping. Do not describe these scores as answer accuracy.

The answer summary reports attempts, generation errors, valid JSON, supplied citation-ID validity, context omissions and latency separately by source group and condition. A shuffled worksheet supports human grading of correctness, citation entailment and caveats, with a separate condition key. Overall semantic accuracy remains ungraded until that review is completed.

## Development checks

Run `python3 -m unittest discover -s tests/workshop -v` for the dedicated workshop suite. The original source-integrity suite is separate. `scripts/build_four_notebooks.py` creates new notebook files and refuses overwrites. `scripts/execute_four_notebooks.py` executes the actual workflow in fresh local kernels, passing exported artifacts between notebooks and enabling documented smoke controls. Local execution is not proof of Colab GPU compatibility.
