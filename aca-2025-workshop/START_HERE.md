# Four Colab notebooks: documents → RAG → fine-tuning → comparison

Open each notebook in [Google Colab](https://colab.research.google.com/). Each includes the same workshop code and historical sample sources, so it can start independently. Model weights download separately. For your own source documents, carry the saved corpus artifact through the workflow.

| Step | Notebook | Main action | Save for later |
| --- | --- | --- | --- |
| 1 | [Documents + default LLM](notebooks/workshop/01_documents_default_llm.ipynb) | Upload PDF/CSV/TXT/MD or use the sample; enable `RUN_LLM` | Corpus ZIP; baseline answer |
| 2 | [Hybrid RAG](notebooks/workshop/02_hybrid_rag.ipynb) | Import corpus; build and inspect BM25, semantic and graph retrieval | Index ZIP; RAG answer |
| 3 | [Fine-tune the LLM](notebooks/workshop/03_fine_tune_llm.ipynb) | Load reviewed examples; enable `RUN_TRAINING` | Adapter ZIP, including training provenance |
| 4 | [Compare all approaches](notebooks/workshop/04_compare_all.ipynb) | Import matching artifacts; enable `USE_ADAPTER` and `RUN_COMPARISON` | Comparison ZIP and manual-review worksheet |

## Start with a small run

1. Upload notebook 01, run the setup cells and keep the included sample. Enable `RUN_LLM` to get an actual answer. The default question is a Kentucky enrollment lookup.
2. Save the corpus ZIP using the download control. In notebook 02, choose **Upload corpus artifact**. Build the index and inspect all four retrieval views before enabling generation.
3. In notebook 03, first use the two-step smoke mode to verify training and saving. Set **SMOKE_ONLY=False** for a full training run; the smoke adapter is not a model-quality result.
4. In notebook 04, import the same corpus and index, and upload the saved adapter. Use **validation** while changing settings. Use **test** only after freezing choices.
5. Download results before Colab storage disappears. No Drive connection is required.

For your own uploaded documents, give them the correct source namespace and scope. Unknown state/plan metadata will not pass a specific scope filter. Notebook 03 requires separately reviewed training examples for your corpus; it does not invent labels from raw documents.

## What the comparison measures

- Base LLM + document context.
- Base LLM + hybrid RAG.
- Fine-tuned LLM + the same document context.
- Fine-tuned LLM + the same hybrid RAG.

The same model revision, prompt template and maximum input/output budgets are used. Numeric tools are disabled for all four conditions. Thus this tests document selection and adapter effects; it does not give RAG an extra calculator.

**Default document baseline: explicit partial context.** Complete chunks are selected in stable source order until the budget is reached, with omission counts shown. This is not full-document coverage. Choose strict mode to refuse oversized document selections instead. RAG selects ranked chunks within the same maximum prompt budget. Base and adapter prompts must match for each evidence mode.

## Scope and limits

The sample contains 2025 ACA enrollment sources and selected individual-plan source records. Use `enrollment`, `plans` or `both` filters. The graph follows source-backed record relationships and document structure; it does not infer medical facts, provider participation or personalized prices. It is not Microsoft GraphRAG's community-summary pipeline.

The notebooks install pinned dependencies in their runtime. Retrieval works on CPU; use a supported NVIDIA GPU for extended generation and full training. No paid API is required. GPU availability and actual account charges are external to these notebooks.

See [workshop methods](docs/workshop.md) and [verification results](reports/workshop-results.md) for what was actually tested.
