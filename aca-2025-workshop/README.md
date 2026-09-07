# ACA 2025: four independent Colab notebooks

Build and compare document-context inference, hybrid retrieval and adapter fine-tuning using historical ACA data.

| Step | Notebook | Open in Colab |
| --- | --- | --- |
| 1 | [Documents + default LLM](notebooks/workshop/01_documents_default_llm.ipynb) | [Open](https://colab.research.google.com/github/roshanis/notebooks/blob/codex/aca-four-notebook-workshop/aca-2025-workshop/notebooks/workshop/01_documents_default_llm.ipynb) |
| 2 | [BM25 + semantic + graph RAG](notebooks/workshop/02_hybrid_rag.ipynb) | [Open](https://colab.research.google.com/github/roshanis/notebooks/blob/codex/aca-four-notebook-workshop/aca-2025-workshop/notebooks/workshop/02_hybrid_rag.ipynb) |
| 3 | [Fine-tune the LLM](notebooks/workshop/03_fine_tune_llm.ipynb) | [Open](https://colab.research.google.com/github/roshanis/notebooks/blob/codex/aca-four-notebook-workshop/aca-2025-workshop/notebooks/workshop/03_fine_tune_llm.ipynb) |
| 4 | [Compare all approaches](notebooks/workshop/04_compare_all.ipynb) | [Open](https://colab.research.google.com/github/roshanis/notebooks/blob/codex/aca-four-notebook-workshop/aca-2025-workshop/notebooks/workshop/04_compare_all.ipynb) |

[Instructions](START_HERE.md) · [Methods](docs/workshop.md) · [Measured verification](reports/workshop-results.md) · [Known limitations](KNOWN_LIMITATIONS.md)

Each notebook includes the same runtime and sample corpus. Model weights download separately. Save corpus, index and adapter ZIPs between Colab sessions. Generation and training are opt-in.

All four workflows executed in fresh local Jupyter kernels with real BGE embeddings, pretrained Qwen inference, two LoRA optimizer steps and saved-artifact handoffs. Full training, Colab GPU validation and independently reviewed answer quality remain unverified. A two-step adapter is a mechanics check.

## Local checks

Install `requirements-rag.txt` in an isolated environment when running models. The pure-Python portable tests run with `python3 -m unittest discover -s tests/workshop -v` from this directory. The original workspace additionally ran a CMS raw-source integration test; large original ZIP/PDF archives are not duplicated here. Prebuilt source text and provenance are in `sample/`. The maintenance importer needs those original archives, whereas the four notebooks use the included sample or uploaded documents.

This is a historical educational prototype, not personal insurance or clinical advice. No paid API is required; GPU charges depend on the runtime account. Colab links point to the publication branch so they work before the PR is merged.

The notebook builders and frozen dataset snapshot are included. For example, create new copies with `python scripts/build_four_notebooks.py --corpus sample --output NEW_DIRECTORY`. Executed notebook copies referenced by the original verification report remain in the local development workspace; portable execution summaries, predictions and review records are included here.
