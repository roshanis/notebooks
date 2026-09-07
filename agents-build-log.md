## [AGENT: Codex] 2026-09-07T18:35Z
### Action: Add an OpenAI Luna edition of the Arize workshop
### Files changed:
- Arize_101_Workshop_OpenAI_Luna.ipynb
- tests/test_luna_notebook.py
- tests/test_workshop_flow.py
- tests/README.md
- agents-build-log.md
### Diff summary:
- Added a separate 72-cell notebook using gpt-5.6-luna for research, report writing, evaluation, and prompt rewriting. The original workshop remains the source reference.
- Switched to OpenAI Responses with required hosted web search, OpenInference instrumentation, explicit research/citation handoff, and per-report async clients.
- Added credential setup, pinned tested dependencies, response validation, and span-ID-based evaluation alignment. Cleared saved outputs and stale score claims.
- User approved implementation and requested a push to the source repository. Prepared codex/luna-notebook from the current main branch.
### Recommendations / Next steps:
- Run the offline suite with `python -m unittest discover -s tests -v` using the dependencies in the notebook and `nbformat==5.11.1`.
- Earlier validation passed all 12 tests, including execution of all 36 Python cells with mocked services and Arize evaluation-payload validation. Luna review found no blocking implementation issues.
- Pre-push validation in the repository checkout: all 12 tests passed in 10.923 seconds. Publication scan found no credential strings or personal local paths in the five files being added.
- Live model access, hosted search quality, Arize ingestion, human annotations, and the saved-dataset experiment require validation in the user's account. Follow the Colab instructions at the top of the notebook.
