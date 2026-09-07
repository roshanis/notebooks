# Offline notebook checks

The notebook's first cell lists the pinned runtime dependencies. Install those in an isolated Python 3.12 environment and add `nbformat==5.11.1`, then run:

```sh
python -m unittest discover -s tests -v
```

`test_luna_notebook.py` checks notebook validity, response errors, required web search, source handoff, tracing, prompt parsing, and span-ID alignment.

`test_workshop_flow.py` executes all 36 Python cells using the real OpenAI, Phoenix, and OpenInference packages. OpenAI HTTP responses, Arize exports, human labels, and the experiment service are simulated; network connections are blocked. Arize's installed dataframe validators check evaluation payloads before the simulated upload.

These tests do not verify live model access, hosted web search, Arize ingestion, the Arize UI, or financial-analysis quality. To verify those, upload the notebook to Colab, configure the three secrets documented at the top, run Steps 1–7, label reports in Arize for Step 8, and save the failure dataset before running the Step 9 experiment.
