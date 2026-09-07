"""Execute every Python cell against simulated OpenAI and Arize services.

No financial conclusions or live-service verification can be drawn from this test.
"""
import ast
import asyncio
import contextlib
import inspect
import io
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
import nbformat
import openai
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pandas as pd

from test_luna_notebook import NOTEBOOK, response_payload


class WorkshopFlowTests(unittest.TestCase):
    def test_all_python_cells_with_mocked_services(self):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        requests, updates, experiments = [], [], []
        sync_class, async_class = openai.OpenAI, openai.AsyncOpenAI
        created_clients = []

        def http_response(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(body["model"], "gpt-5.6-luna")
            if request.url.path.endswith("/chat/completions"):
                # Exercise Phoenix's real structured-output adapter and request kwargs.
                self.assertEqual(body["reasoning_effort"], "low")
                self.assertEqual(body["max_completion_tokens"], 4096)
                schema = body["response_format"]["json_schema"]["schema"]
                labels = schema["properties"]["label"]["enum"]
                label = "not actionable" if "not actionable" in labels else labels[0]
                return httpx.Response(200, json={
                    "id": "chatcmpl_offline", "object": "chat.completion", "created": 1,
                    "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps({
                        "label": label, "explanation": "Offline fixture explanation."
                    })}}], "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
                })
            if "tools" in body:
                text = f"Sourced fixture research: {body['input']}"
                return httpx.Response(200, json=response_payload(text))
            if isinstance(body["input"], list):
                text = f"Fixture report: {body['input'][0]['content']}"
            else:
                text = ("<research_prompt>Research {tickers} about {focus} with sources.</research_prompt>"
                        "<write_prompt>Write an actionable report from the research.</write_prompt>")
            return httpx.Response(200, json=response_payload(text, research=False))

        def sync_client(**kwargs):
            kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(http_response))
            client = sync_class(**kwargs)
            created_clients.append(client)
            return client

        def async_client(**kwargs):
            kwargs["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(http_response))
            client = async_class(**kwargs)
            created_clients.append(client)
            return client

        def export_spans(**kwargs):
            rows = []
            for span in exporter.get_finished_spans():
                row = {
                    "name": span.name,
                    "context.span_id": f"{span.context.span_id:016x}",
                    "context.trace_id": f"{span.context.trace_id:032x}",
                    "parent_id": f"{span.parent.span_id:016x}" if span.parent else None,
                    **{f"attributes.{key}": value for key, value in span.attributes.items()},
                }
                if span.name == "financial_report":
                    row["annotation.human_actionable.label"] = "not_actionable"
                rows.append(row)
            # Simulate an export whose row ordering changes between reads.
            return pd.DataFrame(rows[::-1])

        def update_evaluations(**kwargs):
            frame = kwargs["dataframe"]
            from arize.spans.validation.evals import evals_validation
            import pyarrow as pa
            self.assertFalse(evals_validation.validate_argument_types(
                evals_dataframe=frame, project_name=kwargs["project_name"]))
            self.assertFalse(evals_validation.validate_dataframe_form(evals_dataframe=frame))
            self.assertFalse(evals_validation.validate_values(
                evals_dataframe=frame, project_name=kwargs["project_name"]))
            self.assertEqual(pa.Table.from_pandas(frame, preserve_index=False).num_rows, len(frame))
            ids = {f"{s.context.span_id:016x}" for s in exporter.get_finished_spans()
                   if s.name == "financial_report"}
            self.assertTrue(set(frame["context.span_id"]).issubset(ids))
            self.assertEqual(len(frame.columns), 4)
            self.assertTrue(frame["context.span_id"].is_unique)
            updates.append(frame)

        def run_experiment(**kwargs):
            rows = export_spans()
            rows = rows[rows["name"] == "financial_report"].head(2)
            results = []
            for _, row in rows.iterrows():
                # Test the alternate normalized dataset field too.
                example = {"input": row["attributes.input.value"]}
                output = kwargs["task"](example)
                score = kwargs["evaluators"][0](output, example)
                self.assertEqual(score.label, "not actionable")
                results.append({"output": output, "score": score.score})
            experiments.append(results)
            return SimpleNamespace(id="offline-experiment"), pd.DataFrame(results)

        client = SimpleNamespace(
            spans=SimpleNamespace(export_to_df=export_spans, update_evaluations=update_evaluations),
            experiments=SimpleNamespace(run=run_experiment),
        )
        namespace = {"display": lambda *args: None, "__name__": "workshop_offline"}
        executed = []

        async def run_cells():
            for i, cell in enumerate(nbformat.read(NOTEBOOK, as_version=4).cells):
                if cell.cell_type != "code" or cell.source.startswith("%pip"):
                    continue
                try:
                    code = compile(cell.source, f"notebook-cell-{i}", "exec",
                                   flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                    result = eval(code, namespace)
                    if inspect.isawaitable(result):
                        await result
                except Exception as exc:
                    raise AssertionError(f"Notebook code cell {i} failed: {exc}") from exc
                executed.append(i)

        try:
            with (
                patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test-key",
                                       "ARIZE_API_KEY": "offline-test-key",
                                       "ARIZE_SPACE_ID": "offline-space"}),
                patch("socket.socket.connect", side_effect=AssertionError("Network disabled")),
                patch("openai.OpenAI", side_effect=sync_client),
                patch("openai.AsyncOpenAI", side_effect=async_client),
                patch("arize.ArizeClient", return_value=client),
                patch("arize.otel.register", return_value=provider),
                patch("opentelemetry.trace.get_tracer", side_effect=lambda name, *a, **k: provider.get_tracer(name)),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                asyncio.run(run_cells())
            self.assertIn(70, executed)
            self.assertEqual(len(updates), 4)
            self.assertEqual(len(experiments), 1)
            self.assertEqual(len(namespace["parent_spans"]), 13)
            self.assertEqual(len(namespace["experiment_df"]), 2)
            self.assertTrue(namespace["comparison"]["agree"].all())
            self.assertGreater(len(requests), 70)
        finally:
            OpenAIInstrumentor().uninstrument()
            provider.shutdown()
            for client in created_clients:
                if not client.is_closed():
                    if isinstance(client, async_class):
                        asyncio.run(client.close())
                    else:
                        client.close()


if __name__ == "__main__":
    unittest.main()
