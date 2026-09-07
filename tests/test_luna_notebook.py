"""Offline regression tests: execute notebook functions with real SDKs and fake HTTP."""
import ast
import asyncio
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import httpx
import nbformat
from openai import AsyncOpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / "Arize_101_Workshop_OpenAI_Luna.ipynb"


def notebook_functions(*names, **namespace):
    """Load only requested function definitions, never setup or paid notebook cells."""
    cells = nbformat.read(NOTEBOOK, as_version=4).cells
    nodes = []
    for cell in cells:
        if cell.cell_type != "code" or cell.source.startswith("%"):
            continue
        nodes.extend(n for n in ast.parse(cell.source).body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names)
    found = {n.name for n in nodes}
    if found != set(names):
        raise AssertionError(f"Missing notebook functions: {set(names) - found}")
    scope = {"re": re, "pd": pd, **namespace}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), scope)
    return scope


def response_payload(text="Research: NVDA revenue", *, research=True, status="completed"):
    output = [{"type": "message", "id": "msg_test", "role": "assistant",
               "status": "completed", "content": [{"type": "output_text", "text": text,
               "annotations": ([{"type": "url_citation", "url": "https://example.com/filing",
                                 "title": "Filing", "start_index": 0, "end_index": 8}]
                               if research else [])}]}]
    if research:
        output.insert(0, {"type": "web_search_call", "id": "ws_test", "status": "completed",
                          "action": {"type": "search", "query": "NVDA revenue"}})
    return {"id": "resp_test", "object": "response", "created_at": 1,
            "model": "gpt-5.6-luna", "status": status, "output": output,
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None}


class NotebookTests(unittest.TestCase):
    def test_valid_notebook_and_compilable_cells(self):
        nb = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(nb)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == "code" and not cell.source.startswith("%"):
                compile(cell.source, f"cell-{i}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)

    def test_openai_only_and_clean_outputs(self):
        nb = nbformat.read(NOTEBOOK, as_version=4)
        source = "\n".join(c.source for c in nb.cells)
        self.assertNotRegex(source.lower(), r"anthropic|claude|sonnet|haiku")
        self.assertIn("gpt-5.6-luna", source)
        self.assertNotIn("Correctness gave us 0/13", source)
        for cell in nb.cells:
            if cell.cell_type == "code":
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)

    def test_prompt_parser_rejects_missing_tags_or_placeholders(self):
        parse = notebook_functions("parse_improved_prompts")["parse_improved_prompts"]
        valid = "<research_prompt>Find {tickers}: {focus}</research_prompt><write_prompt>Write</write_prompt>"
        self.assertEqual(parse(valid), ("Find {tickers}: {focus}", "Write"))
        for bad in ["", valid.replace("{focus}", "nothing"),
                    valid.replace("Write", " "), valid + "<write_prompt>Again</write_prompt>"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse(bad)

    def test_export_selects_completed_report_roots_and_research_context(self):
        prepare = notebook_functions("prepare_parent_spans")["prepare_parent_spans"]
        frame = pd.DataFrame([
            {"name": "financial_report", "parent_id": None, "context.span_id": "a",
             "attributes.input.value": "Research: AAPL", "attributes.output.value": "AAPL report",
             "attributes.research.context": "AAPL source evidence"},
            {"name": "Responses", "parent_id": "a", "context.span_id": "b",
             "attributes.output.value": "wrong child output"},
            {"name": "unrelated", "parent_id": None, "context.span_id": "c"},
            {"name": "financial_report", "parent_id": None, "context.span_id": "d",
             "attributes.output.value": ""},
        ])
        result = prepare(frame)
        self.assertEqual(result.index.tolist(), ["a"])
        self.assertEqual(result.loc["a", "context"], "AAPL source evidence")
        self.assertEqual(result.loc["a", "output"], "AAPL report")

    def test_scores_remain_keyed_by_span_after_export_reordering(self):
        unpack = notebook_functions("evaluation_scores")["evaluation_scores"]
        results = pd.DataFrame({"actionability_score": [
            {"label": "not actionable", "score": 0, "explanation": "failure A"},
            {"label": "actionable", "score": 1, "explanation": "pass B"},
        ]}, index=pd.Index(["a", "b"], name="context.span_id"))
        scores = unpack(results, "actionability")
        parents = pd.DataFrame({"output": ["B report", "A report"]}, index=["b", "a"])
        joined = parents.join(scores.add_prefix("actionability_"), validate="one_to_one")
        self.assertEqual(joined.loc["a", "actionability_explanation"], "failure A")
        self.assertEqual(joined.loc["b", "actionability_label"], "actionable")


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.replies = [response_payload(), response_payload("NVDA report", research=False)]
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.instrumentor = OpenAIInstrumentor()
        self.instrumentor.instrument(tracer_provider=self.provider)

        def handle(request):
            self.requests.append(json.loads(request.content))
            return httpx.Response(200, json=self.replies.pop(0))

        def client_factory(**kwargs):
            return AsyncOpenAI(api_key="offline-test-key", max_retries=0,
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))

        self.ns = notebook_functions(
            "require_response_text", "research_context_from_response", "_financial_report",
            "financial_report", "improved_financial_report",
            AsyncOpenAI=client_factory, tracer=self.provider.get_tracer("test"),
            tracer_provider=self.provider, AGENT_MODEL="gpt-5.6-luna",
            RESEARCH_PROMPT="Research {tickers}. Focus: {focus}", WRITE_PROMPT="Write a report.",
            IMPROVED_RESEARCH_PROMPT="Better {tickers}, {focus}; JSON {other}",
            IMPROVED_WRITE_PROMPT="Improved writing.",
        )

    async def asyncTearDown(self):
        self.instrumentor.uninstrument()
        self.provider.shutdown()

    async def test_research_search_required_writer_gets_evidence_and_citations(self):
        report = await self.ns["financial_report"]("NVDA", "revenue", verbose=False)
        self.assertEqual(report, "NVDA report")
        research, writer = self.requests
        self.assertEqual(research["model"], "gpt-5.6-luna")
        self.assertEqual(research["tools"], [{"type": "web_search"}])
        self.assertEqual(research["tool_choice"], "required")
        self.assertFalse(research["store"])
        self.assertFalse(writer["store"])
        self.assertIn("NVDA revenue", writer["input"][1]["content"])
        self.assertIn("https://example.com/filing", writer["input"][1]["content"])
        self.assertNotIn("temperature", research)
        self.assertNotIn("max_tokens", research)
        spans = self.exporter.get_finished_spans()
        roots = [s for s in spans if s.name == "financial_report"]
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0].attributes["output.value"], report)
        self.assertIn("NVDA revenue", roots[0].attributes["research.context"])
        children = [s for s in spans if s.parent and s.parent.span_id == roots[0].context.span_id]
        self.assertEqual(len(children), 2, "Both real OpenAI SDK requests should be instrumented")

    async def test_incomplete_response_never_becomes_a_successful_report(self):
        self.replies[0] = response_payload(status="incomplete")
        with self.assertRaisesRegex(RuntimeError, "research"):
            await self.ns["financial_report"]("NVDA", "revenue", verbose=False)
        self.assertEqual(len(self.requests), 1)

    async def test_empty_writer_response_is_an_error(self):
        self.replies[1] = response_payload("", research=False)
        with self.assertRaisesRegex(RuntimeError, "report"):
            await self.ns["financial_report"]("NVDA", "revenue", verbose=False)

    async def test_research_without_web_search_is_an_error(self):
        self.replies[0] = response_payload(research=False)
        with self.assertRaisesRegex(RuntimeError, "search"):
            await self.ns["financial_report"]("NVDA", "revenue", verbose=False)

    async def test_improved_agent_preserves_extra_braces(self):
        await self.ns["improved_financial_report"]("NVDA", "revenue")
        self.assertEqual(self.requests[0]["input"], "Better NVDA, revenue; JSON {other}")
        self.assertEqual(self.requests[1]["input"][-1]["content"], "Improved writing.")

    async def test_empty_input_does_not_call_the_api(self):
        with self.assertRaises(ValueError):
            await self.ns["financial_report"](" ", "revenue", verbose=False)
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    # These tests must remain offline, even if a mock is accidentally omitted.
    with patch("socket.socket.connect", side_effect=AssertionError("Network disabled in tests")):
        unittest.main()
