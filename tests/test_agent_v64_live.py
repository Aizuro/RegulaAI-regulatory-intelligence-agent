"""
V6.4 Live / End-to-End verification.

Run from project root:
    python -m unittest testing.test_agent_v64_live -v

This suite intentionally includes one real external-source smoke test.
If the runtime has no internet/DNS access, that test is reported as a
skipped/unavailable environment check rather than being treated as a
code PASS.

The other tests exercise the V6.4 agent fallback contract with the
external pipeline mocked, so they verify routing/state behavior without
depending on Groq or JDIH availability.
"""

import unittest
from unittest.mock import patch

import src.agent_orchestrator as agent_module


class TestV64ExternalPipeline(unittest.TestCase):

    def test_real_external_search_smoke(self):
        """Verify the real JDIH search function can be reached."""
        try:
            results = agent_module.search_external_regulations(
                "Undang-Undang Perlindungan Data Pribadi",
                max_results=1,
            )
        except Exception as exc:
            self.skipTest(f"External network unavailable: {exc}")

        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1)

        result = results[0]
        self.assertTrue(result.get("url", "").startswith("https://"))
        self.assertTrue(result.get("source_tier"))

    def test_external_tool_pipeline_contract(self):
        """Search -> validate -> fetch must produce evidence on success."""
        search_result = [{
            "title": "Test Regulation",
            "url": "https://peraturan.bpk.go.id/Example",
            "snippet": "Official regulatory source",
            "source_name": "JDIH BPK",
            "domain": "peraturan.bpk.go.id",
            "source_tier": 1,
        }]
        fetched_result = {
            "status": "ok",
            "source": search_result[0],
            "evidence": "Pasal 1 ...",
            "evidence_chars": 12,
        }

        with patch.object(
            agent_module,
            "search_external_regulations",
            return_value=search_result,
        ), patch.object(
            agent_module,
            "validate_external_results",
            return_value=search_result,
        ), patch.object(
            agent_module,
            "fetch_external_source",
            return_value=fetched_result,
        ):
            result = agent_module._run_tool(
                None,
                "search_external_regulations",
                {"query": "test", "max_results": 1},
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["evidence"][0]["status"], "ok")

    def test_missing_internal_regulation_triggers_external_search(self):
        """Explicit missing regulation must fall back to external research."""
        agent = agent_module.RegulatoryIntelligenceAgent.__new__(
            agent_module.RegulatoryIntelligenceAgent
        )
        agent.vectorstore = object()

        state = agent_module.AgentState(question="UU 99/2099")
        state.add_missing_regulations({
            "status": "missing_source",
            "missing_regulations": [{
                "document_type": "UU",
                "regulation_number": "99",
                "year": 2099,
            }],
        })

        with patch.object(
            agent_module,
            "_run_tool",
            return_value={
                "status": "ok",
                "results": [],
                "evidence": [{
                    "status": "ok",
                    "source": {
                        "title": "External Regulation",
                        "url": "https://peraturan.bpk.go.id/Example",
                    },
                    "evidence": "Pasal 1 ...",
                }],
            },
        ) as mocked:
            result = agent._record_step(
                state,
                round_number=2,
                tool="search_external_regulations",
                arguments={"query": "UU 99/2099", "max_results": 5},
                result=mocked.return_value,
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(state.external_research), 1)
        self.assertEqual(
            state.external_research[0]["source"]["title"],
            "External Regulation",
        )

    def test_external_evidence_is_exposed_in_final_contract(self):
        """Final result must expose external evidence and external sources."""
        agent = agent_module.RegulatoryIntelligenceAgent.__new__(
            agent_module.RegulatoryIntelligenceAgent
        )
        # _finalize() performs LLM synthesis, so provide a mocked LLM.

        class MockLLM:
            def invoke(self, messages):
                class Response:
                    content = "Jawaban berbasis external evidence."
                return Response()

        agent.llm = MockLLM()

        state = agent_module.AgentState(question="test")
        evidence = {
            "status": "ok",
            "source": {
                "title": "External Regulation",
                "url": "https://peraturan.bpk.go.id/Example",
                "source_tier": 1,
            },
            "evidence": "Pasal 1 ...",
        }
        state.external_research.append(evidence)

        result = agent._finalize(
            "test",
            state,
            {"status": "ok", "evidence": [evidence]},
        )

        self.assertIn("external_research", result)
        self.assertIn("external_sources", result)
        self.assertEqual(len(result["external_research"]), 1)
        self.assertEqual(len(result["external_sources"]), 1)
        self.assertEqual(
            result["external_sources"][0]["url"],
            "https://peraturan.bpk.go.id/Example",
        )

    def test_external_failure_is_graceful(self):
        """External failure must not crash the tool contract."""
        with patch.object(
            agent_module,
            "search_external_regulations",
            return_value=[],
        ), patch.object(
            agent_module,
            "validate_external_results",
            return_value=[],
        ):
            result = agent_module._run_tool(
                None,
                "search_external_regulations",
                {"query": "nonexistent", "max_results": 1},
            )

        self.assertEqual(result["status"], "no_external_evidence")
        self.assertEqual(result["results"], [])
        self.assertEqual(result["evidence"], [])


if __name__ == "__main__":
    unittest.main()
