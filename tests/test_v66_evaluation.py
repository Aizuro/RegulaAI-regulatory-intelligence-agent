"""V6.6 regression evaluation for the Regulatory Intelligence Agent.

Tests the existing V6.5 production contract without modifying production code
or the vectorstore.
"""

import unittest
from unittest.mock import patch

from src.agent_orchestrator import RegulatoryIntelligenceAgent


class MockLLM:
    def bind_tools(self, tools):
        return self

    def invoke(self, *args, **kwargs):
        class Response:
            content = "Mock answer"
        return Response()


def make_agent():
    return RegulatoryIntelligenceAgent(vectorstore=None, llm=MockLLM())


class TestV66Evaluation(unittest.TestCase):

    def test_eval_internal_regulation_routes_to_get_regulation(self):
        agent = make_agent()
        plan = agent._build_plan(
            "Apa ketentuan UU Nomor 1 Tahun 2024?"
        )
        self.assertEqual(plan[0], "get_regulation")

    def test_eval_comparison_routes_to_compare(self):
        agent = make_agent()
        plan = agent._build_plan(
            "Bandingkan UU Nomor 1 Tahun 2024 dengan UU Nomor 3 Tahun 2024"
        )
        self.assertEqual(plan[0], "compare_regulations")

    def test_eval_missing_source_is_preserved(self):
        agent = make_agent()
        from src.agent_orchestrator import AgentState

        state = AgentState(question="test")
        result = {
            "missing_regulations": [
                {
                    "document_type": "UU",
                    "regulation_number": "19",
                    "year": "2016",
                }
            ]
        }

        agent._record_step(
            state,
            round_number=1,
            tool="get_regulation",
            arguments={
                "document_type": "UU",
                "regulation_number": "19",
                "year": "2016",
            },
            result=result,
        )

        self.assertEqual(
            state.missing_regulations,
            result["missing_regulations"],
        )

    def test_eval_comparison_report_is_preserved(self):
        agent = make_agent()
        from src.agent_orchestrator import AgentState

        state = AgentState(question="test")
        report = {
            "status": "ok",
            "only_in_a": 2,
            "only_in_b": 1,
            "aligned_provisions": 3,
        }

        # _record_step delegates comparison report construction to the
        # production comparison engine. Patch only that dependency so this
        # test isolates state preservation.
        with patch(
            "src.agent_orchestrator.build_comparison_report",
            return_value=report,
        ):
            agent._record_step(
                state,
                round_number=1,
                tool="compare_regulations",
                arguments={},
                result={},
            )

        self.assertEqual(state.comparison_report, report)

    def test_eval_sql_results_are_preserved(self):
        agent = make_agent()
        from src.agent_orchestrator import AgentState

        state = AgentState(question="test")
        sql_result = {
            "rows": [
                {"sector": "Teknologi", "jumlah": 4},
                {"sector": "Keuangan", "jumlah": 2},
            ],
            "row_count": 2,
        }

        agent._record_step(
            state,
            round_number=1,
            tool="run_sql_query",
            arguments={"query": "SELECT ..."},
            result=sql_result,
        )

        self.assertEqual(state.sql_results, [sql_result])

    def test_eval_external_citation_is_grounded_and_deduplicated(self):
        agent = make_agent()

        # V6.5 stores external evidence as:
        # {"source": {...}, "evidence": "..."}
        external_research = [
            {
                "source": {
                    "title": "Regulasi resmi",
                    "url": "https://peraturan.bpk.go.id/example",
                    "source_name": "JDIH BPK",
                    "domain": "peraturan.bpk.go.id",
                    "source_tier": 1,
                },
                "evidence": "Ketentuan resmi tentang regulasi.",
            },
            {
                "source": {
                    "title": "Regulasi resmi",
                    "url": "https://peraturan.bpk.go.id/example",
                    "source_name": "JDIH BPK",
                    "domain": "peraturan.bpk.go.id",
                    "source_tier": 1,
                },
                "evidence": "Duplikat.",
            },
            {
                "source": {
                    "title": "Sumber tanpa evidence",
                    "url": "https://peraturan.bpk.go.id/other",
                    "source_name": "JDIH BPK",
                    "domain": "peraturan.bpk.go.id",
                    "source_tier": 1,
                },
                "evidence": "",
            },
        ]

        citations = agent._build_external_citations(external_research)

        # Production behavior deduplicates by URL and excludes entries
        # without fetched evidence, so only the first valid citation remains.
        self.assertEqual(len(citations), 1)
        self.assertEqual(
            citations[0]["url"],
            "https://peraturan.bpk.go.id/example",
        )
        self.assertTrue(citations[0]["evidence"])
        self.assertNotIn(
            "https://peraturan.bpk.go.id/other",
            [citation["url"] for citation in citations],
        )

    def test_eval_external_failure_does_not_create_fake_evidence(self):
        agent = make_agent()

        with patch(
            "src.agent_orchestrator.search_external_regulations",
            return_value=[],
        ), patch(
            "src.agent_orchestrator.validate_external_results",
            return_value=[],
        ):
            results = agent._build_external_citations([])

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
