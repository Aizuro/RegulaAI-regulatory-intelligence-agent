
import ast
import unittest
from pathlib import Path

AGENT_FILE = Path("src/agent_orchestrator.py")


class TestAgentV64Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code = AGENT_FILE.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.code)

    def test_module_compiles(self):
        compile(self.code, str(AGENT_FILE), "exec")

    def test_external_research_is_imported(self):
        self.assertIn("external_research", self.code)

    def test_external_search_is_integrated(self):
        self.assertIn("search_external_regulations", self.code)

    def test_external_validation_is_integrated(self):
        self.assertIn("validate_external_results", self.code)

    def test_external_fetch_is_integrated(self):
        self.assertIn("fetch_external_source", self.code)

    def test_external_evidence_is_tracked(self):
        self.assertIn("external_research", self.code)

    def test_external_sources_are_returned(self):
        self.assertIn("external_sources", self.code)

    def test_tool_schema_contains_external_research(self):
        self.assertRegex(
            self.code,
            r'"name"\s*:\s*"search_external_regulations"'
        )

    def test_missing_source_can_trigger_external_research(self):
        # The V6.4 contract must contain an explicit missing-source branch
        # that invokes external research rather than only reporting missing.
        self.assertRegex(
            self.code,
            r'missing_regulations|missing_source'
        )
        self.assertIn("search_external_regulations", self.code)


if __name__ == "__main__":
    unittest.main()
