import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "src" / "agent_orchestrator.py"
API = ROOT / "src" / "api.py"


class TestV65Citation(unittest.TestCase):
    def test_agent_has_external_citation_builder(self):
        source = AGENT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIn("def _build_external_citations", source)
        self.assertIn('"external_citations"', source)
        self.assertIsInstance(tree, ast.Module)

    def test_external_citation_fields(self):
        source = AGENT.read_text(encoding="utf-8")
        for field in ("title", "url", "source_name", "domain", "source_tier", "evidence"):
            self.assertIn(f'"{field}"', source)

    def test_https_and_deduplication_are_guarded(self):
        source = AGENT.read_text(encoding="utf-8")
        self.assertIn('url.startswith("https://")', source)
        self.assertIn("seen_urls", source)
        self.assertIn("OFFICIAL_DOMAINS.get(domain)", source)

    def test_api_exposes_external_sources_and_citations_without_downgrading_version(self):
        source = API.read_text(encoding="utf-8")
        self.assertIn("external_sources", source)
        self.assertIn("external_citations", source)
        match = re.search(r'version="([0-9]+\.[0-9]+\.[0-9]+)"', source)
        self.assertIsNotNone(match, "FastAPI version must be explicitly declared")
        self.assertNotEqual(match.group(1), "6.5.0", "Do not downgrade the current API version for a legacy test")

    def test_citations_are_emitted_in_both_agent_return_paths(self):
        source = AGENT.read_text(encoding="utf-8")
        self.assertEqual(source.count('"external_citations": self._build_external_citations(state.external_research)'), 2)

    def test_evidence_is_required_and_bounded(self):
        source = AGENT.read_text(encoding="utf-8")
        self.assertIn('if not evidence:', source)
        self.assertIn('"evidence": evidence[:1200]', source)
        self.assertIn('"evidence_chars": len(evidence)', source)

    def test_all_v65_files_compile(self):
        ast.parse(AGENT.read_text(encoding="utf-8"))
        ast.parse(API.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
