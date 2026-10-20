import unittest
from unittest.mock import patch

from src.external_research import (
    OFFICIAL_DOMAINS,
    search_external_regulations,
    validate_external_source,
)


class TestExternalResearchV61(unittest.TestCase):

    def test_empty_query(self):
        self.assertEqual(search_external_regulations(""), [])

    def test_max_results_is_bounded(self):
        self.assertEqual(search_external_regulations("   "), [])

    @patch("src.external_research._fetch")
    def test_official_source_result(self, mock_fetch):
        mock_fetch.return_value = """
        <html>
          <head><title>JDIH BPK - Pencarian</title></head>
          <body>Undang-Undang Nomor 27 Tahun 2022 tentang Perlindungan Data Pribadi</body>
        </html>
        """

        results = search_external_regulations(
            "Undang-Undang Perlindungan Data Pribadi",
            max_results=5,
        )

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result["source_name"], "JDIH BPK")
        self.assertEqual(result["domain"], "peraturan.bpk.go.id")
        self.assertEqual(result["source_tier"], 1)
        self.assertIn("peraturan.bpk.go.id", result["url"])
        self.assertIn("Perlindungan Data Pribadi", result["snippet"])

    def test_official_source_validation(self):
        result = {
            "title": "UU Perlindungan Data Pribadi",
            "snippet": "Ketentuan resmi mengenai perlindungan data pribadi.",
            "url": "https://peraturan.bpk.go.id/Details/123",
            "source_name": "JDIH BPK",
            "source_tier": 1,
        }
        self.assertTrue(validate_external_source(result))

    def test_untrusted_source_rejected(self):
        result = {
            "url": "https://example.com/regulasi",
            "source_name": "Example",
            "source_tier": 4,
        }
        self.assertFalse(validate_external_source(result))

    def test_allowlist_contains_bpk_and_jdihn(self):
        self.assertIn("peraturan.bpk.go.id", OFFICIAL_DOMAINS)
        self.assertIn("jdihn.go.id", OFFICIAL_DOMAINS)


if __name__ == "__main__":
    unittest.main()
