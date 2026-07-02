import os
import unittest
from unittest.mock import MagicMock, patch

# Enable mock LLMs for test execution before importing graph_app
os.environ["STRATASIFT_MOCK"] = "true"

from stratasift.core.models import SanitisedLiterature
from stratasift.graph import graph_app


class TestGraph(unittest.TestCase):
    """Unit tests for the LangGraph multi-agent orchestrator.

    Uses British spelling rules.
    """

    @patch("stratasift.tools.librarian.LanceLibrarian")
    def test_triage_reject_flow(self, mock_librarian_class: MagicMock) -> None:
        """Verify that an irrelevant document is rejected early in triage."""
        doc = SanitisedLiterature(
            title="Broken Case Study",
            metadata={"status": "broken"},
            abstract_intro="This is a broken and completely irrelevant paper regarding finance.",
            methods=None,
            results_discussion=None,
            conclusions=None,
        )

        initial_state = {
            "source_doc": doc,
            "reading_directive": "",
            "raw_extractions": [],
            "retry_count": 0,
            "insight_dossier": "",
            "vault_context": [],
            "final_markdown": "",
            "relevance_score": 0.0,
            "feedback": None,
            "atomic_insights": [],
            "source_filename": "broken_case.md",
            "routing_results": [],
            "domain_relevance": 0.0,
            "methodology_relevance": 0.0,
            "match_type": "",
            "central_hypothesis": "",
            "in_flight_notes": [],
        }

        final_state = graph_app.invoke(initial_state)

        # Verify early exit
        self.assertLess(final_state["relevance_score"], 0.75)
        self.assertEqual(final_state["reading_directive"], "")
        self.assertEqual(len(final_state["raw_extractions"]), 0)
        self.assertEqual(final_state["insight_dossier"], "")

        # Verify dual-vector fields
        self.assertLess(final_state["domain_relevance"], 0.75)
        self.assertLess(final_state["methodology_relevance"], 0.75)
        self.assertEqual(final_state["match_type"], "")

    @patch("stratasift.tools.librarian.LanceLibrarian")
    def test_battery_success_with_retry_flow(
        self, mock_librarian_class: MagicMock
    ) -> None:
        """Verify the full success flow for a relevant battery paper including a reflection retry."""
        # Setup mock librarian search match
        mock_librarian = MagicMock()
        mock_librarian.search_vault.return_value = [
            {
                "id": "Solid-State Calcination Limits.md",
                "title": "Solid-State Calcination Limits",
                "content": "Previous work shows sintering limits around 900°C.",
            }
        ]
        mock_librarian_class.return_value = mock_librarian

        doc = SanitisedLiterature(
            title="Success Case Battery",
            metadata={"keywords": ["battery", "electrolyte"]},
            abstract_intro="Abstract describing sintering of LLZO solid-state battery electrolyte.",
            methods="The sintering was performed at 800°C for 6h in air. Precursors were mixed.",
            results_discussion="Final compound had ionic conductivity of 1.2e-4 S/cm yielding an overall 85% conversion.",
            conclusions="In conclusion, the LLZO electrolyte demonstrates optimal performance for high-energy density storage.",
        )

        initial_state = {
            "source_doc": doc,
            "reading_directive": "",
            "raw_extractions": [],
            "retry_count": 0,
            "insight_dossier": "",
            "vault_context": [],
            "final_markdown": "",
            "relevance_score": 0.0,
            "feedback": None,
            "atomic_insights": [],
            "source_filename": "success_case_battery.md",
            "routing_results": [],
            "domain_relevance": 0.0,
            "methodology_relevance": 0.0,
            "match_type": "",
            "central_hypothesis": "",
            "in_flight_notes": [],
        }

        final_state = graph_app.invoke(initial_state)

        # Verify triage passed
        self.assertGreaterEqual(final_state["relevance_score"], 0.75)
        self.assertIn("battery synthesis", final_state["reading_directive"])

        # Verify dual-vector triage fields
        self.assertGreaterEqual(final_state["domain_relevance"], 0.75)
        self.assertGreaterEqual(final_state["methodology_relevance"], 0.75)
        self.assertEqual(final_state["match_type"], "full")
        self.assertIn("LLZO", final_state["central_hypothesis"])

        # Verify consolidated extractions present
        self.assertGreater(len(final_state["raw_extractions"]), 0)

        # Verify reflection loop triggered (retry_count incremented to 1)
        self.assertEqual(final_state["retry_count"], 1)

        # Verify final dossier and markdown outputs
        self.assertIn("Cross-reference verified", final_state["insight_dossier"])
        self.assertIn("Solid-State Calcination Limits", final_state["vault_context"])
        self.assertIn(
            "Aggregated Insight from success_case_battery.md",
            final_state["final_markdown"],
        )

        # Verify networking decision
        routing_results = final_state["routing_results"]
        self.assertGreaterEqual(len(routing_results), 1)
        self.assertEqual(routing_results[0]["decision"], "append")
        self.assertEqual(
            routing_results[0]["target_note"], "Solid-State Calcination Limits.md"
        )

        # Clean up database file
        from pathlib import Path

        db_path = Path(final_state["context_db_path"])
        self.assertTrue(db_path.exists())
        db_path.unlink()

    @patch("stratasift.tools.librarian.LanceLibrarian")
    def test_battery_verification_failure_exceed_retry_flow(
        self, mock_librarian_class: MagicMock
    ) -> None:
        """Verify that when verification fails and loop retry limit is exceeded,
        the flow proceeds to Supervisor_Network and flags the output note as unverified.
        """
        # Setup mock librarian search match
        mock_librarian = MagicMock()
        mock_librarian.search_vault.return_value = []
        mock_librarian_class.return_value = mock_librarian

        from stratasift.config import get_runtime_config, set_runtime_config

        config = get_runtime_config()

        # Override debate loops limit to 0
        original_loops = config.blocks.analysis_block.max_debate_loops
        config.blocks.analysis_block.max_debate_loops = 0
        set_runtime_config(config)

        try:
            doc = SanitisedLiterature(
                title="Success Case Battery",
                metadata={"keywords": ["battery", "electrolyte"]},
                abstract_intro="Abstract describing sintering of LLZO solid-state battery electrolyte.",
                methods="The sintering was performed at 800°C for 6h in air. Precursors were mixed.",
                results_discussion="Final compound had ionic conductivity of 1.2e-4 S/cm yielding an overall 95% conversion.",  # triggers discrepancy
                conclusions=None,
            )

            initial_state = {
                "source_doc": doc,
                "reading_directive": "",
                "raw_extractions": [],
                "retry_count": 0,
                "insight_dossier": "",
                "vault_context": [],
                "final_markdown": "",
                "relevance_score": 0.0,
                "feedback": None,
                "atomic_insights": [],
                "source_filename": "success_case_battery.md",
                "routing_results": [],
                "domain_relevance": 0.0,
                "methodology_relevance": 0.0,
                "match_type": "",
                "central_hypothesis": "",
                "in_flight_notes": [],
            }

            final_state = graph_app.invoke(initial_state)

            # Since max_debate_loops=0, retry_count remains 0
            self.assertEqual(final_state["retry_count"], 0)

            # Verify final_markdown contains the unverified status and discrepancy log
            self.assertIn('status: "unverified"', final_state["final_markdown"])
            self.assertIn("## Discrepancy Log", final_state["final_markdown"])
            self.assertIn(
                "Discrepancy found in Yield metric", final_state["final_markdown"]
            )

            # Verify tmp-ContextDB JSON file exists and is populated
            import json
            from pathlib import Path

            db_path = Path(final_state["context_db_path"])
            self.assertTrue(db_path.exists())
            with open(db_path, "r", encoding="utf-8") as f:
                db_data = json.load(f)
            self.assertEqual(db_data["verified"], False)
            self.assertIn("Discrepancy found in Yield metric", db_data["feedback"])

            # Clean up db file
            db_path.unlink()
        finally:
            config.blocks.analysis_block.max_debate_loops = original_loops
            set_runtime_config(config)

    @patch("stratasift.tools.librarian.LanceLibrarian")
    def test_methodology_only_match(self, mock_librarian_class: MagicMock) -> None:
        """Verify methodology-only match type when domain is low but methodology is high."""
        mock_librarian = MagicMock()
        mock_librarian.search_vault.return_value = []
        mock_librarian_class.return_value = mock_librarian

        doc = SanitisedLiterature(
            title="Biomass Pyrolysis Study",
            metadata={"keywords": ["biomass", "pyrolysis"]},
            abstract_intro="Abstract describing biomass pyrolysis conditions using machine learning force fields.",
            methods="Pyrolysis was performed at 500°C under nitrogen atmosphere.",
            results_discussion="Biochar yield reached 65% with slower heating rates.",
            conclusions="These methods enable broader applications in bioenergy.",
        )

        initial_state = {
            "source_doc": doc,
            "reading_directive": "",
            "raw_extractions": [],
            "retry_count": 0,
            "insight_dossier": "",
            "vault_context": [],
            "final_markdown": "",
            "relevance_score": 0.0,
            "feedback": None,
            "atomic_insights": [],
            "source_filename": "variant_case_biomass.md",
            "routing_results": [],
            "domain_relevance": 0.0,
            "methodology_relevance": 0.0,
            "match_type": "",
            "central_hypothesis": "",
            "in_flight_notes": [],
        }

        final_state = graph_app.invoke(initial_state)

        # Verify dual-vector triage: methodology high, domain low
        self.assertLess(final_state["domain_relevance"], 0.95)  # 0.75
        self.assertGreaterEqual(final_state["methodology_relevance"], 0.75)
        self.assertEqual(final_state["match_type"], "methodology_only")

        # Verify the paper still passes triage via max() gate
        self.assertGreaterEqual(final_state["relevance_score"], 0.75)

        # Verify central hypothesis is populated
        self.assertGreater(len(final_state["central_hypothesis"]), 0)

        # Clean up database file
        from pathlib import Path

        db_path_str = final_state.get("context_db_path")
        if db_path_str:
            db_path = Path(db_path_str)
            if db_path.exists():
                db_path.unlink()

    @patch("stratasift.tools.librarian.LanceLibrarian")
    def test_deduplication_merges_similar_insights(
        self, mock_librarian_class: MagicMock
    ) -> None:
        """Verify that in-flight deduplication merges semantically similar insights."""
        mock_librarian = MagicMock()
        mock_librarian.search_vault.return_value = []
        mock_librarian_class.return_value = mock_librarian

        # Create two near-identical AtomicInsight objects
        from stratasift.core.models import AtomicInsight

        insight_a = AtomicInsight(
            title="Sintering LLZO solid-state electrolyte yields high ionic conductivity",
            core_insight="Solid-state battery synthesis using LLZO electrolyte. Sintering protocol at 800°C for 6 hours yields a cubic phase with ionic conductivity of 1.2e-4 S/cm and an overall yield of 85%. This achieves optimal performance parameters required for high-energy density storage applications.",
            context_evidence="Evidence block A",
            related_vectors=["success_case_battery"],
        )
        insight_b = AtomicInsight(
            title="LLZO electrolyte sintering protocol produces high conductivity cubic phase",
            core_insight="LLZO electrolyte sintering at 800°C for 6h in air produces a cubic phase. The resulting compound shows ionic conductivity of 1.2e-4 S/cm with an overall yield of 85%. These parameters confirm suitability for high-energy density solid-state battery applications.",
            context_evidence="Evidence block B",
            related_vectors=["success_case_battery"],
        )

        from stratasift.graph.nodes import supervisor_network_node
        from stratasift.config import get_runtime_config, set_runtime_config

        config = get_runtime_config()
        set_runtime_config(config)

        state = {
            "source_doc": SanitisedLiterature(
                title="Test",
                metadata={},
                abstract_intro="Test abstract",
                methods="Test methods",
                results_discussion="Test results",
                conclusions=None,
            ),
            "relevance_score": 0.9,
            "atomic_insights": [insight_a, insight_b],
            "source_filename": "test_dedup.md",
            "context_db_path": "",
            "run_id": "test-dedup-run",
            "in_flight_notes": [],
            "domain_relevance": 0.9,
            "methodology_relevance": 0.85,
            "match_type": "full",
            "central_hypothesis": "Test hypothesis",
        }

        result = supervisor_network_node(state)

        # With dedup, only one unique insight should produce a routing result
        # (the second near-identical one should be merged)
        self.assertEqual(len(result["routing_results"]), 1)
        self.assertEqual(result["routing_results"][0]["decision"], "create")


if __name__ == "__main__":
    unittest.main()
