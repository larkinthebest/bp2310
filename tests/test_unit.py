"""
Unit tests for local logic that doesn't require external services.
Tests config parsing, deterministic ID generation, context building, source derivation.
"""

import os
import sys
import unittest

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestConfig(unittest.TestCase):
    def test_config_loads(self):
        from src.config import cfg
        self.assertIsNotNone(cfg)
        self.assertIn(cfg.domain_profile, ("sports", "general"))

    def test_prompt_profiles(self):
        from src.config import PROMPT_PROFILES
        self.assertIn("sports", PROMPT_PROFILES)
        self.assertIn("general", PROMPT_PROFILES)
        for profile in PROMPT_PROFILES.values():
            self.assertIn("system", profile)
            self.assertIn("caption", profile)

    def test_config_validate(self):
        from src.config import cfg
        warnings = cfg.validate()
        self.assertIsInstance(warnings, list)


class TestVectorStoreHelpers(unittest.TestCase):
    def test_deterministic_ids(self):
        from src.rag.vector_store import _make_vector_id
        id1 = _make_vector_id("video.mp4", "video_frame", 0, 12.0)
        id2 = _make_vector_id("video.mp4", "video_frame", 0, 12.0)
        id3 = _make_vector_id("video.mp4", "video_frame", 1, 12.0)
        self.assertEqual(id1, id2, "Same inputs must produce same ID")
        self.assertNotEqual(id1, id3, "Different index must produce different ID")
        self.assertEqual(len(id1), 32, "ID should be 32-char hex")

    def test_id_varies_by_timestamp(self):
        from src.rag.vector_store import _make_vector_id
        id_a = _make_vector_id("v.mp4", "video_frame", 0, 3.0)
        id_b = _make_vector_id("v.mp4", "video_frame", 0, 6.0)
        self.assertNotEqual(id_a, id_b)


class TestPipelineHelpers(unittest.TestCase):
    def test_extract_sources_from_metadata(self):
        from src.rag.pipeline import RAGPipeline
        from unittest.mock import MagicMock

        # Create mock documents
        doc1 = MagicMock()
        doc1.metadata = {"source": "/data/report.pdf", "type": "text"}
        doc2 = MagicMock()
        doc2.metadata = {"source": "/data/game.mp3", "type": "audio"}

        video_matches = [
            {"metadata": {"source": "/data/video.mp4", "timestamp": 10.0}},
            {"metadata": {"source": "/data/video.mp4", "timestamp": 20.0}},  # duplicate source
        ]

        sources = RAGPipeline._extract_sources([doc1], [doc2], video_matches)
        self.assertEqual(sources, ["report.pdf", "game.mp3", "video.mp4"])

    def test_group_temporal_context_empty(self):
        from src.rag.pipeline import RAGPipeline
        result = RAGPipeline._group_temporal_context([])
        self.assertEqual(result, [])


class TestTextLoaderMetadata(unittest.TestCase):
    def test_chunk_metadata_enrichment(self):
        """Verify chunks get proper metadata after splitting."""
        import tempfile
        from src.ingestion.text_loader import UniversalTextLoader

        loader = UniversalTextLoader(chunk_size=50, chunk_overlap=10)

        # Create a temp file with enough text to produce multiple chunks
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, dir=".") as f:
            f.write("This is a test sentence. " * 20)
            tmp_path = f.name

        try:
            chunks = loader.load_file(tmp_path)
            self.assertGreater(len(chunks), 1, "Should produce multiple chunks")

            for i, chunk in enumerate(chunks):
                self.assertEqual(chunk.metadata["chunk_index"], i)
                self.assertEqual(chunk.metadata["chunk_count"], len(chunks))
                self.assertEqual(chunk.metadata["type"], "text")
                self.assertIn("source_basename", chunk.metadata)
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
