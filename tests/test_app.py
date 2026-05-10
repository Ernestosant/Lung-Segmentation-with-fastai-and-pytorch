from __future__ import annotations

import importlib.util
import tempfile
import unittest

from lung_segmentation.infer import list_model_checkpoints


class AppSmokeTests(unittest.TestCase):
    def test_model_registry_handles_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(list_model_checkpoints(tmp), [])

    @unittest.skipIf(importlib.util.find_spec("gradio") is None, "gradio is not installed")
    def test_create_demo_without_models(self) -> None:
        from lung_segmentation.app import create_demo

        with tempfile.TemporaryDirectory() as tmp:
            demo = create_demo(tmp)
            self.assertIsNotNone(demo)


if __name__ == "__main__":
    unittest.main()
