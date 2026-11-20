"""Live calls to the providers in a project's providers.json. Skipped unless
TAWJIH_LIVE_TESTS=1; they send a short prompt and may cost money.

    TAWJIH_LIVE_TESTS=1 TAWJIH_LIVE_PROJECT=path/to/project \\
        python -m unittest tests.test_live -v
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tawjihlib import providers  # noqa: E402

LIVE = os.environ.get("TAWJIH_LIVE_TESTS") == "1"
PROJECT = os.environ.get("TAWJIH_LIVE_PROJECT", ".")


@unittest.skipUnless(LIVE, "set TAWJIH_LIVE_TESTS=1 to call real providers")
class Live(unittest.TestCase):
    def test_each_provider_answers(self):
        names = sorted(providers.load_providers(PROJECT)["providers"])
        if not names:
            self.skipTest("no providers.json in " + PROJECT)
        for name in names:
            with self.subTest(provider=name):
                out, usage = providers.call_model(
                    PROJECT, name, 'Answer with exactly: {"status": "ok"}',
                    fallback=False)
                self.assertIn("ok", out.lower())
                self.assertEqual(usage["provider"], name)


if __name__ == "__main__":
    unittest.main()
