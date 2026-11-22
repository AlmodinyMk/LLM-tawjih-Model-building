"""Tests for the architecture lab (tawjih arch): specifications as
three-faced files, exact parameter counts checked against published
models, variants, export, and a real build when PyTorch is installed."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tawjih import Base  # noqa: E402
from tawjihlib import arch  # noqa: E402

import importlib.util
HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch",
                                                        "transformers"))


class Counts(unittest.TestCase):
    """Parameter counts of published models, from their configurations."""

    def test_llama_2_7b(self):
        self.assertEqual(arch.count(arch.PRESETS["llama-7b-like"])["total"],
                         6738415616)

    def test_tinyllama_gqa(self):
        spec = dict(arch.DEFAULT, layers=22, hidden=2048, heads=32,
                    kv_heads=4, ffn_hidden=5632, vocab=32000,
                    tie_embeddings=False)
        self.assertEqual(arch.count(spec)["total"], 1100048384)

    def test_gpt2_small(self):
        self.assertEqual(arch.count(arch.PRESETS["gpt2-like"])["total"],
                         124439808)

    def test_moe_active(self):
        spec = arch.PRESETS["moe-small"]
        c = arch.count(spec)
        self.assertLess(c["active"], c["total"])
        dense = arch.count(dict(spec, moe=None))
        self.assertEqual(c["total"] - dense["total"],
                         spec["layers"] * (7 * 3 * spec["hidden"] *
                                           spec["ffn_hidden"]
                                           + spec["hidden"] * 8))

    def test_problems(self):
        self.assertEqual(arch.problems(arch.DEFAULT), [])
        bad = dict(arch.DEFAULT, hidden=770, kv_heads=5, ffn="tanh",
                   extra=1)
        text = " ".join(arch.problems(bad))
        self.assertIn("unknown key", text)
        bad.pop("extra")
        text = " ".join(arch.problems(bad))
        self.assertIn("ffn must be", text)
        bad["ffn"] = "swiglu"
        text = " ".join(arch.problems(bad))
        self.assertIn("divide evenly", text)
        self.assertIn("kv_heads", text)
        self.assertTrue(arch.problems(dict(arch.DEFAULT,
                                           moe={"experts": 2, "top_k": 3})))

    def test_hf_config_families(self):
        self.assertEqual(arch.hf_config(arch.DEFAULT)["model_type"], "llama")
        self.assertEqual(arch.hf_config(arch.PRESETS["moe-small"])
                         ["model_type"], "mixtral")
        self.assertEqual(arch.hf_config(arch.PRESETS["gpt2-like"])
                         ["model_type"], "gpt2")
        import tawjih
        with self.assertRaises(tawjih.Die):
            arch.hf_config(dict(arch.DEFAULT, norm="layernorm"))


class Lab(Base):
    def test_create_derive_diff_mutate(self):
        code, out, err = self.cli("arch", "create", "Apex", "--preset",
                                  "tiny")
        self.assertEqual(code, 0, err)
        st = self.store()
        n = st.nodes[st.resolve("1")]
        self.assertIn("architecture", n.tags)
        self.assertIn("Hypothesis", n.human)
        spec = json.loads(n.instructions)
        self.assertEqual(spec["layers"], 4)
        code, out, err = self.cli("arch", "derive", "1", "Apex 2", "--set",
                                  "kv_heads=1", "--set", "layers=6")
        self.assertEqual(code, 0, err)
        self.assertIn("MQA", out)
        st = self.store()
        d = st.nodes[st.resolve("1.1")]
        self.assertEqual(json.loads(d.instructions)["derived_from"], n.id)
        self.assertIn((n.id, "derived-from"), d.links)
        out = self.cli("arch", "diff", "1", "1.1")[1]
        self.assertIn("layers          4 -> 6", out)
        self.assertIn("parameters", out)
        out = self.cli("arch", "mutate", "1", "--count", "3")[1]
        self.assertIn("3 variant(s)", out)
        self.assertEqual(len(self.store().nodes), 2)
        self.cli("arch", "mutate", "1", "--count", "3", "--yes")
        self.assertEqual(len(self.store().nodes), 5)
        self.assertIn("Apex: layers", self.cli("arch", "list")[1])

    def test_deep_lineage_stays_a_valid_tree(self):
        self.cli("arch", "create", "A", "--preset", "tiny")
        ref = "1"
        for i in range(7):
            code, _, err = self.cli("arch", "derive", ref, "d%d" % i)
            self.assertEqual(code, 0, err)
            ref = self.store().number(max(self.store().nodes.values(),
                                          key=lambda n: n.order * 0 +
                                          n.level).id)
        self.assertIn("0 error(s)", self.cli("check")[1])

    def test_set_validate_export(self):
        self.cli("arch", "create", "A", "--preset", "tiny")
        code, _, err = self.cli("arch", "set", "1", "hidden=250")
        self.assertNotEqual(code, 0)
        self.assertIn("divide evenly", err)
        self.assertEqual(self.cli("arch", "set", "1", "layers=8")[0], 0)
        self.assertEqual(self.cli("arch", "validate", "1")[0], 0)
        out = self.root / "hf"
        self.assertEqual(self.cli("arch", "export", "1", "-o",
                                  str(out))[0], 0)
        cfg = json.loads((out / "config.json").read_text())
        self.assertEqual(cfg["num_hidden_layers"], 8)
        self.cli("new", "plain")
        self.assertIn("not an architecture", self.cli("arch", "show",
                                                      "2")[2])

    def test_build_plan_only(self):
        self.cli("arch", "create", "A", "--preset", "tiny")
        code, out, err = self.cli("arch", "build", "1")
        self.assertEqual(code, 0, err)
        self.assertIn("plan only", out)
        self.cli("arch", "create", "B", "--preset", "llama-7b-like")
        self.assertIn("large", self.cli("arch", "build", "2")[2])

    @unittest.skipUnless(HAVE_TORCH, "needs torch and transformers")
    def test_real_build(self):
        self.cli("arch", "create", "A", "--preset", "tiny", "--set",
                 "layers=2")
        code, out, err = self.cli("arch", "build", "1", "--run", "--python",
                                  sys.executable)
        self.assertEqual(code, 0, err + out)
        self.assertIn("as counted", out)
        self.assertIn("measured", self.cli("arch", "show", "1")[1])


if __name__ == "__main__":
    unittest.main()
