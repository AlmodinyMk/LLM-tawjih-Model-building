"""Tests for plugins/neural_bridge.py, without any training library: the
data it compiles, the scripts and commands it writes, and the adapter
registry with its routing."""
import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "plugins"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import neural_bridge as nb  # noqa: E402
from test_tawjih import Base  # noqa: E402

EX = "Q: 2x+4=10\nA: x = 3\nE: 3\nBAD: x = 7\nBAD: x = 2\n\nس: 3+4؟\nج: 7\n"


class Bridge(Base):
    def setUp(self):
        super().setUp()
        self.tree()
        self.fill("1.1.1", instructions="Answer with x only.", examples=EX)
        self.fill("2", instructions="Explain.",
                  examples="Q: what is a loop?\nA: a repeated block")
        self.cli("contract", "2", "--split", "eval")
        self.ds = self.root / "datasets"

    def nb(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = nb.main(["-p", self.dir] + list(args))
        return code, out.getvalue(), err.getvalue()

    def rows(self, name):
        return [json.loads(x) for x in (self.ds / name).read_text(
            encoding="utf-8").splitlines() if x.strip()]

    def compile(self):
        code, out, err = self.nb("compile", "-o", str(self.ds),
                                 "--with-rules")
        self.assertEqual(code, 0, err + out)

    def fake_adapter(self, name, rank=8, base="tiny"):
        d = self.root / "ad" / name
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text(json.dumps({"r": rank}))
        self.assertEqual(self.nb("lora", "add", name, str(d),
                                 "--base-model", base)[0], 0)
        reg = json.loads((self.root / "models" / "registry.json")
                         .read_text())
        reg["adapters"][name]["rank"] = rank
        (self.root / "models" / "registry.json").write_text(json.dumps(reg))
        return d

    # ------------------------------------------------------------ compile
    def test_compile_splits_rules_and_preference_pairs(self):
        self.compile()
        train = self.rows("sft_train.jsonl")
        self.assertEqual(len(train), 2)
        self.assertEqual(train[0]["messages"][0]["content"],
                         "Answer with x only.\n\n2x+4=10")
        self.assertEqual(train[0]["messages"][1]["content"], "x = 3")
        self.assertEqual(len(self.rows("sft_eval.jsonl")), 1)
        self.assertEqual([r["rejected"] for r in self.rows(
            "dpo_train.jsonl")], ["x = 7", "x = 2"])
        man = json.loads((self.ds / "sft_train.jsonl.manifest.json")
                         .read_text())
        self.assertIn(self.store().resolve("1.1.1"), man["from"])

    def test_compile_reports_questions_in_train_and_eval(self):
        self.fill("2", examples="Q: 2x+4=10\nA: x = 3")
        code, out, _ = self.nb("compile", "-o", str(self.ds))
        self.assertEqual(code, 1)
        self.assertIn("in both", out)

    # -------------------------------------------------------------- train
    def test_train_plan_writes_script_config_and_lineage(self):
        self.compile()
        code, out, err = self.nb("train", "alg", "--base-model", "tiny",
                                 "--dataset", str(self.ds /
                                                  "sft_train.jsonl"),
                                 "--qlora", "--rank", "8")
        self.assertEqual(code, 0, err)
        adir = self.root / "models" / "adapters" / "alg"
        cfg = json.loads((adir / "train_config.json").read_text())
        self.assertEqual((cfg["lora_r"], cfg["lora_alpha"], cfg["qlora"],
                          cfg["format"]), (8, 16, True, "chat"))
        self.assertTrue((adir / "dataset.manifest.json").is_file())
        compile((adir / "train.py").read_text(), "train.py", "exec")
        self.assertIn("plan only", out)
        self.assertFalse((adir / "adapter_config.json").exists())
        self.assertIn("alg", json.loads((self.root / "models" /
                                         "registry.json").read_text())
                      ["adapters"])

    def test_dpo_needs_preference_data(self):
        self.compile()
        code, _, err = self.nb("train", "d", "--base-model", "tiny",
                               "--method", "dpo", "--dataset",
                               str(self.ds / "sft_train.jsonl"))
        self.assertEqual(code, 1)
        self.assertIn("DPO needs", err)
        code, _, err = self.nb("train", "d", "--base-model", "tiny",
                               "--method", "dpo", "--dataset",
                               str(self.ds / "dpo_train.jsonl"))
        self.assertEqual(code, 0, err)
        cfg = json.loads((self.root / "models" / "adapters" / "d" /
                          "train_config.json").read_text())
        self.assertEqual((cfg["method"], cfg["format"]), ("dpo", "dpo"))

    def test_engines_unsloth_and_axolotl(self):
        self.compile()
        ds = str(self.ds / "sft_train.jsonl")
        self.nb("train", "u", "--base-model", "tiny", "--dataset", ds,
                "--engine", "unsloth", "--qlora")
        script = (self.root / "models" / "adapters" / "u" / "train.py"
                  ).read_text()
        compile(script, "train.py", "exec")
        self.assertLess(script.index("from unsloth"),
                        script.index("import trl"))
        self.nb("train", "x", "--base-model", "tiny", "--dataset", ds,
                "--engine", "axolotl", "--eval", str(self.ds /
                                                     "sft_eval.jsonl"))
        yml = (self.root / "models" / "adapters" / "x" / "axolotl.yml"
               ).read_text()
        self.assertIn('adapter: "lora"', yml)
        self.assertIn('    type: "chat_template"', yml)
        self.assertIn("test_datasets:", yml)
        code, _, err = self.nb("train", "y", "--base-model", "tiny",
                               "--dataset", str(self.ds /
                                                "dpo_train.jsonl"),
                               "--method", "dpo", "--engine", "axolotl")
        self.assertIn("SFT configs only", err)

    # ----------------------------------------------- convert and quantize
    def llama_dir(self):
        d = self.root / "llama"
        (d / "build" / "bin").mkdir(parents=True)
        (d / "convert_hf_to_gguf.py").write_text(
            "import sys\nout = sys.argv[sys.argv.index('--outfile') + 1]\n"
            "open(out, 'w').write('gguf')\n")
        q = d / "build" / "bin" / "llama-quantize"
        q.write_text("#!/bin/sh\necho \"$@\" > \"$2\"\n")
        q.chmod(0o755)
        return d

    def test_convert_runs_llama_cpp_script(self):
        d = self.llama_dir()
        model = self.root / "merged"
        model.mkdir()
        out = self.root / "m-f16.gguf"
        code, o, err = self.nb("--llama-cpp", str(d), "convert", str(model),
                               "-o", str(out))
        self.assertEqual(code, 0, err)
        self.assertFalse(out.exists())
        code, o, err = self.nb("--llama-cpp", str(d), "convert", str(model),
                               "-o", str(out), "--run")
        self.assertEqual(code, 0, err)
        self.assertEqual(out.read_text(), "gguf")
        self.assertIn("conversion script was not found",
                      self.nb("convert", str(model))[2])

    @unittest.skipIf(os.name == "nt", "the stand-in program is a shell "
                     "script")
    def test_quantize_finds_build_bin_and_checks_type(self):
        d = self.llama_dir()
        src = self.root / "m-f16.gguf"
        src.write_text("x")
        code, out, err = self.nb("--llama-cpp", str(d), "quantize",
                                 str(src), "--run")
        self.assertEqual(code, 0, err)
        self.assertTrue((self.root / "m-Q4_K_M.gguf").is_file())
        self.assertIn("unknown type", self.nb("--llama-cpp", str(d),
                                              "quantize", str(src),
                                              "--type", "Q9")[2])
        self.assertIn("takes a GGUF", self.nb("--llama-cpp", str(d),
                                              "quantize", "model.bin")[2])

    # --------------------------------------------------------------- lora
    def test_add_takes_paths_from_the_current_folder(self):
        d = self.root / "elsewhere" / "ad"
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text("{}")
        old = os.getcwd()
        os.chdir(self.root / "elsewhere")
        try:
            self.assertEqual(self.nb("lora", "add", "x", "ad")[0], 0)
        finally:
            os.chdir(old)
        self.assertIn("x=%s" % d.resolve(), self.nb("lora", "serve")[1])

    def test_bind_route_by_id_survives_moves(self):
        self.fake_adapter("alg")
        self.assertIn("not a PEFT adapter", self.nb(
            "lora", "add", "bad", str(self.root))[2])
        self.nb("lora", "bind", "1.1", "alg")
        nid = self.store().resolve("1.1.1")
        self.assertIn("-> alg (bound at 1.1", self.nb("lora", "route",
                                                      nid)[1])
        self.cli("new", "Other")                      # tree 3
        self.cli("move", "1.1", "--to", "3")
        self.assertIn("-> alg", self.nb("lora", "route", nid)[1])
        self.assertIn("no adapter", self.nb("lora", "route", "1")[1])
        self.assertIn("is bound", self.nb("lora", "rm", "alg")[2])

    def test_apply_sets_model_lines_and_respects_own_models_and_locks(self):
        self.fake_adapter("alg")
        self.cli("new", "own", "--parent", "1.1")
        self.fill("1.1.2", instructions="x", model="llama3")
        self.cli("new", "locked", "--parent", "1.1")
        self.fill("1.1.3", instructions="y")
        self.cli("lock", "1.1.3")
        self.nb("lora", "bind", "1", "alg")
        code, out, _ = self.nb("lora", "apply")
        self.assertIn("would set", out)
        self.assertEqual(self.store().nodes[self.store().resolve("1.1.1")]
                         .model, "")
        code, out, _ = self.nb("lora", "apply", "--yes")
        self.assertEqual(code, 1)                     # the locked file
        st = self.store()
        self.assertEqual(st.nodes[st.resolve("1.1.1")].model, "alg")
        self.assertEqual(st.nodes[st.resolve("1.1.2")].model, "llama3")
        self.assertIn("kept", out)
        self.assertIn("not changed", out)
        self.nb("lora", "unbind", "1")
        self.nb("lora", "apply", "--yes")
        self.assertEqual(self.store().nodes[st.resolve("1.1.1")].model, "")

    def test_serve_vllm_and_ollama(self):
        self.fake_adapter("alg")
        self.fake_adapter("ar")
        out = self.nb("lora", "serve")[1]
        self.assertIn("vllm serve tiny --enable-lora", out)
        self.assertIn("alg=", out)
        self.assertIn("ar=", out)
        out = self.nb("lora", "serve", "--server", "ollama")[1]
        self.assertIn("no GGUF adapter yet", out)
        reg = json.loads((self.root / "models" / "registry.json").read_text())
        reg["adapters"]["alg"]["gguf"] = "alg.gguf"
        (self.root / "models" / "registry.json").write_text(json.dumps(reg))
        out = self.nb("lora", "serve", "--server", "ollama",
                      "--ollama-base", "llama3.2")[1]
        self.assertIn("ollama create alg", out)
        mf = (self.root / "models" / "ollama" / "alg.Modelfile").read_text()
        self.assertTrue(mf.startswith("FROM llama3.2\nADAPTER "))
        self.assertIn("cannot route", self.nb("lora", "serve", "--server",
                                              "llama")[1])

    def test_fuse_checks_bases_and_ranks_and_writes_spec(self):
        self.fake_adapter("a", rank=8)
        self.fake_adapter("b", rank=16)
        self.fake_adapter("c", rank=8, base="other")
        self.assertIn("different bases", self.nb("lora", "fuse", "n",
                                                 "a:0.5", "c:0.5")[2])
        self.assertIn("same rank", self.nb("lora", "fuse", "n", "a:0.7",
                                           "b:0.3")[2])
        code, out, err = self.nb("lora", "fuse", "n", "a:0.7", "b:0.3",
                                 "--how", "cat")
        self.assertEqual(code, 0, err)
        spec = json.loads((self.root / "models" / "scripts" /
                           "fuse-n.json").read_text())
        self.assertEqual([p["weight"] for p in spec["parts"]], [0.7, 0.3])
        compile((self.root / "models" / "scripts" / "fuse.py").read_text(),
                "fuse.py", "exec")

    @unittest.skipIf(os.name == "nt", "the stand-in program is a shell "
                     "script")
    def test_run_records_paths_relative_to_the_project(self):
        fake = self.root / "fakepy"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o755)
        self.fake_adapter("a", rank=8)
        self.fake_adapter("b", rank=8)
        code, out, err = self.nb("--python", str(fake), "lora", "fuse", "n",
                                 "a:0.5", "b:0.5", "--run")
        self.assertEqual(code, 0, err)
        self.nb("--python", str(fake), "merge", "a", "--run")
        reg = json.loads((self.root / "models" / "registry.json").read_text())
        self.assertEqual(Path(reg["adapters"]["n"]["path"]).as_posix(),
                         "models/adapters/fused-n/n")
        self.assertEqual(Path(reg["adapters"]["a"]["merged"]).as_posix(),
                         "models/merged/a")
        self.assertIn("is a fusion (linear) of: a 0.50, b 0.50",
                      self.nb("lora", "lineage", "n")[1])

    def test_lineage_traces_an_adapter_to_its_files(self):
        self.compile()
        self.nb("train", "alg", "--base-model", "tiny", "--dataset",
                str(self.ds / "sft_train.jsonl"))
        out = self.nb("lora", "lineage", "alg")[1]
        self.assertIn("المعادلات", out)
        self.assertNotIn("changed since", out)
        self.fill("1.1.1", human="new")
        self.assertIn("changed since", self.nb("lora", "lineage", "alg")[1])

    def test_merge_plan(self):
        self.fake_adapter("alg")
        code, out, err = self.nb("merge", "alg")
        self.assertEqual(code, 0, err)
        self.assertIn("tiny", out)
        self.assertIn("merge.py", out)
        compile((self.root / "models" / "scripts" / "merge.py").read_text(),
                "merge.py", "exec")


class PeftMethods(Bridge):
    """--peft: the configuration each method gets, the checks before
    training, and which commands each kind of adapter can go through."""

    def train(self, *args):
        return self.nb("train", "x", "--base-model", "tiny", "--dataset",
                       str(self.ds / "sft_train.jsonl"), "--overwrite",
                       *args)

    def adapter(self, name, cfg):
        d = self.root / "ad" / name
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text(json.dumps(cfg))
        self.assertEqual(self.nb("lora", "add", name, str(d),
                                 "--base-model", "tiny")[0], 0)
        return d

    def test_every_method_is_planned(self):
        self.compile()
        adir = self.root / "models" / "adapters" / "x"
        for m in nb.PEFT_METHODS:
            code, out, err = self.train("--peft", m)
            self.assertEqual(code, 0, m + err)
            cfg = json.loads((adir / "train_config.json").read_text())
            self.assertEqual(cfg["peft"], m)
            script = (adir / "train.py").read_text()
            compile(script, "train.py", "exec")
            self.assertIn('"%s": (' % m, script)
            self.assertIn("path_initial_model_for_weight_conversion", script)
        self.assertIn("virtual tokens", out)
        reg = json.loads((self.root / "models" / "registry.json")
                         .read_text())
        self.assertEqual(reg["adapters"]["x"]["peft"], "ptuning")

    def test_options_reach_the_configuration(self):
        self.compile()
        code, out, err = self.train(
            "--peft", "dora", "--target", "q_proj,v_proj", "--peft-arg",
            "init_lora_weights=gaussian", "--peft-arg", "layers_to_transform"
            "=[0, 1]", "--peft-arg", "bias=lora_only")
        self.assertEqual(code, 0, err)
        cfg = json.loads((self.root / "models" / "adapters" / "x" /
                          "train_config.json").read_text())
        self.assertEqual(cfg["target"], "q_proj,v_proj")
        self.assertEqual(cfg["peft_args"], {
            "init_lora_weights": "gaussian", "layers_to_transform": [0, 1],
            "bias": "lora_only"})
        code, out, err = self.train("--peft", "prompt", "--virtual-tokens",
                                    "12")
        self.assertIn("12 virtual tokens", out)

    def test_checks(self):
        self.compile()
        for args, word in (
                (["--peft-arg", "no equals"], "KEY=VALUE"),
                (["--peft", "hra", "--rank", "7"], "even"),
                (["--peft", "lora", "--virtual-tokens", "5"], "prompt or"),
                (["--peft", "prompt", "--target", "mlp"], "no --target"),
                (["--peft", "loha", "--engine", "unsloth"], "--engine torch"),
                (["--peft", "ia3", "--engine", "axolotl"], "--engine torch"),
                (["--peft", "pissa", "--qlora"], "--qlora")):
            code, _, err = self.train(*args)
            self.assertNotEqual(code, 0, args)
            self.assertIn(word, err, args)
        ds = self.ds / "dpo.jsonl"
        ds.write_text('{"prompt": "p", "chosen": "a", "rejected": "b"}\n')
        code, _, err = self.nb("train", "x", "--base-model", "tiny",
                               "--dataset", str(ds), "--method", "dpo",
                               "--peft", "olora")
        self.assertIn("reference model", err)
        code, out, err = self.train("--peft", "rslora", "--engine",
                                    "unsloth")
        self.assertEqual(code, 0, err)
        code, out, err = self.train("--peft", "dora", "--engine", "axolotl")
        self.assertEqual(code, 0, err)
        yml = (self.root / "models" / "adapters" / "x" / "axolotl.yml")
        self.assertIn("peft_use_dora: true", yml.read_text())

    def test_kind_of_an_adapter(self):
        d = self.root / "k"
        d.mkdir()
        for cfg, kind in (({"peft_type": "LORA"}, "lora"),
                          ({"peft_type": "LORA", "use_dora": True}, "dora"),
                          ({"peft_type": "LORA", "use_rslora": True},
                           "rslora"),
                          ({"peft_type": "LOHA"}, "loha"),
                          ({"peft_type": "PROMPT_TUNING"}, "prompt"),
                          ({"peft_type": "P_TUNING"}, "ptuning"),
                          ({"peft_type": "NEWONE"}, "newone")):
            (d / "adapter_config.json").write_text(json.dumps(cfg))
            self.assertEqual(nb.adapter_kind(d), kind)
        self.assertEqual(nb.adapter_kind(self.root / "none", {"peft": "vera"}),
                         "vera")

    def test_what_each_kind_can_go_through(self):
        self.adapter("plain", {"peft_type": "LORA", "r": 8})
        self.adapter("pissa", {"peft_type": "LORA", "r": 16})
        self.adapter("dora", {"peft_type": "LORA", "r": 8, "use_dora": True})
        self.adapter("loha", {"peft_type": "LOHA", "r": 8})
        self.adapter("soft", {"peft_type": "PROMPT_TUNING"})
        out = self.nb("lora", "list")[1]
        self.assertIn("loha", out)
        self.assertIn("dora", out)
        code, _, err = self.nb("merge", "soft")
        self.assertIn("nothing to merge", err)
        code, out, err = self.nb("merge", "loha")
        self.assertEqual(code, 0, err)
        self.assertIn("fuse combines LoRA", self.nb(
            "lora", "fuse", "n", "plain:0.5", "loha:0.5")[2])
        self.assertIn("same rank (8, 16)", self.nb(
            "lora", "fuse", "n", "plain:0.5", "pissa:0.5")[2])
        out = self.nb("lora", "serve")[1]
        self.assertIn("left out: dora, loha, soft", out)
        self.assertIn("plain=", out)
        self.assertIn("--max-lora-rank 16", out)
        lc = self.llama_dir()
        (lc / "convert_lora_to_gguf.py").write_text("")
        code, _, err = self.nb("--llama-cpp", str(lc), "convert", "dora",
                               "--lora")
        self.assertIn("LoRA adapters only", err)


if __name__ == "__main__":
    unittest.main()
