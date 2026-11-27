"""Tests for 0.17: settings passed to a provider as they are (num_ctx for
Ollama), training and model arguments passed to TRL and transformers, your
own script as a job, and model servers (tawjih server) driven with a
stand-in engine."""
import contextlib
import http.server
import io
import json
import os
import socket
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "plugins"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import neural_bridge as nb  # noqa: E402
from test_tawjih import Base, tawjih  # noqa: E402
from tawjihlib import providers, serving  # noqa: E402


class Recorder:
    """An HTTP server that answers as Ollama and as an OpenAI-compatible
    server, and keeps every request body."""
    def __init__(self):
        bodies = self.bodies = []
        headers = self.headers = []

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n))
                bodies.append((self.path, body))
                headers.append(dict(self.headers))
                if self.path.endswith(":generateContent"):
                    text = body["contents"][0]["parts"][0]["text"]
                    out = {"modelVersion": "gemini-x-001",
                           "usageMetadata": {"promptTokenCount": 7,
                                             "candidatesTokenCount": 2,
                                             "thoughtsTokenCount": 3}}
                    if text == "blocked":
                        out["promptFeedback"] = {"blockReason": "SAFETY"}
                    else:
                        out["candidates"] = [{"finishReason": "STOP",
                                              "content": {"parts": [
                                                  {"text": "thinking",
                                                   "thought": True},
                                                  {"text": "gem ok"}]}}]
                elif self.path == "/api/chat":
                    out = {"model": body["model"], "message": {
                        "content": "ok"}, "prompt_eval_count": 5,
                        "eval_count": 1}
                else:
                    out = {"model": body["model"], "choices": [{
                        "message": {"content": "ok"}}], "usage": {}}
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever,
                         daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class ProviderOptions(Base):
    def setUp(self):
        super().setUp()
        self.srv = Recorder()
        self.addCleanup(self.srv.close)
        providers._told_ctx.clear()
        os.environ.pop("TAWJIH_NUM_CTX", None)

    def call(self, name, prompt="hello"):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            answer, _ = providers.call_model(self.dir, name, prompt,
                                             model="m")
        return answer, err.getvalue()

    def test_ollama_gets_num_ctx_and_options(self):
        code, out, err = self.cli("provider", "add", "loc", "--kind",
                                  "ollama", "--base-url", self.srv.url,
                                  "--num-ctx", "32768", "--option",
                                  "temperature=0.2", "--option", "seed=7")
        self.assertEqual(code, 0, err)
        self.assertIn("num_ctx 32768", out)
        self.call("loc")
        path, body = self.srv.bodies[-1]
        self.assertEqual(path, "/api/chat")
        self.assertEqual(body["options"], {"temperature": 0.2, "seed": 7,
                                           "num_ctx": 32768})
        # an empty value removes an option; --num-ctx 0 removes num_ctx
        self.cli("provider", "set", "loc", "--option", "seed=",
                 "--num-ctx", "0")
        self.call("loc")
        self.assertEqual(self.srv.bodies[-1][1]["options"],
                         {"temperature": 0.2})

    def test_a_long_prompt_is_warned_about(self):
        self.cli("provider", "add", "loc", "--kind", "ollama", "--base-url",
                 self.srv.url)
        _, err = self.call("loc", "x" * 40000)
        self.assertIn("4096 tokens by default", err)
        self.assertNotIn("options", self.srv.bodies[-1][1])
        _, err = self.call("loc", "x" * 40000)
        self.assertEqual(err, "")                     # said once
        self.cli("provider", "set", "loc", "--num-ctx", "8192")
        _, err = self.call("loc", "x" * 40000)
        self.assertIn("num_ctx is 8192", err)
        self.cli("provider", "set", "loc", "--num-ctx", "16384")
        _, err = self.call("loc", "x" * 40000)
        self.assertEqual(err, "")                     # it fits now
        os.environ["TAWJIH_NUM_CTX"] = "65536"
        self.addCleanup(os.environ.pop, "TAWJIH_NUM_CTX", None)
        self.cli("provider", "set", "loc", "--num-ctx", "0")
        self.call("loc")
        self.assertEqual(self.srv.bodies[-1][1]["options"]["num_ctx"], 65536)

    def test_openai_options_go_in_the_request(self):
        self.cli("provider", "add", "srv", "--kind", "openai", "--base-url",
                 self.srv.url + "/v1", "--option", "top_p=0.9", "--option",
                 'chat_template_kwargs={"enable_thinking": false}',
                 "--option", "max_tokens=512", "--option", "model=other")
        self.call("srv")
        path, body = self.srv.bodies[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(body["top_p"], 0.9)
        self.assertEqual(body["max_tokens"], 512)
        self.assertEqual(body["chat_template_kwargs"],
                         {"enable_thinking": False})
        self.assertEqual(body["model"], "m")          # stays Tawjih's

    def test_gemini(self):
        os.environ["TEST_GEM_KEY"] = "g-secret"
        self.addCleanup(os.environ.pop, "TEST_GEM_KEY", None)
        code, out, err = self.cli(
            "provider", "add", "gem", "--kind", "gemini", "--base-url",
            self.srv.url + "/v1beta", "--model", "gemini-2.5-flash",
            "--api-key-env", "TEST_GEM_KEY", "--option",
            'generationConfig={"temperature": 0.2}')
        self.assertEqual(code, 0, err)
        answer, usage = providers.call_model(self.dir, "gem", "hello")
        self.assertEqual(answer, "gem ok")             # no thought parts
        path, body = self.srv.bodies[-1]
        self.assertEqual(path, "/v1beta/models/gemini-2.5-flash:"
                         "generateContent")
        self.assertEqual(body["contents"], [{"role": "user", "parts": [
            {"text": "hello"}]}])
        self.assertEqual(body["generationConfig"], {"temperature": 0.2})
        sent = {k.lower(): v for k, v in self.srv.headers[-1].items()}
        self.assertEqual(sent["x-goog-api-key"], "g-secret")
        self.assertNotIn("g-secret", path)              # never in the address
        self.assertEqual((usage["in"], usage["out"], usage["served"]),
                         (7, 5, "gemini-x-001"))
        with self.assertRaises(RuntimeError) as e:
            providers.call_model(self.dir, "gem", "blocked")
        self.assertIn("SAFETY", str(e.exception))
        with self.assertRaises(tawjih.Die):
            with contextlib.redirect_stderr(io.StringIO()):
                providers.call_model(self.dir, "gem", "x", model="a/b")
        out = self.cli("provider", "list")[1]
        self.assertIn("kind gemini", out)

    def test_no_secret_in_options(self):
        for bad in ("api_key=x", "token=y", "Authorization=z"):
            code, _, err = self.cli("provider", "add", "s", "--kind",
                                    "openai", "--option", bad)
            self.assertEqual(code, 1, bad)
            self.assertIn("key or password", err)
        code, _, err = self.cli("provider", "add", "s", "--kind", "openai",
                                "--option", "bad key=1")
        self.assertEqual(code, 1)

    def test_key_values(self):
        self.assertEqual(tawjih.key_values(["a=1", "b=x y", "c=", "d=null",
                                            "e=[1,2]", "f.g=true"]),
                         {"a": 1, "b": "x y", "c": None, "d": None,
                          "e": [1, 2], "f.g": True})


class TrainingArgs(Base):
    def setUp(self):
        super().setUp()
        self.ds = self.root / "sft.jsonl"
        self.ds.write_text(json.dumps({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"}]}) + "\n", encoding="utf-8")
        os.environ["TAWJIH_RUNTIMES"] = str(self.root / "rt.json")
        self.addCleanup(os.environ.pop, "TAWJIH_RUNTIMES", None)

    def nb(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = nb.main(["-p", self.dir] + list(args))
        return code, out.getvalue(), err.getvalue()

    def test_train_and_model_args_and_gpus(self):
        ds_conf = self.root / "ds.json"
        ds_conf.write_text('{"zero_optimization": {"stage": 3}}')
        code, out, err = self.nb(
            "train", "a1", "--base-model", "tiny", "--dataset", str(self.ds),
            "--gpus", "2", "--train-arg", "deepspeed=%s" % ds_conf,
            "--train-arg", "gradient_checkpointing=true",
            "--model-arg", "attn_implementation=flash_attention_2")
        self.assertEqual(code, 0, err)
        self.assertIn("torch.distributed.run --standalone --nproc_per_node 2",
                      out)
        self.assertIn("training arguments as given", out)
        adir = self.root / "models" / "adapters" / "a1"
        cfg = json.loads((adir / "train_config.json").read_text())
        self.assertEqual(cfg["train_args"], {
            "deepspeed": str(ds_conf.resolve()),
            "gradient_checkpointing": True})
        self.assertEqual(cfg["model_args"],
                         {"attn_implementation": "flash_attention_2"})
        self.assertEqual(cfg["gpus"], 2)
        script = (adir / "train.py").read_text()
        compile(script, "train.py", "exec")
        self.assertIn('load.update(CFG.get("model_args") or {})', script)
        self.assertIn("kw.update(mine)", script)
        self.assertIn("WORLD_SIZE", script)

    def test_a_job_carries_its_config_files(self):
        ds_conf = self.root / "ds.json"
        ds_conf.write_text("{}")
        code, out, err = self.nb(
            "train", "a2", "--base-model", "tiny", "--dataset", str(self.ds),
            "--gpus", "4", "--train-arg", "deepspeed=%s" % ds_conf,
            "--runtime", "local")
        self.assertEqual(code, 0, err)
        jdir = next((self.root / "models" / "jobs").iterdir())
        self.assertTrue((jdir / "conf" / "ds.json").is_file())
        cfg = json.loads((jdir / "train_config.json").read_text())
        self.assertEqual(cfg["train_args"], {"deepspeed": "conf/ds.json"})
        rec = json.loads((jdir / "job.json").read_text())
        self.assertEqual(rec["gpus"], 4)
        self.assertIn("--nproc_per_node", rec["command"])

    def test_refusals(self):
        base = ["train", "x", "--base-model", "tiny", "--dataset",
                str(self.ds)]
        for extra, says in (
                (["--gpus", "2", "--engine", "unsloth"], "--gpus with"),
                (["--train-arg", "output_dir=/tmp/x"], "adapter's folder"),
                (["--train-arg", "no-equals"], "KEY=VALUE"),
                (["--model-arg", "pretrained_model_name_or_path=y"],
                 "--base-model"),
                (["--engine", "axolotl", "--model-arg", "a=1"],
                 "--model-arg is for")):
            code, out, err = self.nb(*(base + extra))
            self.assertEqual(code, 1, extra)
            self.assertIn(says, err + out, extra)

    def test_axolotl_takes_its_own_keys(self):
        code, _, err = self.nb("train", "ax", "--base-model", "tiny",
                               "--dataset", str(self.ds), "--engine",
                               "axolotl", "--train-arg", "flash_attention=true")
        self.assertEqual(code, 0, err)
        yml = (self.root / "models" / "adapters" / "ax" / "axolotl.yml"
               ).read_text()
        self.assertIn("flash_attention: true", yml)

    def test_the_trl_config_check_is_strict(self):
        """The config() of the written script, run against a stand-in TRL:
        the person's own arguments must all be known."""
        import dataclasses
        import types

        @dataclasses.dataclass
        class SFTConfig:
            output_dir: str = ""
            max_length: int = 0
            use_liger_kernel: bool = False
        trl = types.SimpleNamespace(__version__="9.9")
        src = nb.TRAIN_COMMON
        start = src.index("def config(")
        end = src.index("\n\n\nARGS")
        ns = {"dataclasses": dataclasses, "sys": sys, "trl": trl,
              "CFG": {"train_args": {"use_liger_kernel": True}}}
        exec(src[start:end], ns)
        with contextlib.redirect_stdout(io.StringIO()):
            c = ns["config"](SFTConfig, output_dir="o", max_length=8,
                             report_to=[])
        self.assertTrue(c.use_liger_kernel)
        ns["CFG"] = {"train_args": {"no_such_field": 1}}
        with self.assertRaises(SystemExit) as e:
            ns["config"](SFTConfig, output_dir="o")
        self.assertIn("no_such_field", str(e.exception))

    @unittest.skipIf(os.name == "nt", "jobs run with sh")
    def test_your_own_script(self):
        prog = self.root / "mine.py"
        prog.write_text(
            "import json, os, sys\n"
            "c = json.load(open(sys.argv[1]))\n"
            "os.makedirs(c['output_dir'], exist_ok=True)\n"
            "rows = sum(1 for d in c['data'] for _ in open(d))\n"
            "json.dump({'rows': rows, 'lr': c['args']['lr'], 'files': "
            "c['files']}, open(os.path.join(c['output_dir'], 'r.json'), "
            "'w'))\n", encoding="utf-8")
        extra = self.root / "extra.txt"
        extra.write_text("x")
        code, out, err = self.nb("script", str(prog), "--data", str(self.ds),
                                 "--set", "lr=0.001", "--with-file",
                                 str(extra), "--run")
        self.assertEqual(code, 0, err + out)
        res = json.loads((self.root / "models" / "scripts" / "mine" /
                          "r.json").read_text())
        self.assertEqual(res, {"rows": 1, "lr": 0.001,
                               "files": ["files/extra.txt"]})
        jdir = next((self.root / "models" / "jobs").iterdir())
        rec = json.loads((jdir / "job.json").read_text())
        self.assertEqual((rec["kind"], rec["script"], rec["args"]),
                         ("script", "mine.py", {"lr": 0.001}))
        self.assertIn("by", rec)
        self.assertEqual(len(rec["sha256"]), 64)

    def test_script_refusals(self):
        self.assertEqual(self.nb("script", "nope.py")[0], 1)
        prog = self.root / "p.py"
        prog.write_text("")
        self.assertEqual(self.nb("script", str(prog), "--name", "a b")[0], 1)
        self.assertEqual(self.nb("script", str(prog), "--data", "none")[0], 1)


STAND_IN = r'''
import http.server, json, os, sys
port = int(sys.argv[sys.argv.index("--port") + 1])
model = sys.argv[sys.argv.index("--model") + 1]
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        print("request", self.command, self.path, flush=True)
    def send(self, code, body, ctype="application/json"):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health":
            return self.send(200, "{}")
        if self.path == "/v1/models":
            return self.send(200, json.dumps({"data": [{"id": model}]}))
        if self.path == "/metrics":
            return self.send(200, "# HELP t\ntokens_total 42\nrunning 1\n",
                             "text/plain")
        self.send(404, "{}")
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n).decode()
        if self.path == "/flush_cache":
            return self.send(200, json.dumps({
                "auth": self.headers.get("Authorization"),
                "admin": self.headers.get("x-admin-key"), "body": body}))
        if self.path == "/v1/chat/completions":
            req = json.loads(body)
            return self.send(200, json.dumps({
                "model": req["model"], "usage": {},
                "choices": [{"message": {"content": "served " + req["model"]
                             + " " + os.environ.get("CUDA_VISIBLE_DEVICES",
                                                    "-")}}]}))
        self.send(404, "{}")
print("stand-in on", port, "gpus", os.environ.get("CUDA_VISIBLE_DEVICES"),
      flush=True)
http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Servers(Base):
    def setUp(self):
        super().setUp()
        self.prog = self.root / "stand_in.py"
        self.prog.write_text(STAND_IN, encoding="utf-8")
        self.port = free_port()

    def add(self, *extra):
        cmd = '"%s" "%s" --port {port} --model {model}' % (
            Path(sys.executable).as_posix(), self.prog.as_posix())
        return self.cli("server", "add", "fake", "--engine", "custom",
                        "--command", cmd, "--model", "m1", "--port",
                        str(self.port), *extra)

    def test_engines_command_lines(self):
        cases = {
            "vllm": (dict(model="Q", tp=4, pp=2, context=65536, mem=0.9,
                          kv_cache="fp8", served_name="q",
                          api_key_env="K"),
                     "vllm serve Q --host 127.0.0.1 --port 8000 "
                     "--tensor-parallel-size 4 --pipeline-parallel-size 2 "
                     "--max-model-len 65536 --gpu-memory-utilization 0.9 "
                     "--kv-cache-dtype fp8 --served-model-name q",
                     {"VLLM_API_KEY": "$K"}),
            "sglang": (dict(model="/m", tp=2, mem=0.88, api_key_env="K",
                            python="py"),
                       "py -m sglang.launch_server --model-path /m --host "
                       "127.0.0.1 --port 8000 --enable-metrics --tp-size 2 "
                       "--mem-fraction-static 0.88 --api-key $K", {}),
            "lmdeploy": (dict(model="/m", kv_cache="int8", context=32768),
                         "lmdeploy serve api_server /m --server-name "
                         "127.0.0.1 --server-port 8000 --session-len 32768 "
                         "--quant-policy 8", {}),
            "trtllm": (dict(model="/e", tp=8),
                       "trtllm-serve /e --host 127.0.0.1 --port 8000 "
                       "--tp_size 8", {}),
            "ollama": (dict(model="q", context=32768, kv_cache="q8_0"),
                       "ollama serve",
                       {"OLLAMA_HOST": "127.0.0.1:8000",
                        "OLLAMA_CONTEXT_LENGTH": "32768",
                        "OLLAMA_KV_CACHE_TYPE": "q8_0",
                        "OLLAMA_FLASH_ATTENTION": "1"}),
            "llamacpp": (dict(model="m.gguf", kv_cache="q4_0", context=8192),
                         "llama-server -m m.gguf --host 127.0.0.1 --port "
                         "8000 --metrics -c 8192 --cache-type-k q4_0 "
                         "--cache-type-v q4_0", {}),
        }
        for engine, (extra, want, env) in cases.items():
            cfg = dict(engine=engine, host="127.0.0.1", port=8000, **extra)
            serving.check_server(cfg)
            argv, got_env = serving.command(cfg)
            self.assertEqual(" ".join(argv), want, engine)
            for k, v in env.items():
                self.assertEqual(got_env[k], v, engine)
        # the key's value is used only when the server starts
        argv, env = serving.command(dict(cases["vllm"][0], engine="vllm",
                                         host="h", port=1), "secret")
        self.assertEqual(env["VLLM_API_KEY"], "secret")

    def test_settings_an_engine_does_not_take(self):
        for argv, says in (
                (["--engine", "trtllm", "--model", "x", "--kv-cache", "fp8"],
                 "does not take --kv-cache"),
                (["--engine", "litellm", "--model", "c.yaml", "--tp", "2"],
                 "does not take --tp"),
                (["--engine", "vllm", "--model", "x", "--kv-cache", "int4"],
                 "--kv-cache takes"),
                (["--engine", "vllm"], "--model is needed"),
                (["--engine", "tabby", "--model", "x"], "--dir"),
                (["--engine", "custom"], "--command"),
                (["--engine", "vllm", "--model", "x", "--mem", "2"],
                 "fraction"),
                (["--engine", "vllm", "--model", "x", "--gpus", "a"],
                 "GPU numbers"),
                (["--engine", "vllm", "--model", "x", "--env", "HF_TOKEN=1"],
                 "secret")):
            code, out, err = self.cli("server", "add", "s", *argv)
            self.assertEqual(code, 1, argv)
            self.assertIn(says, err, argv)
        self.assertFalse((self.root / "servers.json").exists())

    def test_start_status_use_and_stop(self):
        code, out, err = self.add("--gpus", "1", "--provider")
        self.assertEqual(code, 0, err)
        self.assertIn("provider fake: openai", out)
        code, out, err = self.cli("server", "start", "fake", "--wait", "30")
        self.addCleanup(self.cli, "server", "stop", "fake")
        self.assertEqual(code, 0, err + out)
        self.assertIn("is ready", out)
        out = self.cli("server", "status", "fake")[1]
        self.assertIn("running (pid", out)
        self.assertIn("health 200", out)
        self.assertIn("models: m1", out)
        # it serves the project as a provider, on the GPUs it was given
        answer, _ = providers.call_model(self.dir, "fake", "hi")
        self.assertEqual(answer, "served m1 1")
        out = self.cli("server", "metrics", "fake", "--grep", "tokens")[1]
        self.assertEqual(out.strip(), "tokens_total 42")
        os.environ["TEST_ADMIN"] = "adm"
        self.addCleanup(os.environ.pop, "TEST_ADMIN", None)
        code, out, err = self.cli("server", "api", "fake", "POST",
                                  "/flush_cache", "--data", '{"a": 1}',
                                  "--key-env", "TEST_ADMIN",
                                  "--key-header", "x-admin-key")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out), {"auth": None, "admin": "adm",
                                           "body": '{"a": 1}'})
        self.assertEqual(self.cli("server", "api", "fake", "GET",
                                  "/nothing")[0], 1)
        self.assertEqual(self.cli("server", "start", "fake")[0], 1)
        out = self.cli("server", "logs", "fake", "--lines", "50")[1]
        self.assertIn("stand-in on", out)
        self.assertIn("request GET /health", out)
        code, out, err = self.cli("server", "stop", "fake")
        self.assertEqual(code, 0, err)
        self.assertIn("stopped", out)
        out = self.cli("server", "status", "fake")[1]
        self.assertIn("not started here", out)
        self.assertIn("no answer", out)
        hist = "".join(f.read_text(encoding="utf-8") for f in
                       (self.root / "history").glob("servers-*.jsonl"))
        actions = [json.loads(x)["action"] for x in hist.splitlines()]
        self.assertEqual(actions, ["start", "api", "api", "stop"])
        self.assertIn('"by"', hist)

    def test_a_used_port_is_named(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", self.port))
            s.listen(1)
            self.add()
            code, out, err = self.cli("server", "start", "fake")
            self.assertEqual(code, 1)
            self.assertIn("port %d is in use" % self.port, err)

    def test_a_server_that_ends_at_once(self):
        cmd = '"%s" -c "import sys; sys.exit(3)" {port}' % Path(
            sys.executable).as_posix()
        self.cli("server", "add", "dead", "--engine", "custom", "--command",
                 cmd, "--port", str(self.port))
        code, out, err = self.cli("server", "start", "dead", "--wait", "20")
        self.assertEqual(code, 1)
        self.assertIn("ended with status 3", out)

    def test_gpus_auto(self):
        gpus = [{"index": 0, "name": "A", "total": 81920, "free": 2000,
                 "busy": 90},
                {"index": 1, "name": "A", "total": 81920, "free": 80000,
                 "busy": 0},
                {"index": 2, "name": "A", "total": 81920, "free": 60000,
                 "busy": 5},
                {"index": 3, "name": "A", "total": 81920, "free": 79000,
                 "busy": 0}]
        self.assertEqual(serving.choose_gpus({"tp": 2}, gpus), "1,3")
        self.assertEqual(serving.choose_gpus({"tp": 2, "pp": 1,
                                              "min_free": 70000}, gpus),
                         "1,3")
        self.assertEqual(serving.choose_gpus({}, gpus), "1")
        with self.assertRaises(tawjih.Die):
            with contextlib.redirect_stderr(io.StringIO()):
                serving.choose_gpus({"tp": 4, "min_free": 70000}, gpus)
        # auto is kept as auto, and chosen only when the server starts
        self.cli("server", "add", "v", "--engine", "vllm", "--model", "Q",
                 "--tp", "2", "--gpus", "auto", "--min-free", "40000")
        cfg = json.loads((self.root / "servers.json").read_text())[
            "servers"]["v"]
        self.assertEqual((cfg["gpus"], cfg["min_free"]), ("auto", 40000))
        self.assertNotIn("CUDA_VISIBLE_DEVICES",
                         serving.command(cfg)[1])
        code, _, err = self.cli("server", "unit", "v")
        self.assertEqual(code, 1)
        self.assertIn("give their numbers", err)

    @unittest.skipIf(os.name == "nt", "the stand-in programs are scripts")
    def test_prepare_with_a_stand_in(self):
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        tool = bin_dir / "lmdeploy"
        tool.write_text("#!/bin/sh\necho \"lmdeploy $@\"\n"
                        "echo \"gpus $CUDA_VISIBLE_DEVICES\"\n")
        tool.chmod(0o755)
        old = os.environ["PATH"]
        os.environ["PATH"] = str(bin_dir) + os.pathsep + old
        self.addCleanup(os.environ.__setitem__, "PATH", old)
        self.cli("server", "add", "lm", "--engine", "lmdeploy", "--model",
                 "/m/q", "--gpus", "2")
        code, out, err = self.cli("server", "prepare", "lm", "--out",
                                  "/m/q-awq", "--tool-arg=--w-group-size",
                                  "--tool-arg", "128")
        self.assertEqual(code, 0, err + out)
        self.assertIn("lmdeploy lite auto_awq /m/q --work-dir /m/q-awq "
                      "--w-group-size 128", out)
        self.assertIn("gpus 2", out)
        self.assertIn("--model-format --pass=awq", out)
        hist = "".join(f.read_text(encoding="utf-8") for f in
                       (self.root / "history").glob("servers-*.jsonl"))
        self.assertIn('"action": "prepare"', hist)
        self.assertEqual(self.cli("server", "prepare", "lm")[0], 1)
        self.cli("server", "add", "v", "--engine", "vllm", "--model", "x")
        code, _, err = self.cli("server", "prepare", "v", "--out", "o")
        self.assertIn("nothing to prepare", err)

    def test_bench(self):
        self.add("--provider")
        self.assertEqual(self.cli("server", "start", "fake", "--wait",
                                  "30")[0], 0)
        self.addCleanup(self.cli, "server", "stop", "fake")
        self.tree()
        self.fill("1.1.1", instructions="Solve it.\n" * 50)
        code, out, err = self.cli("server", "bench", "fake", "--from-file",
                                  "1.1.1", "--input", "2x=4", "--requests",
                                  "6", "--concurrency", "3", "--vary")
        self.assertEqual(code, 0, err + out)
        self.assertIn("ok 6, failed 0", out)
        self.assertIn("each different", out)
        rec = json.loads(next((self.root / "history").glob(
            "bench-fake-*.json")).read_text(encoding="utf-8"))
        self.assertEqual(rec["summary"]["ok"], 6)
        self.assertEqual(rec["summary"]["concurrency"], 3)
        self.assertIn("by", rec)
        self.assertTrue(rec["source"].startswith("file "))
        self.assertGreater(rec["prompt_chars"], 500)
        self.assertEqual(self.cli("server", "bench", "fake")[0], 1)
        self.cli("server", "stop", "fake")
        code, _, err = self.cli("server", "bench", "fake", "--prompt", "x")
        self.assertEqual(code, 1)
        self.assertIn("start it first", err)

    def test_units(self):
        self.cli("server", "add", "v", "--engine", "vllm", "--model",
                 "Qwen/Q", "--gpus", "0,1", "--tp", "2", "--api-key-env",
                 "VKEY", "--pass=--enable-prefix-caching")
        code, out, err = self.cli("server", "unit", "v")
        self.assertEqual(code, 0, err)
        self.assertIn("EnvironmentFile=/etc/tawjih/v.env", out)
        self.assertIn("holds VLLM_API_KEY=<the key>", out)
        self.assertIn("Environment=CUDA_VISIBLE_DEVICES=0,1", out)
        self.assertIn("--tensor-parallel-size 2 --enable-prefix-caching", out)
        self.assertTrue((self.root / "servers" / "v" / "tawjih-v.service")
                        .is_file())
        out = self.cli("server", "unit", "v", "--docker")[1]
        self.assertIn("--gpus '\"device=0,1\"'", out)
        self.assertIn("-e VLLM_API_KEY=$VKEY", out)
        self.assertIn("vllm/vllm-openai", out)
        self.assertIn("--host 0.0.0.0", out)
        self.cli("server", "add", "s", "--engine", "sglang", "--model", "m",
                 "--api-key-env", "SK")
        out = self.cli("server", "unit", "s")[1]
        self.assertIn("--api-key ${SK}", out)
        self.assertEqual(self.cli("server", "unit", "s", "--docker")[0], 1)

    def test_list_set_rm_and_ollama_provider(self):
        self.cli("server", "add", "ol", "--engine", "ollama", "--model",
                 "qwen3:32b", "--context", "32768", "--port", "11500")
        self.cli("server", "set", "ol", "--kv-cache", "q8_0")
        out = self.cli("server", "list")[1]
        self.assertIn("ol", out)
        self.assertIn("stopped", out)
        self.cli("server", "provider", "ol", "local")
        p = json.loads((self.root / "providers.json").read_text())
        self.assertEqual(p["providers"]["local"], {
            "kind": "ollama", "base_url": "http://127.0.0.1:11500",
            "model": "qwen3:32b", "num_ctx": 32768})
        self.assertEqual(self.cli("server", "rm", "ol")[0], 0)
        self.assertEqual(self.cli("server", "show", "ol")[0], 1)
        self.assertEqual(self.cli("خادم", "list")[0], 0)


if __name__ == "__main__":
    unittest.main()


class SizeConditions(Base):
    """IF tokens > N, chars, lines: the size of a step's output decides."""
    def setUp(self):
        super().setUp()
        self.cli("new", "root")
        for name in ("pass", "long", "short"):
            self.cli("new", name, "--parent", "1")
        self.fill("1.1", code="output = input")
        self.fill("1.2", code="output = 'LONG'")
        self.fill("1.3", code="output = 'SHORT'")

    def run_path(self, text):
        code, out, err = self.cli("run", "r", "--input", text,
                                  "--allow-code", "--backend", "fake")
        self.assertEqual(code, 0, err)
        return out

    def test_route_by_tokens(self):
        code, out, err = self.cli("path", "r", "1.1", "TO", "1.2", "IF",
                                  "tokens", ">", "100", "ELSE", "1.3")
        self.assertEqual(code, 0, err + out)
        self.assertIn("LONG", self.run_path("x" * 404))   # 101 tokens
        self.assertIn("SHORT", self.run_path("x" * 400))
        saved = (self.root / "paths" / "r.path").read_text(encoding="utf-8")
        self.assertIn("tokens>100", saved)

    def test_arabic_and_other_measures(self):
        for words, long_text, short_text in (
                (["إذا", "الرموز", ">", "10"], "y" * 44, "y" * 40),
                (["IF", "lines>=3"], "a\nb\nc", "a\nb"),
                (["IF", "chars", "<=", "5"], "abc", "abcdef")):
            code, out, err = self.cli("path", "r", "1.1", "TO", "1.2",
                                      *words, "ELSE", "1.3")
            self.assertEqual(code, 0, (words, err))
            self.assertIn("LONG", self.run_path(long_text), words)
            self.assertIn("SHORT", self.run_path(short_text), words)

    def test_holds_and_plain_words(self):
        self.assertTrue(tawjih.condition_holds("x" * 8, "tokens>=2"))
        self.assertFalse(tawjih.condition_holds("x" * 8, "tokens>2"))
        self.assertTrue(tawjih.condition_holds("a\nb", "lines<3"))
        # a word 'tokens' with no number stays a word
        code, _, err = self.cli("path", "w", "1.1", "TO", "1.2", "IF",
                                "tokens", "ELSE", "1.3")
        self.assertEqual(code, 0, err)
        self.assertIn("SHORT", self.cli("run", "w", "--input", "none here",
                                        "--allow-code")[1])
        self.assertIn("LONG", self.cli("run", "w", "--input", "tokens",
                                       "--allow-code")[1])

    def test_iterate_until_size(self):
        self.fill("1.1", code="output = input + 'x' * 10")
        code, _, err = self.cli("path", "it", "1.1", "/", "ITERATE", "9",
                                "UNTIL", "chars", ">", "35")
        self.assertEqual(code, 0, err)
        out = self.cli("run", "it", "--input", "s", "--allow-code")[1]
        self.assertIn("s" + "x" * 40, out)
        self.assertNotIn("x" * 41, out)


class UsingProvider(Base):
    """A step whose USING names a provider goes to that provider."""
    def setUp(self):
        super().setUp()
        self.a, self.b = Recorder(), Recorder()
        self.addCleanup(self.a.close)
        self.addCleanup(self.b.close)
        self.cli("provider", "add", "big", "--kind", "openai", "--base-url",
                 self.a.url + "/v1", "--model", "big-model")
        self.cli("provider", "add", "small", "--kind", "openai",
                 "--base-url", self.b.url + "/v1", "--model", "small-model")
        self.cli("new", "root")
        self.cli("new", "one", "--parent", "1")
        self.cli("new", "two", "--parent", "1")
        self.fill("1.1", instructions="first")
        self.fill("1.2", instructions="second")
        self.cli("path", "p", "1.1", "USING", "big", "TO", "1.2",
                 "بواسطة", "small")

    def test_each_step_to_its_provider(self):
        code, out, err = self.cli("run", "p", "--input", "q", "--backend",
                                  "fake")
        self.assertEqual(code, 0, err)
        self.assertEqual([b["model"] for _, b in self.a.bodies],
                         ["big-model"])
        self.assertEqual([b["model"] for _, b in self.b.bodies],
                         ["small-model"])
        self.assertIn("first", self.a.bodies[0][1]["messages"][0]["content"])
        self.assertEqual(self.prompts, [])           # fake got nothing
        # --model still overrides the provider's model
        self.cli("run", "p", "--input", "q2", "--backend", "fake",
                 "--model", "m9")
        self.assertEqual(self.a.bodies[-1][1]["model"], "m9")

    def test_a_dry_run_calls_none(self):
        code, out, err = self.cli("run", "p", "--input", "q", "--backend",
                                  "dry")
        self.assertEqual(code, 0, err)
        self.assertEqual((self.a.bodies, self.b.bodies), ([], []))


class ServerControl(Base):
    def test_tune(self):
        self.cli("server", "add", "sg", "--engine", "sglang", "--model", "m",
                 "--python", "py")
        code, out, err = self.cli("server", "tune", "sg", "--set",
                                  "mem-fraction-static=0.8", "--set",
                                  "schedule_policy=lpm", "--set",
                                  "enable-prefix-caching=true", "--set",
                                  "tp=2")
        self.assertEqual(code, 0, err)
        cfg = json.loads((self.root / "servers.json").read_text())[
            "servers"]["sg"]
        self.assertEqual((cfg["mem"], cfg["tp"]), (0.8, 2))
        self.assertEqual(cfg["args"], {"schedule_policy": "lpm",
                                       "enable-prefix-caching": True})
        self.assertIn("--tp-size 2 --mem-fraction-static 0.8 "
                      "--schedule-policy lpm --enable-prefix-caching", out)
        self.cli("server", "tune", "sg", "--set", "schedule_policy=",
                 "--set", "enable-prefix-caching=false")
        cfg = json.loads((self.root / "servers.json").read_text())[
            "servers"]["sg"]
        self.assertNotIn("args", cfg)
        # trtllm writes its own flags with underscores
        self.cli("server", "add", "t", "--engine", "trtllm", "--model", "e")
        out = self.cli("server", "tune", "t", "--set", "max-batch-size=64")[1]
        self.assertIn("--max_batch_size 64", out)
        # settings are still checked
        code, _, err = self.cli("server", "tune", "sg", "--set", "mem=3")
        self.assertEqual(code, 1)
        self.cli("server", "add", "o", "--engine", "ollama")
        code, _, err = self.cli("server", "tune", "o", "--set", "x=1")
        self.assertIn("environment variables", err)

    def test_kill_shows_first(self):
        import subprocess
        proc = subprocess.Popen([sys.executable, "-c",
                                 "import time; time.sleep(60)"])
        self.addCleanup(proc.kill)
        code, out, err = self.cli("server", "kill", str(proc.pid))
        self.assertEqual(code, 1)
        self.assertIn("process %d" % proc.pid, out)
        self.assertIn("--yes", out)
        self.assertIsNone(proc.poll())
        code, out, err = self.cli("server", "kill", str(proc.pid), "--yes")
        self.assertEqual(code, 0, err)
        proc.wait(timeout=20)
        hist = "".join(f.read_text(encoding="utf-8") for f in
                       (self.root / "history").glob("servers-*.jsonl"))
        self.assertIn('"action": "kill"', hist)
        for bad in ("1", str(os.getpid()), "x"):
            self.assertEqual(self.cli("server", "kill", bad, "--yes")[0], 1)

    def test_all_units(self):
        self.cli("server", "add", "a", "--engine", "vllm", "--model", "A",
                 "--gpus", "0")
        self.cli("server", "add", "b", "--engine", "vllm", "--model", "B",
                 "--gpus", "1", "--port", "8001")
        self.cli("server", "add", "c", "--engine", "vllm", "--model", "C",
                 "--gpus", "auto", "--port", "8002")
        code, out, err = self.cli("server", "unit", "--all")
        self.assertEqual(code, 0, err)
        d = self.root / "servers" / "units"
        self.assertIn("c left out", out)
        target = (d / "tawjih-servers.target").read_text()
        self.assertIn("Wants=tawjih-a.service tawjih-b.service", target)
        unit = (d / "tawjih-a.service").read_text()
        self.assertIn("PartOf=tawjih-servers.target", unit)
        self.assertIn("WantedBy=tawjih-servers.target", unit)


class Batch(Base):
    def test_inputs_shared_among_providers(self):
        a, b = Recorder(), Recorder()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        self.cli("provider", "add", "g0", "--kind", "openai", "--base-url",
                 a.url + "/v1", "--model", "m")
        self.cli("provider", "add", "g1", "--kind", "openai", "--base-url",
                 b.url + "/v1", "--model", "m")
        self.cli("new", "root")
        self.fill("1", instructions="answer")
        self.cli("path", "p", "1")
        inputs = self.root / "in"
        inputs.mkdir()
        for i in range(6):
            (inputs / ("q%d.txt" % i)).write_text("question %d" % i,
                                                  encoding="utf-8")
        (inputs / "skip.md").write_text("no")
        out = self.root / "answers"
        code, text, err = self.cli("batch", "p", "--inputs-dir", str(inputs),
                                   "--pattern", "*.txt", "--backends",
                                   "g0,g1", "--out", str(out))
        self.assertEqual(code, 0, err + text)
        self.assertIn("6 of 6 ok", text)
        rep = json.loads((out / "batch.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(r["input"] for r in rep["runs"]),
                         ["q%d.txt" % i for i in range(6)])
        self.assertEqual({r["backend"] for r in rep["runs"]}, {"g0", "g1"})
        self.assertEqual(len(a.bodies) + len(b.bodies), 6)
        self.assertEqual((out / "q3.txt.out").read_text(encoding="utf-8")
                         .strip(), "ok")
        self.assertIn("by", rep)
        sent = sorted(body["messages"][0]["content"].split()[-1]
                      for _, body in a.bodies + b.bodies)
        self.assertEqual(sent, [str(i) for i in range(6)])

    def test_refusals(self):
        self.cli("new", "root")
        self.cli("path", "p", "1")
        d = self.root / "in"
        d.mkdir()
        for argv, says in (
                (["nope", "--inputs-dir", str(d), "--backends", "dry"],
                 "no saved path"),
                (["p", "--inputs-dir", str(d), "--backends", "dry"],
                 "no input"),
                (["p", "--inputs-dir", str(d / "x"), "--backends", "dry"],
                 "no folder")):
            code, _, err = self.cli("batch", *argv)
            self.assertEqual(code, 1, argv)
            self.assertIn(says, err, argv)
        (d / "a.txt").write_text("x")
        code, _, err = self.cli("batch", "p", "--inputs-dir", str(d),
                                "--backends", "nobody")
        self.assertIn("not a provider", err)
