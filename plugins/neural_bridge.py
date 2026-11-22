#!/usr/bin/env python3
"""neural_bridge - from a Tawjih project to trained adapters and back.

A bridge, not a trainer: standard library only. The heavy work (PyTorch,
transformers, PEFT, TRL, Unsloth, axolotl, llama.cpp) is done by programs
this bridge writes or calls, in the Python you name with --python. Every
command that would start such a program only shows its plan unless --run
is given.

    compile    the examples of the project as SFT, evaluation and DPO data
    env        which engines the training Python has, and the GPU
    setup      installs what the work needs into the training Python, here
               or on a runtime (torch, PEFT, TRL, huggingface_hub, vLLM...)
    train      a PEFT adapter: LoRA, QLoRA, DoRA, rsLoRA, PiSSA, OLoRA,
               LoHa, LoKr, IA3, VeRA, OFT, BOFT, FourierFT, HRA, RandLoRA,
               LayerNorm tuning, prompt tuning, P-tuning (--peft; engines:
               torch, unsloth, axolotl)
    merge      an adapter merged into its base model (PyTorch + PEFT)
    push       an adapter, a merged model or a dataset to the Hugging Face
               Hub (private unless --public, with a model card)
    pipeline   compile, train, merge and push in one go, here or as one
               job on a runtime
    convert    a merged model, or an adapter, to GGUF (llama.cpp)
    quantize   a GGUF model to Q4_K_M, Q8_0, ... (llama.cpp)
    lora       the project's adapters: add, rm, list, bind, unbind, route,
               apply (sets each file's model: line), fuse, serve
               (--serverless: vLLM set up for cold starts), lineage
    (train and merge take --runtime NAME: a job on this computer, another
    one over ssh, or a service you drive; see `tawjih runtime`)

Routing works through what Tawjih already has: a file's `model:` line
chooses the model of its step, and a server that offers each adapter under
its own name (vLLM with --enable-lora, or Ollama with one model per
adapter) answers with that adapter. `lora apply` writes those lines from
the bindings, which are kept by stable id and inherited down the tree.

Usage: python plugins/neural_bridge.py -p PROJECT COMMAND ...

Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tawjih  # noqa: E402
from tawjihlib import runtimes  # noqa: E402

ENGINES = ("torch", "unsloth", "axolotl")
QUANTS = ("Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", "Q4_K_S",
          "Q4_K_M", "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0",
          "F16", "BF16", "IQ4_XS", "IQ4_NL")
FUSE_HOW = ("linear", "cat", "svd", "ties", "dare_linear", "dare_ties",
            "magnitude_prune")
UNSLOTH_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                   "up_proj", "down_proj"]


class BridgeError(Exception):
    pass


def fail(msg):
    raise BridgeError(msg)


def models_dir(a):
    return Path(a.project) / "models"


def show_cmd(cmd):
    return " ".join(shlex.quote(str(c)) for c in cmd)


def run_or_plan(a, cmd, what, cwd=None):
    """Print the command; run it with --run. Returns the exit status."""
    print("%s: %s" % (what, show_cmd(cmd)))
    if not a.run:
        print("(plan only; add --run to start it)")
        return 0
    try:
        return subprocess.run([str(c) for c in cmd], cwd=cwd).returncode
    except FileNotFoundError as e:
        fail("cannot start %s: %s" % (cmd[0], e))


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def read_jsonl_head(path, count=5):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= count:
                    break
    return rows


# ------------------------------------------------------------------ compile
def _dataset(a, split, out, dpo=None):
    ns = argparse.Namespace(
        project=a.project, source="examples", branch=a.branch, split=split,
        with_rules=a.with_rules, repeat_by_weight=a.repeat_by_weight and
        split == "train", dpo=dpo, path=None, step=None,
        include_failed=False, format=a.format, meta=False, out=str(out))
    tawjih.cmd_dataset(ns)


def cmd_compile(a):
    """sft_train.jsonl (split train, with weights), sft_eval.jsonl (split
    eval) and dpo_train.jsonl (from BAD: lines of train files), each with a
    manifest, through Tawjih's own dataset compiler; then a check that no
    question is in both."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _dataset(a, "train", out / "sft_train.jsonl", out / "dpo_train.jsonl")
    _dataset(a, "eval", out / "sft_eval.jsonl")

    def questions(f):
        qs = set()
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                q = r["prompt"] if "prompt" in r else r["messages"][-2][
                    "content"]
                qs.add(" ".join(q.split()).casefold())
        return qs
    both = questions(out / "sft_train.jsonl") & questions(out /
                                                          "sft_eval.jsonl")
    if both:
        print("warning: %d question(s) are in both the training and the "
              "evaluation data, so the evaluation would test what was "
              "trained (see: tawjih.py check)" % len(both))
        return 1
    return 0


# ---------------------------------------------------------------------- env
ENV_PROBE = r"""
import importlib, json, shutil, sys
out = {"python": sys.version.split()[0], "modules": {}}
for m in ("torch", "transformers", "peft", "trl", "datasets", "accelerate",
          "bitsandbytes", "unsloth", "axolotl"):
    try:
        mod = importlib.import_module(m)
        out["modules"][m] = getattr(mod, "__version__", "installed")
    except Exception as e:
        out["modules"][m] = None
try:
    import torch
    out["cuda"] = torch.cuda.is_available()
    out["gpus"] = [torch.cuda.get_device_name(i)
                   for i in range(torch.cuda.device_count())]
    out["mps"] = bool(getattr(torch.backends, "mps", None)
                      and torch.backends.mps.is_available())
except Exception:
    out["cuda"], out["gpus"], out["mps"] = False, [], False
print(json.dumps(out))
"""


def find_llama_tool(a, name):
    """A llama.cpp program or script: in --llama-cpp, its build/bin, or
    PATH."""
    exe = name + (".exe" if os.name == "nt" and not name.endswith(".py")
                  else "")
    if a.llama_cpp:
        for d in (Path(a.llama_cpp), Path(a.llama_cpp) / "build" / "bin",
                  Path(a.llama_cpp) / "bin"):
            if (d / exe).is_file():
                return d / exe
    found = shutil.which(exe) or shutil.which(name)
    return Path(found) if found else None


def cmd_env(a):
    try:
        r = subprocess.run([a.python, "-c", ENV_PROBE], capture_output=True,
                           text=True, timeout=300)
        info = json.loads(r.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as e:
        fail("could not ask %s: %s" % (a.python, e))
    print("training Python: %s (%s)" % (a.python, info["python"]))
    for m, v in info["modules"].items():
        print("  %-13s %s" % (m, v or "-"))
    print("GPU: %s" % (", ".join(info["gpus"]) if info["cuda"] else
                       "Apple MPS" if info["mps"] else "none found"))
    for tool in ("llama-quantize", "convert_hf_to_gguf.py",
                 "convert_lora_to_gguf.py"):
        p = find_llama_tool(a, tool)
        print("  %-24s %s" % (tool, p or "-"))
    for tool in ("ollama", "vllm", "axolotl"):
        print("  %-24s %s" % (tool, shutil.which(tool) or "-"))
    ready = {"torch": all(info["modules"][m] for m in
                          ("torch", "transformers", "peft", "trl",
                           "datasets")),
             "unsloth": bool(info["modules"]["unsloth"] and
                             info["modules"]["trl"]),
             "axolotl": bool(info["modules"]["axolotl"])}
    print("engines ready: %s" % (", ".join(k for k, v in ready.items() if v)
                                 or "none"))
    if not (info["cuda"] or info["mps"]):
        print("note: without a GPU only very small models can be trained")
    return 0


# -------------------------------------------------------------------- train
# PEFT methods: name -> what it is. All train with --engine torch; the
# ones in PLAIN_LORA save a plain LoRA adapter (served, fused, converted to
# GGUF); prompt and ptuning add virtual tokens and cannot be merged.
PEFT_METHODS = {
    "lora": "LoRA: low-rank matrices beside the weights",
    "dora": "DoRA: LoRA with the weights split into size and direction",
    "rslora": "rank-stabilised LoRA: alpha/sqrt(r), for high ranks",
    "pissa": "PiSSA: LoRA started from the main singular vectors",
    "olora": "OLoRA: LoRA started from an orthonormal (QR) basis",
    "loha": "LoHa: a Hadamard product of two low-rank pairs",
    "lokr": "LoKr: a Kronecker product, very few parameters",
    "ia3": "IA3: learned vectors that scale activations",
    "vera": "VeRA: shared random matrices, small trained vectors",
    "oft": "OFT: an orthogonal rotation of the weights (--rank = block)",
    "boft": "BOFT: a butterfly orthogonal rotation (--rank = block)",
    "fourierft": "FourierFT: a few trained frequencies of the change",
    "hra": "HRA: Householder reflections (--rank even)",
    "randlora": "RandLoRA: full rank from random bases",
    "ln": "LayerNorm tuning: only the normalisation layers",
    "prompt": "prompt tuning: trained virtual tokens before the input",
    "ptuning": "P-tuning: virtual tokens made by a small encoder",
}
PLAIN_LORA = ("lora", "rslora", "pissa", "olora")
NO_MERGE = ("prompt", "ptuning")
PEFT_TYPES = {"LOHA": "loha", "LOKR": "lokr", "IA3": "ia3", "VERA": "vera",
              "OFT": "oft", "BOFT": "boft", "FOURIERFT": "fourierft",
              "HRA": "hra", "RANDLORA": "randlora", "LN_TUNING": "ln",
              "PROMPT_TUNING": "prompt", "P_TUNING": "ptuning"}


def adapter_rank(path, entry=None):
    """The rank an adapter really has (a converted PiSSA or OLoRA adapter
    has twice the rank it was trained with)."""
    try:
        cfg = json.loads((Path(path) / "adapter_config.json").read_text(
            encoding="utf-8"))
        return int(cfg["r"])
    except (OSError, ValueError, KeyError, TypeError):
        return (entry or {}).get("rank")


def adapter_kind(path, entry=None):
    """The PEFT method of an adapter folder, from its adapter_config.json
    (a converted PiSSA is a plain LoRA); else what the registry says."""
    f = Path(path) / "adapter_config.json"
    try:
        cfg = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return (entry or {}).get("peft", "lora")
    t = str(cfg.get("peft_type", "LORA")).upper()
    if t == "LORA":
        return "dora" if cfg.get("use_dora") else (
            "rslora" if cfg.get("use_rslora") else "lora")
    return PEFT_TYPES.get(t, t.lower())


PEFT_BUILD = r'''
# the PEFT method: its configuration from train_config.json, then any
# --peft-arg; arguments this version of PEFT does not know are left out
import inspect
import peft
METHOD = CFG.get("peft", "lora")
ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn",
             "query_key_value", "Wqkv", "out_proj"]
MLP = ["gate_proj", "up_proj", "down_proj", "c_fc", "fc1", "fc2", "w1", "w2",
       "w3", "dense_h_to_4h", "dense_4h_to_h"]
target = CFG.get("target") or ("default" if METHOD in (
    "ia3", "vera", "ln", "prompt", "ptuning") else "all-linear")
target = {"all-linear": "all-linear", "attention": ATTENTION, "mlp": MLP,
          "default": None}.get(target, [t.strip() for t in target.split(",")
                                        if t.strip()])
r, alpha, drop = CFG["lora_r"], CFG["lora_alpha"], CFG["lora_dropout"]
lora = dict(r=r, lora_alpha=alpha, lora_dropout=drop, target_modules=target)
hidden = getattr(model.config, "hidden_size", None) or getattr(
    model.config, "n_embd", None)
SPEC = {
    "lora": ("LoraConfig", lora),
    "dora": ("LoraConfig", dict(lora, use_dora=True)),
    "rslora": ("LoraConfig", dict(lora, use_rslora=True)),
    "pissa": ("LoraConfig", dict(lora, init_lora_weights="pissa",
                                 lora_dropout=0.0)),
    "olora": ("LoraConfig", dict(lora, init_lora_weights="olora",
                                 lora_dropout=0.0)),
    "loha": ("LoHaConfig", dict(r=r, alpha=alpha, module_dropout=drop,
                                target_modules=target)),
    "lokr": ("LoKrConfig", dict(r=r, alpha=alpha, module_dropout=drop,
                                target_modules=target)),
    "ia3": ("IA3Config", dict(target_modules=target,
                              feedforward_modules=None if target is None
                              else [])),
    "vera": ("VeraConfig", dict(r=r, vera_dropout=drop,
                                target_modules=target)),
    "oft": ("OFTConfig", dict(r=0, oft_block_size=r, module_dropout=drop,
                              target_modules=target)),
    "boft": ("BOFTConfig", dict(boft_block_size=r, boft_block_num=0,
                                boft_dropout=drop, target_modules=target)),
    "fourierft": ("FourierFTConfig", dict(target_modules=target)),
    "hra": ("HRAConfig", dict(r=r, target_modules=target)),
    "randlora": ("RandLoraConfig", dict(r=r, randlora_dropout=drop,
                                        target_modules=target)),
    "ln": ("LNTuningConfig", dict(target_modules=target)),
    "prompt": ("PromptTuningConfig", dict(
        num_virtual_tokens=CFG.get("virtual_tokens", 20))),
    "ptuning": ("PromptEncoderConfig", dict(
        num_virtual_tokens=CFG.get("virtual_tokens", 20),
        encoder_hidden_size=hidden)),
}
cls_name, kw = SPEC[METHOD]
kw = {k: v for k, v in kw.items()
      if not (k in ("target_modules", "feedforward_modules") and v is None)}
kw.update(CFG.get("peft_args") or {})
cls = getattr(peft, cls_name)
known = set(inspect.signature(cls).parameters)
left = sorted(k for k in kw if k not in known)
if left:
    print("not known to this PEFT, left out:", ", ".join(left))
PEFT_CONFIG = cls(task_type="CAUSAL_LM",
                  **{k: v for k, v in kw.items() if k in known})
print("PEFT method:", METHOD, "(%s)" % cls_name)
'''

TRAIN_COMMON = r'''
import dataclasses, json, sys
CFG = json.load(open(sys.argv[1], encoding="utf-8"))
from datasets import load_dataset
import trl
train = load_dataset("json", data_files=CFG["dataset"], split="train")
evald = load_dataset("json", data_files=CFG["eval"], split="train") \
    if CFG.get("eval") else None


def config(cls, **kw):
    """Only the arguments this version of TRL knows (names have changed)."""
    names = {f.name for f in dataclasses.fields(cls)}
    if "max_length" not in names and "max_seq_length" in names:
        kw["max_seq_length"] = kw.pop("max_length")
    dropped = sorted(k for k in kw if k not in names)
    if dropped:
        print("not known to this TRL, left out:", ", ".join(dropped))
    return cls(**{k: v for k, v in kw.items() if k in names})


ARGS = dict(output_dir=CFG["output_dir"], num_train_epochs=CFG["epochs"],
            learning_rate=CFG["learning_rate"],
            per_device_train_batch_size=CFG["batch_size"],
            gradient_accumulation_steps=CFG["grad_accum"],
            max_length=CFG["max_seq_len"], logging_steps=10,
            save_strategy="epoch", report_to=[], seed=CFG["seed"],
            bf16=BF16, fp16=FP16)
if CFG["method"] == "dpo":
    Trainer = trl.DPOTrainer
    targs = config(trl.DPOConfig, beta=CFG["beta"], **ARGS)
else:
    Trainer = trl.SFTTrainer
    targs = config(trl.SFTConfig, **ARGS)
kw = dict(model=model, args=targs, train_dataset=train, eval_dataset=evald)
import inspect
params = inspect.signature(Trainer.__init__).parameters
kw["processing_class" if "processing_class" in params else "tokenizer"] = tok
if PEFT_CONFIG is not None:
    kw["peft_config"] = PEFT_CONFIG
trainer = Trainer(**kw)
# PiSSA and OLoRA change the base weights when they start: the adapter is
# saved converted to a plain LoRA (twice the rank) that applies to the
# original base model, so it can be merged, served and combined like one
import os, shutil
init = getattr(PEFT_CONFIG, "init_lora_weights", True)
convert = isinstance(init, str) and init.split("_")[0] in ("pissa", "olora")
if convert:
    first = os.path.join(CFG["output_dir"], "initial-adapter")
    trainer.model.peft_config["default"].init_lora_weights = True
    trainer.model.save_pretrained(first)
    trainer.model.peft_config["default"].init_lora_weights = init
trainer.train()
if convert:
    trainer.model.save_pretrained(
        CFG["output_dir"], path_initial_model_for_weight_conversion=first)
    shutil.rmtree(first, ignore_errors=True)
    print("converted to a plain LoRA adapter for the original base model")
else:
    trainer.save_model(CFG["output_dir"])
tok.save_pretrained(CFG["output_dir"])
print("adapter saved in", CFG["output_dir"])
'''

TRAIN_TORCH = r'''# written by neural_bridge: a PEFT adapter (LoRA, DoRA, LoHa, LoKr,
# IA3, VeRA, OFT, prompt tuning and others; 4-bit with QLoRA) with PyTorch,
# transformers, PEFT and TRL. Run: python train.py train_config.json
import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
CFG = json.load(open(sys.argv[1], encoding="utf-8"))
GPU = torch.cuda.is_available()
BF16 = GPU and torch.cuda.is_bf16_supported()
FP16 = GPU and not BF16
quant = None
if CFG["qlora"]:
    from transformers import BitsAndBytesConfig
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16)
tok = AutoTokenizer.from_pretrained(CFG["base_model"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    CFG["base_model"], quantization_config=quant,
    torch_dtype=torch.bfloat16 if BF16 else torch.float16 if GPU
    else torch.float32, device_map="auto" if GPU else None)
if CFG["qlora"]:
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
''' + PEFT_BUILD + TRAIN_COMMON

TRAIN_UNSLOTH = r'''# written by neural_bridge: LoRA / QLoRA with Unsloth and TRL.
# Run: python train.py train_config.json
import json, sys
from unsloth import FastLanguageModel   # before trl and transformers
import torch
CFG = json.load(open(sys.argv[1], encoding="utf-8"))
BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
FP16 = torch.cuda.is_available() and not BF16
model, tok = FastLanguageModel.from_pretrained(
    model_name=CFG["base_model"], max_seq_length=CFG["max_seq_len"],
    load_in_4bit=CFG["qlora"], dtype=None)
model = FastLanguageModel.get_peft_model(
    model, r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"], lora_dropout=0,
    target_modules=%r, use_gradient_checkpointing="unsloth",
    use_rslora=CFG.get("peft") == "rslora",
    random_state=CFG["seed"])
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
PEFT_CONFIG = None
''' % UNSLOTH_TARGETS + TRAIN_COMMON


def _yaml(value, indent=0):
    """A small YAML writer: JSON scalars are valid YAML."""
    pad = " " * indent
    if isinstance(value, dict):
        out = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                out.append("%s%s:\n%s" % (pad, k, _yaml(v, indent + 2)))
            else:
                out.append("%s%s: %s" % (pad, k, json.dumps(v)))
        return "\n".join(out)
    if isinstance(value, list):
        out = []
        for v in value:
            if isinstance(v, dict):
                inner = _yaml(v, indent + 2).lstrip()
                out.append("%s- %s" % (pad, inner))
            else:
                out.append("%s- %s" % (pad, json.dumps(v)))
        return "\n".join(out)
    return pad + json.dumps(value)


def axolotl_config(cfg):
    if cfg["method"] != "sft":
        fail("the axolotl engine here writes SFT configs only; use --engine "
             "torch or unsloth for DPO")
    if cfg["format"] != "chat":
        fail("the axolotl engine here needs chat data (compile --format "
             "chat)")
    y = {"base_model": cfg["base_model"], "load_in_4bit": cfg["qlora"],
         "adapter": "qlora" if cfg["qlora"] else "lora",
         "lora_r": cfg["lora_r"], "lora_alpha": cfg["lora_alpha"],
         "lora_dropout": cfg["lora_dropout"], "lora_target_linear": True,
         "peft_use_dora": cfg.get("peft") == "dora",
         "peft_use_rslora": cfg.get("peft") == "rslora",
         "sequence_len": cfg["max_seq_len"],
         "datasets": [{"path": cfg["dataset"], "ds_type": "json",
                       "type": "chat_template",
                       "field_messages": "messages"}],
         "output_dir": cfg["output_dir"], "num_epochs": cfg["epochs"],
         "learning_rate": cfg["learning_rate"],
         "micro_batch_size": cfg["batch_size"],
         "gradient_accumulation_steps": cfg["grad_accum"],
         "seed": cfg["seed"], "bf16": "auto"}
    if cfg.get("eval"):
        y["test_datasets"] = [{"path": cfg["eval"], "ds_type": "json",
                               "type": "chat_template",
                               "field_messages": "messages",
                               "split": "train"}]
    return _yaml(y) + "\n"


def data_format(path, method):
    rows = read_jsonl_head(path)
    if not rows:
        fail("%s holds no examples" % path)
    keys = set(rows[0])
    if method == "dpo":
        if not {"prompt", "chosen", "rejected"} <= keys:
            fail("DPO needs rows with prompt, chosen and rejected (compile "
                 "writes dpo_train.jsonl); %s has %s" % (path,
                                                         sorted(keys)))
        return "dpo"
    if "messages" in keys:
        return "chat"
    if {"prompt", "completion"} <= keys:
        return "prompt"
    fail("SFT needs rows with messages, or prompt and completion; %s has %s"
         % (path, sorted(keys)))


def prepare_train(a):
    """Check the options and write models/adapters/NAME/ (the script or
    axolotl config, train_config.json, the dataset's manifest). Returns
    (cfg, dataset, folder, command)."""
    ds = Path(a.dataset)
    if not ds.is_file():
        fail("no dataset %s (make one with: compile)" % ds)
    fmt = data_format(ds, a.method)
    if a.eval:
        if not Path(a.eval).is_file():
            fail("no evaluation data %s" % a.eval)
        if data_format(Path(a.eval), a.method) != fmt:
            fail("the evaluation data is not in the training data's form")
    if a.engine == "unsloth" and not a.qlora:
        print("note: Unsloth is usually used with --qlora")
    engine_peft = {"unsloth": ("lora", "rslora"),
                   "axolotl": ("lora", "dora", "rslora")}.get(a.engine)
    if engine_peft and a.peft not in engine_peft:
        fail("--engine %s trains %s; --peft %s needs --engine torch" % (
            a.engine, ", ".join(engine_peft), a.peft))
    peft_args = {}
    for item in a.peft_arg or []:
        key, eq, value = item.partition("=")
        if not eq or not key.strip().isidentifier():
            fail("--peft-arg takes KEY=VALUE, not %s" % item)
        try:
            peft_args[key.strip()] = json.loads(value)
        except ValueError:
            peft_args[key.strip()] = value
    if a.virtual_tokens and a.peft not in NO_MERGE:
        fail("--virtual-tokens is for --peft prompt or ptuning")
    if a.target and a.peft in NO_MERGE:
        fail("--peft %s adds virtual tokens; it has no --target" % a.peft)
    if a.peft in ("pissa", "olora") and a.method == "dpo":
        fail("DPO takes its reference model by switching the adapter off; "
             "%s changes the base weights, so that reference would be "
             "wrong. Use lora, dora or rslora for DPO" % a.peft)
    if a.peft in ("pissa", "olora") and a.qlora:
        fail("%s starts from the full-precision weights; it is not used "
             "with --qlora here" % a.peft)
    if a.qlora and a.peft not in PLAIN_LORA + ("dora",):
        print("note: 4-bit training with --peft %s depends on PEFT's "
              "support for it; it was not tried" % a.peft)
    if a.peft == "hra" and a.rank % 2:
        fail("HRA needs an even --rank")
    adir = models_dir(a) / "adapters" / a.name
    if (adir / "adapter_config.json").exists() and not a.overwrite:
        fail("adapter %s exists (choose another name or --overwrite)"
             % a.name)
    adir.mkdir(parents=True, exist_ok=True)
    cfg = {"adapter": a.name, "base_model": a.base_model, "method": a.method,
           "engine": a.engine, "format": fmt, "qlora": a.qlora,
           "lora_r": a.rank, "lora_alpha": a.alpha or 2 * a.rank,
           "lora_dropout": a.dropout, "learning_rate": a.lr,
           "epochs": a.epochs, "batch_size": a.batch_size,
           "grad_accum": a.grad_accum, "max_seq_len": a.max_seq_len,
           "beta": a.beta, "seed": a.seed, "peft": a.peft,
           "target": a.target or "", "peft_args": peft_args,
           "virtual_tokens": a.virtual_tokens or 20,
           "dataset": str(ds.resolve()),
           "eval": str(Path(a.eval).resolve()) if a.eval else "",
           "dataset_sha256": hashlib.sha256(ds.read_bytes()).hexdigest(),
           "output_dir": str(adir.resolve()),
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    man = ds.with_name(ds.name + ".manifest.json")
    if man.is_file():
        shutil.copy(man, adir / "dataset.manifest.json")
    else:
        print("note: %s has no manifest; the adapter cannot be traced back "
              "to its files" % ds)
    write_json(adir / "train_config.json", cfg)
    if a.engine == "axolotl":
        (adir / "axolotl.yml").write_text(axolotl_config(cfg),
                                          encoding="utf-8")
        cmd = ([shutil.which("axolotl"), "train", adir / "axolotl.yml"]
               if shutil.which("axolotl") else
               [a.python, "-m", "axolotl.cli.train", adir / "axolotl.yml"])
    else:
        (adir / "train.py").write_text(
            TRAIN_TORCH if a.engine == "torch" else TRAIN_UNSLOTH,
            encoding="utf-8")
        cmd = [a.python, adir / "train.py", adir / "train_config.json"]
    size = ("%d virtual tokens" % cfg["virtual_tokens"]
            if a.peft in NO_MERGE else "r=%d alpha=%d" % (
                cfg["lora_r"], cfg["lora_alpha"]))
    print("adapter %s: %s, %s %s, %s data, %s%s" % (
        a.name, a.engine, a.method.upper(), PEFT_METHODS[a.peft].split(":")[0],
        fmt, size, ", 4-bit (QLoRA)" if a.qlora else ""))
    return cfg, ds, adir, cmd


def train_job_parts(a, cfg, ds, adir, out="out"):
    """The files and texts of a job folder that trains with relative paths,
    to run here or elsewhere: (program or None, files, texts)."""
    job_cfg = dict(cfg, dataset="data/" + ds.name, output_dir=out,
                   base_model=job_base(a, a.base_model),
                   eval="data/eval-" + Path(a.eval).name if a.eval else "")
    files = {"data/" + ds.name: ds}
    if a.eval:
        files["data/eval-" + Path(a.eval).name] = Path(a.eval)
    texts = {"train_config.json": json.dumps(job_cfg, indent=2)}
    if a.engine == "axolotl":
        texts["axolotl.yml"] = axolotl_config(job_cfg)
        return ["axolotl", "train", "axolotl.yml"], files, texts
    texts["train.py"] = (adir / "train.py").read_text(encoding="utf-8")
    return None, files, texts


def train_record(a, cfg, adir):
    return {"kind": "train", "adapter": a.name, "path": rel(a, adir),
            "base": a.base_model, "rank": cfg["lora_r"], "peft": a.peft}


def cmd_train(a):
    """Write the training script (or axolotl config) and the configuration
    into models/adapters/NAME/, with the dataset's manifest for lineage;
    run it with --run."""
    cfg, ds, adir, cmd = prepare_train(a)
    if getattr(a, "runtime", None):
        prog, files, texts = train_job_parts(a, cfg, ds, adir)
        return job(a, "train " + a.name, prog, ["train.py",
                                                 "train_config.json"],
                   files, texts, adir, train_record(a, cfg, adir))
    code = run_or_plan(a, cmd, "train")
    reg = load_registry(a)
    entry = reg["adapters"].get(a.name, {})
    entry.update(path=rel(a, adir),
                 base=a.base_model, trained=code == 0 and a.run,
                 rank=cfg["lora_r"], peft=a.peft)
    reg["adapters"][a.name] = entry
    save_registry(a, reg)
    return code


# ---------------------------------------------------------- merge, convert
MERGE_SCRIPT = r'''# written by neural_bridge: merge a PEFT adapter into its base model.
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base, adapter, out = sys.argv[1:4]
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.save_pretrained(out, safe_serialization=True)
try:
    tok = AutoTokenizer.from_pretrained(adapter)
except Exception:
    tok = AutoTokenizer.from_pretrained(base)
tok.save_pretrained(out)
print("merged model saved in", out)
'''

FUSE_SCRIPT = r'''# written by neural_bridge: a weighted combination of LoRA adapters.
import json, sys, torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
spec = json.load(open(sys.argv[1], encoding="utf-8"))
model = AutoModelForCausalLM.from_pretrained(spec["base"],
                                             torch_dtype=torch.float16)
first = spec["parts"][0]
model = PeftModel.from_pretrained(model, first["path"],
                                  adapter_name=first["name"])
for p in spec["parts"][1:]:
    model.load_adapter(p["path"], adapter_name=p["name"])
model.add_weighted_adapter(adapters=[p["name"] for p in spec["parts"]],
                           weights=[p["weight"] for p in spec["parts"]],
                           adapter_name=spec["name"],
                           combination_type=spec["how"],
                           **spec.get("extra", {}))
model.save_pretrained(spec["out"], selected_adapters=[spec["name"]])
print("fused adapter saved in", spec["out"])
'''


def adapter_path(a, name):
    reg = load_registry(a)
    if name not in reg["adapters"]:
        fail("no adapter %s (see: lora list)" % name)
    return abs_path(a, reg["adapters"][name]["path"]), reg["adapters"][name]


def cmd_merge(a):
    path, entry = adapter_path(a, a.name)
    if adapter_kind(path, entry) in NO_MERGE:
        fail("%s is %s: its virtual tokens go before each input at run time "
             "and change no weight, so there is nothing to merge; load it "
             "with PEFT beside its base" % (a.name, adapter_kind(path, entry)))
    base = a.base_model or entry.get("base")
    if not base:
        fail("give --base-model")
    out = Path(a.out) if a.out else models_dir(a) / "merged" / a.name
    sdir = models_dir(a) / "scripts"
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "merge.py").write_text(MERGE_SCRIPT, encoding="utf-8")
    if getattr(a, "runtime", None):
        files = {"adapter/" + f.name: f for f in Path(path).iterdir()
                 if f.is_file()}
        return job(a, "merge " + a.name, None, [
            "merge.py", job_base(a, base), "adapter", "out"], files,
                   {"merge.py": MERGE_SCRIPT}, out,
                   {"kind": "merge", "adapter": a.name})
    code = run_or_plan(a, [a.python, sdir / "merge.py", base, path, out],
                       "merge")
    if code == 0 and a.run:
        reg = load_registry(a)
        reg["adapters"][a.name]["merged"] = rel(a, out)
        save_registry(a, reg)
    return code


def cmd_convert(a):
    """A merged model folder to GGUF (convert_hf_to_gguf.py), or with
    --lora an adapter to a GGUF adapter (convert_lora_to_gguf.py)."""
    script = find_llama_tool(a, "convert_lora_to_gguf.py" if a.lora else
                             "convert_hf_to_gguf.py")
    if not script:
        fail("llama.cpp's conversion script was not found: give --llama-cpp "
             "FOLDER (a clone of llama.cpp)")
    if a.lora:
        path, entry = adapter_path(a, a.source)
        if adapter_kind(path, entry) not in PLAIN_LORA:
            fail("llama.cpp converts LoRA adapters only; %s is %s: merge "
                 "it, then convert the merged model" % (
                     a.source, adapter_kind(path, entry)))
        base = a.base_model or entry.get("base")
        out = Path(a.out or models_dir(a) / "gguf" / (a.source + ".gguf"))
        cmd = [a.python, script, path, "--outfile", out, "--outtype",
               a.outtype]
        if base:
            cmd += ["--base", base] if Path(base).exists() else \
                ["--base-model-id", base]
    else:
        src = Path(a.source)
        if not src.is_dir():
            path, entry = adapter_path(a, a.source)
            src = abs_path(a, entry["merged"]) if entry.get("merged") \
                else Path("")
            if not entry.get("merged") or not src.is_dir():
                fail("%s is neither a model folder nor a merged adapter "
                     "(run merge first)" % a.source)
        out = Path(a.out or models_dir(a) / "gguf" / (src.name + "-" +
                                                      a.outtype + ".gguf"))
        cmd = [a.python, script, src, "--outfile", out, "--outtype",
               a.outtype]
    out.parent.mkdir(parents=True, exist_ok=True)
    code = run_or_plan(a, cmd, "convert")
    if code == 0 and a.run and a.lora:
        reg = load_registry(a)
        reg["adapters"][a.source]["gguf"] = rel(a, out)
        save_registry(a, reg)
    return code


def cmd_quantize(a):
    src = Path(a.source)
    if not src.suffix.lower() == ".gguf":
        fail("quantize takes a GGUF file (convert a model first)")
    if a.run and not src.is_file():
        fail("no such file: %s" % src)
    tool = find_llama_tool(a, "llama-quantize")
    if not tool:
        fail("llama-quantize was not found: build llama.cpp and give "
             "--llama-cpp FOLDER, or put it on PATH")
    qtype = a.type.upper()
    if qtype not in QUANTS:
        fail("unknown type %s (%s)" % (a.type, ", ".join(QUANTS)))
    out = Path(a.out) if a.out else src.with_name(
        src.stem.replace("-F16", "").replace("-f16", "") + "-" + qtype +
        ".gguf")
    return run_or_plan(a, [tool, src, out, qtype], "quantize")


# --------------------------------------------------------------------- lora
def registry_file(a):
    return models_dir(a) / "registry.json"


def load_registry(a):
    f = registry_file(a)
    if f.is_file():
        try:
            reg = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            fail("%s is not readable JSON" % f)
    else:
        reg = {}
    reg.setdefault("adapters", {})
    reg.setdefault("bindings", {})
    return reg


def save_registry(a, reg):
    write_json(registry_file(a), reg)


def abs_path(a, p):
    p = Path(p)
    return p if p.is_absolute() else (Path(a.project) / p).resolve()


def rel(a, p):
    """A path as kept in the registry: relative to the project."""
    return os.path.relpath(Path(p).resolve(), Path(a.project).resolve())


def served_name(name, entry):
    return entry.get("served_as") or name


def route(st, reg, nid):
    """The adapter bound to nid or to its nearest ancestor: (name, from)."""
    for n in reversed(st.ancestors(nid)):
        if n.id in reg["bindings"]:
            return reg["bindings"][n.id], n.id
    return None, None


def cmd_lora(a):
    reg = load_registry(a)
    act = a.action
    if act == "add":
        name, path = a.args[0], Path(a.args[1])
        if not (path / "adapter_config.json").is_file() and not a.force:
            fail("%s has no adapter_config.json (not a PEFT adapter; "
                 "--force to add it anyway)" % path)
        reg["adapters"][name] = {"path": rel(a, path), "base": a.base_model or
                                 "", "served_as": a.served_as or "",
                                 "added": time.strftime("%Y-%m-%d")}
        save_registry(a, reg)
        print("adapter %s: %s" % (name, path))
    elif act == "rm":
        name = a.args[0]
        if name not in reg["adapters"]:
            fail("no adapter %s" % name)
        bound = [i for i, n in reg["bindings"].items() if n == name]
        if bound and not a.force:
            fail("%s is bound to %s; unbind first (or --force)" % (
                name, ", ".join(bound)))
        for i in bound:
            del reg["bindings"][i]
        del reg["adapters"][name]
        save_registry(a, reg)
        print("removed %s from the registry (its files stay)" % name)
    elif act == "list":
        st = tawjih.Store(a.project)
        for name, e in sorted(reg["adapters"].items()):
            files = [i for i, n in reg["bindings"].items() if n == name]
            print("%-16s %-9s base %-28s %s%s" % (
                name, adapter_kind(abs_path(a, e["path"]), e),
                e.get("base") or "?", e.get("path"),
                "  served as %s" % e["served_as"] if e.get("served_as")
                else ""))
            for i in files:
                print("    bound to %s [%s] %s" % (
                    st.number(i) if i in st.nodes else "?", i,
                    st.nodes[i].title if i in st.nodes else "(missing)"))
        if not reg["adapters"]:
            print("no adapters (train one, or: lora add NAME FOLDER)")
    elif act == "bind":
        st = tawjih.Store(a.project)
        ref, name = a.args
        if name not in reg["adapters"]:
            fail("no adapter %s (see: lora list)" % name)
        nid = st.resolve(ref)
        reg["bindings"][nid] = name
        save_registry(a, reg)
        print("%s [%s] %s and the files under it -> %s" % (
            st.number(nid), nid, st.nodes[nid].title, name))
    elif act == "unbind":
        st = tawjih.Store(a.project)
        nid = st.resolve(a.args[0])
        if reg["bindings"].pop(nid, None) is None:
            fail("nothing is bound to %s" % a.args[0])
        save_registry(a, reg)
        print("unbound %s [%s]" % (st.number(nid), nid))
    elif act == "route":
        st = tawjih.Store(a.project)
        nid = st.resolve(a.args[0])
        name, src = route(st, reg, nid)
        n = st.nodes[nid]
        if name:
            print("%s [%s] %s -> %s (bound at %s [%s]); model: %s" % (
                st.number(nid), nid, n.title, name, st.number(src), src,
                served_name(name, reg["adapters"].get(name, {}))))
        else:
            print("%s [%s] %s -> no adapter (the base model, or the file's "
                  "own model: %s)" % (st.number(nid), nid, n.title,
                                      n.model or "-"))
    elif act == "apply":
        return lora_apply(a, reg)
    elif act == "fuse":
        return lora_fuse(a, reg)
    elif act == "serve":
        return lora_serve(a, reg)
    elif act == "lineage":
        return lora_lineage(a, reg)
    return 0


def lora_apply(a, reg):
    """Set the model: line of every file with instructions to the name its
    adapter is served under. A file's own model is kept unless it is an
    adapter's name (or --overwrite)."""
    st = tawjih.Store(a.project)
    names = {served_name(k, v) for k, v in reg["adapters"].items()}
    changes, kept = [], []
    for nid in st.subtree_ids(""):
        n = st.nodes[nid]
        if not n.instructions.strip() or n.mount:
            continue
        name, _ = route(st, reg, nid)
        want = served_name(name, reg["adapters"].get(name, {})) if name \
            else ""
        if n.model == want:
            continue
        if n.model and n.model not in names and not a.overwrite:
            if want:
                kept.append(n)
            continue
        changes.append((n, want))
    for n, want in changes:
        print("%s %-10s [%s] %s: %s -> %s" % (
            "set" if a.yes else "would set", st.number(n.id), n.id, n.title,
            n.model or "-", want or "-"))
    for n in kept:
        print("kept      %-10s [%s] %s: its own model %s" % (
            st.number(n.id), n.id, n.title, n.model))
    if not a.yes:
        if changes:
            print("run again with --yes to write these model: lines")
        else:
            print("every file already has its adapter")
        return 0
    failed = 0
    for n, want in changes:
        tawjih._backup(st, n)
        n.model = want
        try:
            st.save(n)
        except tawjih.Die as e:
            failed += 1
            print("not changed: %s" % e.msg)
    return 1 if failed else 0


def lora_fuse(a, reg):
    if len(a.args) < 3:
        fail("lora fuse NEW A:0.7 B:0.3 [--how linear]")
    new, parts = a.args[0], []
    for item in a.args[1:]:
        name, _, w = item.rpartition(":")
        if not name:
            fail("give adapter:weight, not %s" % item)
        try:
            weight = float(w)
        except ValueError:
            fail("the weight of %s is not a number" % name)
        path, entry = adapter_path(a, name)
        if adapter_kind(path, entry) not in PLAIN_LORA:
            fail("fuse combines LoRA adapters; %s is %s" % (
                name, adapter_kind(path, entry)))
        parts.append({"name": name, "path": str(path), "weight": weight,
                      "base": entry.get("base"),
                      "rank": adapter_rank(path, entry)})
    bases = {p["base"] for p in parts if p["base"]}
    if len(bases) > 1:
        fail("these adapters were trained on different bases (%s); they "
             "cannot be combined" % ", ".join(sorted(bases)))
    base = a.base_model or (bases.pop() if bases else None)
    if not base:
        fail("give --base-model")
    ranks = {p["rank"] for p in parts if p["rank"]}
    if a.how == "linear" and len(ranks) > 1:
        fail("linear needs adapters of the same rank (%s); use --how cat or "
             "svd" % ", ".join(map(str, sorted(ranks))))
    if new in reg["adapters"]:
        fail("adapter %s exists" % new)
    out = models_dir(a) / "adapters" / ("fused-" + new)
    spec = {"name": new, "base": base, "how": a.how, "out": str(out),
            "parts": [{k: p[k] for k in ("name", "path", "weight")}
                      for p in parts]}
    if a.how in ("ties", "dare_linear", "dare_ties", "magnitude_prune"):
        spec["extra"] = {"density": a.density}
    sdir = models_dir(a) / "scripts"
    write_json(sdir / ("fuse-%s.json" % new), spec)
    (sdir / "fuse.py").write_text(FUSE_SCRIPT, encoding="utf-8")
    code = run_or_plan(a, [a.python, sdir / "fuse.py",
                           sdir / ("fuse-%s.json" % new)], "fuse")
    if code == 0 and a.run:
        # PEFT saves an adapter that is not 'default' in a folder of its name
        reg["adapters"][new] = {
            "path": rel(a, out / new), "base": base,
            "fused_from": {p["name"]: p["weight"] for p in parts},
            "how": a.how, "added": time.strftime("%Y-%m-%d")}
        save_registry(a, reg)
    return code


def lora_serve(a, reg):
    """How to offer the adapters under their names: a vLLM command, or one
    Ollama model per adapter (Modelfiles written to models/ollama/)."""
    if not reg["adapters"] and not (getattr(a, "serverless", False)
                                    and a.base_model):
        fail("no adapters in the registry")
    by_base = {}
    for name, e in sorted(reg["adapters"].items()):
        by_base.setdefault(e.get("base") or "?", []).append((name, e))
    if a.server == "vllm":
        other = [n for n, e in sorted(reg["adapters"].items())
                 if adapter_kind(abs_path(a, e["path"]), e) not in
                 PLAIN_LORA]
        if other:
            print("left out: %s (vLLM serves LoRA adapters; merge these "
                  "and serve the merged model)" % ", ".join(other))
        by_base = {b: [(n, e) for n, e in items if n not in other]
                   for b, items in by_base.items()}
        by_base = {b: items for b, items in by_base.items() if items}
        if getattr(a, "serverless", False):
            if len(by_base) != 1 and not a.base_model:
                fail("the adapters have %d bases; give --base-model for the "
                     "one to serve" % len(by_base))
            base = a.base_model or next(iter(by_base))
            return serverless(a, base, by_base.get(base, []))
        for base, items in by_base.items():
            mods = ["%s=%s" % (served_name(n, e), abs_path(a, e["path"]))
                    for n, e in items]
            rank = max([adapter_rank(abs_path(a, e["path"]), e) or 16
                        for _, e in items])
            print(show_cmd(["vllm", "serve", base, "--enable-lora",
                            "--max-lora-rank", str(vllm_rank(rank)),
                            "--max-loras", str(len(items)),
                            "--lora-modules"] + mods))
        print("then: TAWJIH_BASE_URL=http://localhost:8000/v1 and "
              "--backend openai; each file's model: line picks its adapter")
        return 0
    if a.server == "ollama":
        odir = models_dir(a) / "ollama"
        odir.mkdir(parents=True, exist_ok=True)
        missing = []
        for name, e in sorted(reg["adapters"].items()):
            gguf = e.get("gguf")
            if not gguf:
                missing.append(name)
                continue
            mf = odir / (name + ".Modelfile")
            mf.write_text("FROM %s\nADAPTER %s\n" % (
                a.ollama_base or e.get("base"), abs_path(a, gguf)),
                encoding="utf-8")
            print(show_cmd(["ollama", "create", served_name(name, e), "-f",
                            mf]))
        if missing:
            print("no GGUF adapter yet for: %s (convert NAME --lora)"
                  % ", ".join(missing))
        print("then --backend ollama; each file's model: line picks its "
              "adapter. FROM must name the Ollama model of the same base "
              "(--ollama-base)")
        return 0
    print("llama-server applies its adapters to every request and does not "
          "choose them by model name, so it cannot route by file. One "
          "server per adapter:")
    for k, (name, e) in enumerate(sorted(reg["adapters"].items())):
        print(show_cmd(["llama-server", "-m", "BASE.gguf", "--lora",
                        e.get("gguf") or "%s.gguf" % name, "--port",
                        str(8081 + k)]))
    return 0


def lora_lineage(a, reg):
    """What an adapter was trained from: its dataset's manifest, and
    whether each source file has changed since."""
    if not a.args:
        fail("lora lineage NAME")
    path, entry = adapter_path(a, a.args[0])
    cfgf, man = path / "train_config.json", path / "dataset.manifest.json"
    if entry.get("fused_from"):
        print("%s is a fusion (%s) of: %s" % (a.args[0], entry.get("how"),
                                             ", ".join("%s %.2f" % kv for kv
                                                       in entry["fused_from"]
                                                       .items())))
        return 0
    if cfgf.is_file():
        cfg = json.loads(cfgf.read_text(encoding="utf-8"))
        print("%s: %s %s on %s, from %s (%s)" % (
            a.args[0], cfg["engine"], cfg["method"].upper(),
            cfg["base_model"], cfg["dataset"], cfg["created"]))
        ds = Path(cfg["dataset"])
        if ds.is_file() and hashlib.sha256(ds.read_bytes()).hexdigest() != \
                cfg["dataset_sha256"]:
            print("  the dataset file has changed since training")
    if not man.is_file():
        print("  no dataset manifest: the source files are unknown")
        return 1
    m = json.loads(man.read_text(encoding="utf-8"))
    st = tawjih.Store(a.project)
    print("  %d example(s) from %d file(s):" % (m.get("examples", 0),
                                                len(m.get("from", {}))))
    for nid, sha in m.get("from", {}).items():
        tawjih._trace_file(st, nid, sha)
    return 0


# ---------------------------------------------------------------------- cli
# -------------------------------------------------------------------- setup
# What each kind of work needs in the training Python. Tawjih installs none
# of it into itself: it brings these libraries to the Python that works
# (here or on a runtime) and passes the work from one to the next.
SETUP_SETS = {
    "train": ["torch", "transformers", "peft", "trl", "datasets",
              "accelerate"],
    "qlora": ["bitsandbytes"],
    "push": ["huggingface_hub"],
    "serve": ["vllm"],
    "stream": ["vllm[runai]"],
    "unsloth": ["unsloth"],
    "axolotl": ["axolotl"],
}


def cmd_setup(a):
    """pip install what the chosen work needs into the training Python
    (--python), or on a runtime as a job; then what it has."""
    wanted = []
    for part in a.for_ or ["train", "push"]:
        for w in part.split(","):
            w = w.strip()
            if w not in SETUP_SETS:
                fail("setup --for takes %s, not %s" % (
                    ", ".join(SETUP_SETS), w))
            wanted.append(w)
    pkgs = []
    for w in wanted:
        for p in SETUP_SETS[w]:
            if p not in pkgs:
                pkgs.append(p)
    args = ["-m", "pip", "install", "--upgrade"] + list(a.pip_arg or []) \
        + pkgs
    if {"serve", "stream"} & set(wanted):
        print("note: vLLM runs on Linux with an NVIDIA GPU (on Windows, in "
              "WSL); install it on the machine that serves")
    if getattr(a, "runtime", None):
        return job(a, "setup " + ",".join(wanted), None, args, {}, {}, None,
                   {"kind": "setup"})
    code = run_or_plan(a, [a.python] + args, "setup")
    if code == 0 and a.run:
        cmd_env(a)
    return code


# --------------------------------------------------------------------- push
PUSH_SCRIPT = r'''# written by neural_bridge: upload a folder to the Hugging Face Hub.
# The token is HF_TOKEN, or the one saved by: hf auth login
import json, sys
from huggingface_hub import HfApi, get_token
spec = json.load(open(sys.argv[1], encoding="utf-8"))
if not get_token():
    sys.exit("no Hugging Face token: set HF_TOKEN, or run: hf auth login")
api = HfApi(endpoint=spec.get("endpoint") or None)
url = api.create_repo(spec["repo"], private=spec["private"],
                      repo_type=spec["type"], exist_ok=True)
info = api.upload_folder(repo_id=spec["repo"], folder_path=spec["folder"],
                         repo_type=spec["type"],
                         commit_message=spec["message"],
                         allow_patterns=spec.get("allow") or None,
                         ignore_patterns=spec.get("ignore") or None)
if spec.get("card"):
    api.upload_file(path_or_fileobj=spec["card"].encode("utf-8"),
                    path_in_repo="README.md", repo_id=spec["repo"],
                    repo_type=spec["type"],
                    commit_message="Model card written by Tawjih")
print("uploaded to", url)
print("commit:", getattr(info, "commit_url", info))
'''

# kept out of an adapter's upload: training leftovers and local paths
ADAPTER_IGNORE = ["checkpoint-*", "checkpoint-*/**", "initial-adapter/**",
                  "train.py", "train_config.json", "axolotl.yml",
                  "dataset.manifest.json", "training_args.bin", "README.md"]


def _hub_id(name):
    """A Hugging Face model id (owner/name), not a local folder."""
    return bool(name) and "/" in name and not Path(name).exists() and \
        not name.startswith((".", "/", "~")) and ":" not in name


def model_card(name, cfg, what):
    """A model card from what Tawjih knows: no file names or titles of the
    project, only counts and the dataset's SHA-256."""
    base = cfg.get("base_model", "")
    if not _hub_id(base):            # a folder here: its path stays here
        base = ""
    peft = cfg.get("peft", "lora")
    rows = None
    try:
        with open(cfg["dataset"], encoding="utf-8") as f:
            rows = sum(1 for line in f if line.strip())
    except (OSError, KeyError):
        pass
    meta = ["---"]
    if base:
        meta.append("base_model: %s" % base)
    meta.append("library_name: %s" % ("peft" if what == "adapter"
                                      else "transformers"))
    meta.append("tags:\n- tawjih\n- %s\n- %s" % (peft, cfg.get(
        "method", "sft")))
    meta.append("---")
    method = PEFT_METHODS.get(peft, peft)
    lines = ["", "# %s" % name, "",
             "%s for %s, trained with "
             "[Tawjih](https://github.com/AlmodinyMk/LLM-tawjih-Model-building)"
             " (%s, %s)." % (
                 "A PEFT adapter (%s)" % method.split(":")[0]
                 if what == "adapter" else "The adapter (%s) merged into the "
                 "weights of its base model" % method.split(":")[0],
                 "`%s`" % base if base else "a local model",
                 cfg.get("method", "sft").upper(),
                 cfg.get("engine", "torch")),
             "", "| | |", "|---|---|",
             "| PEFT method | %s |" % method,
             "| rank / alpha | %s / %s |" % (cfg.get("lora_r"),
                                            cfg.get("lora_alpha")),
             "| epochs | %s |" % cfg.get("epochs"),
             "| learning rate | %s |" % cfg.get("learning_rate"),
             "| longest sequence | %s |" % cfg.get("max_seq_len"),
             "| 4-bit (QLoRA) | %s |" % ("yes" if cfg.get("qlora") else "no"),
             "| training rows | %s |" % (rows if rows is not None else "?"),
             "| dataset SHA-256 | `%s` |" % cfg.get("dataset_sha256", "?"),
             "", "## Use", "", "```python"]
    if what == "adapter":
        lines += ["from transformers import AutoModelForCausalLM, "
                  "AutoTokenizer",
                  "from peft import PeftModel",
                  'base = AutoModelForCausalLM.from_pretrained("%s")' % (
                      base or "THE-BASE-MODEL"),
                  'model = PeftModel.from_pretrained(base, "REPO")',
                  'tok = AutoTokenizer.from_pretrained("REPO")']
    else:
        lines += ["from transformers import AutoModelForCausalLM, "
                  "AutoTokenizer",
                  'model = AutoModelForCausalLM.from_pretrained("REPO")',
                  'tok = AutoTokenizer.from_pretrained("REPO")']
    lines += ["```", "", "## Data", "",
              "The training rows were compiled by Tawjih from the examples "
              "in the files of a project; the files themselves are not part "
              "of this repository.", ""]
    return "\n".join(meta + lines)


def push_spec(a, folder, kind, card=None):
    spec = {"repo": a.repo, "private": not a.public, "folder": str(folder),
            "type": "dataset" if kind == "dataset" else "model",
            "message": a.message or "Upload %s from Tawjih" % kind,
            "endpoint": a.hub_endpoint or "", "card": card or ""}
    if kind == "adapter":
        spec["ignore"] = ADAPTER_IGNORE
    elif kind == "merged":
        spec["ignore"] = ["README.md"]
    else:
        spec["allow"] = ["*.jsonl"]
    return spec


def check_hub_allowed(a):
    pol = tawjih.load_policy(a.project)
    endpoint = a.hub_endpoint or "https://huggingface.co"
    if pol["allow_network"] is not True and not (
            pol["allow_network"] == "local" and tawjih._is_local(endpoint)):
        fail("the project's policy (%s) does not allow sending files to %s"
             % (tawjih.POLICY_FILE, endpoint))


def _repo_ok(repo):
    import re
    if not repo or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        fail("--repo takes OWNER/NAME on the Hugging Face Hub, not %r"
             % repo)


def _show_upload(folder, spec):
    """What would be sent: the files and their total size."""
    import fnmatch
    files, size = [], 0
    for f in sorted(Path(folder).rglob("*")):
        if not f.is_file():
            continue
        r = f.relative_to(folder).as_posix()
        if spec.get("allow") and not any(fnmatch.fnmatch(r, p)
                                         for p in spec["allow"]):
            continue
        if any(fnmatch.fnmatch(r, p) for p in spec.get("ignore") or []):
            continue
        files.append(r)
        size += f.stat().st_size
    print("  %d file(s), %.1f MB: %s%s" % (
        len(files), size / 1e6, ", ".join(files[:8]),
        " ..." if len(files) > 8 else ""))
    return files


def cmd_push(a):
    """Upload an adapter, its merged model or a dataset to the Hugging Face
    Hub: a private repository unless --public, with a model card made from
    the training record. Needs --run and --yes; the token is HF_TOKEN or
    the one `hf auth login` saved, never a file of the project."""
    _repo_ok(a.repo)
    if a.dataset_dir:
        folder, kind, card = Path(a.dataset_dir), "dataset", None
        if not folder.is_dir():
            fail("no folder %s" % folder)
    else:
        if not a.name:
            fail("push NAME --repo OWNER/NAME (or --dataset FOLDER)")
        path, entry = adapter_path(a, a.name)
        kind = a.what
        if kind == "merged":
            if not entry.get("merged"):
                fail("%s has no merged model yet (merge %s)" % (a.name,
                                                                a.name))
            folder = abs_path(a, entry["merged"])
        else:
            folder = path
        if not folder.is_dir():
            fail("no folder %s" % folder)
        cfgf = path / "train_config.json"
        cfg = json.loads(cfgf.read_text(encoding="utf-8")) \
            if cfgf.is_file() else {"base_model": entry.get("base", ""),
                                     "peft": adapter_kind(path, entry)}
        card = model_card(a.repo.split("/")[-1], cfg, kind).replace(
            '"REPO"', '"%s"' % a.repo)
    spec = push_spec(a, folder, kind, card)
    print("push %s %s to %s (%s%s)" % (
        kind, a.name or folder, a.repo, "private" if spec["private"]
        else "PUBLIC", ", " + a.hub_endpoint if a.hub_endpoint else ""))
    files = _show_upload(folder, spec)
    if not files:
        fail("nothing to upload in %s" % folder)
    sdir = models_dir(a) / "scripts"
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "push.py").write_text(PUSH_SCRIPT, encoding="utf-8")
    sfile = sdir / ("push-%s.json" % (a.name or "dataset"))
    write_json(sfile, spec)
    if a.run:
        check_hub_allowed(a)
        if not a.yes:
            fail("this sends the files to %s: check the list above, then "
                 "add --yes" % (a.hub_endpoint or "the Hugging Face Hub"))
    code = run_or_plan(a, [a.python, sdir / "push.py", sfile], "push")
    if code == 0 and a.run and a.name:
        reg = load_registry(a)
        hub = reg["adapters"][a.name].setdefault("hub", {})
        hub[kind] = a.repo
        save_registry(a, reg)
    return code


# ----------------------------------------------------------------- pipeline
def push_targets(a):
    """[(kind, repo)]: with --push both the merged model goes to its own
    repository, REPO-merged, so the two do not mix."""
    if a.push == "both":
        return [("adapter", a.repo), ("merged", a.repo + "-merged")]
    return [(a.push, a.repo)] if a.push else []


def cmd_pipeline(a):
    """compile, train, then (with --merge) merge and (with --push) push, in
    one go: here one after another, or on a runtime as one job that does
    them all there (the adapter comes back; a merged model stays there
    unless it is pushed or --fetch-merged)."""
    if a.push and not a.repo:
        fail("--push needs --repo OWNER/NAME")
    if a.push:
        _repo_ok(a.repo)
    merge = a.merge or a.push in ("merged", "both")
    if merge and a.peft in NO_MERGE:
        fail("--peft %s has nothing to merge" % a.peft)
    rt = getattr(a, "runtime", None)
    remote = bool(rt) and not runtimes.is_local(runtimes.get(rt)[1])
    if a.run and (a.push or remote) and not a.yes:
        fail("this sends files %s: check the plan without --run, then add "
             "--yes" % ("to the Hugging Face Hub" if a.push else
                        "to %s" % rt))
    if a.push and a.run:
        check_hub_allowed(a)
    print("== 1. compile")
    out = Path(a.data_dir)
    ca = argparse.Namespace(project=a.project, out=str(out),
                            branch=a.branch, with_rules=a.with_rules,
                            repeat_by_weight=False, format=a.format)
    if cmd_compile(ca):
        fail("the training and evaluation data share questions, so the "
             "evaluation would not be fair: fix the split first (tawjih.py "
             "check), then run the pipeline again")
    a.dataset = str(out / ("dpo_train.jsonl" if a.method == "dpo"
                           else "sft_train.jsonl"))
    a.eval = None
    print("== 2. train")
    if not remote:
        a.runtime = None
        code = cmd_train(a)
        if code:
            return code
        steps = []
        if merge:
            print("== 3. merge")
            ma = argparse.Namespace(**vars(a))
            ma.out = None
            if a.run:
                code = cmd_merge(ma)
                if code:
                    return code
            else:
                print("merge: after training, merge %s" % a.name)
        if a.push:
            print("== 4. push")
            for kind, repo in push_targets(a):
                pa = argparse.Namespace(**vars(a))
                pa.what, pa.dataset_dir, pa.repo = kind, None, repo
                if a.run:
                    code = cmd_push(pa)
                    if code:
                        return code
                else:
                    steps.append("%s to %s" % (kind, repo))
            if steps:
                print("push: after training, %s" % " and ".join(steps))
        return 0
    # one job on the runtime
    name, rcfg = runtimes.get(rt)
    py = rcfg.get("python") or "python3"
    cfg, ds, adir, _ = prepare_train(a)
    prog, files, texts = train_job_parts(a, cfg, ds, adir)
    if prog:
        fail("pipeline on a runtime uses --engine torch or unsloth")
    lines = ["set -e", "PY=%s" % shlex.quote(py),
             '"$PY" train.py train_config.json']
    merged_dir = "out/merged" if a.fetch_merged else "merged"
    if merge:
        texts["merge.py"] = MERGE_SCRIPT
        lines.append('"$PY" merge.py %s out %s' % (
            shlex.quote(job_base(a, a.base_model)), merged_dir))
    if a.push:
        texts["push.py"] = PUSH_SCRIPT
        for kind, repo in push_targets(a):
            card = model_card(repo.split("/")[-1], cfg, kind).replace(
                '"REPO"', '"%s"' % repo)
            spec = dict(push_spec(a, "out" if kind == "adapter"
                             else merged_dir, kind, card), repo=repo)
            if kind == "adapter":
                spec["ignore"] = ADAPTER_IGNORE + ["merged/**"]
            texts["push-%s.json" % kind] = json.dumps(spec, indent=2)
            lines.append('"$PY" push.py push-%s.json' % kind)
        print("note: the token there is its HF_TOKEN or the one `hf auth "
              "login` saved there; none is sent from here")
    texts["pipeline.sh"] = "\n".join(lines) + "\n"
    rec = dict(train_record(a, cfg, adir), kind="pipeline")
    if merge and a.fetch_merged:
        rec["merged"] = rel(a, adir / "merged")
    if a.push:
        rec["hub"] = dict(push_targets(a))
    print("on %s: %s" % (name, "; ".join(lines[2:])))
    if merge and not a.fetch_merged:
        print("note: the merged model stays on %s (add --fetch-merged to "
              "bring it back)" % name)
    return job(a, "pipeline " + a.name, ["sh", "pipeline.sh"], [], files,
               texts, adir, rec)


# ------------------------------------------------------- serving: vLLM
VLLM_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)   # vLLM 0.32's choices
VLLM_IMAGE = "vllm/vllm-openai:v0.32.0"


def vllm_rank(rank):
    """--max-lora-rank: the smallest value vLLM accepts that holds rank."""
    for r in VLLM_RANKS:
        if r >= rank:
            return r
    fail("vLLM takes LoRA ranks up to %d; this adapter has %d" % (
        VLLM_RANKS[-1], rank))


def serverless(a, base, items):
    """A folder to serve BASE and its adapters with vLLM where machines
    start cold (serverless GPUs): start.sh with the flags that shorten the
    start, a Dockerfile that puts the weights in the image, warmup.sh that
    fills the compile cache once, and README.md explaining each choice."""
    import re
    tag = re.sub(r"[^\w.-]+", "-", base.split("/")[-1]) or "model"
    out = Path(a.out_dir) if a.out_dir else models_dir(a) / "serve" / tag
    (out / "adapters").mkdir(parents=True, exist_ok=True)
    local_base = not _hub_id(base)
    model = "/models/base"
    seqs = max(1, a.max_num_seqs)
    sizes = [n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if n < seqs]
    sizes.append(seqs)
    served = tag if local_base else base     # the name model: lines use
    flags = ["--served-model-name", served, "--max-model-len",
             str(a.max_model_len), "--max-num-seqs", str(seqs)]
    why = []
    if a.eager:
        flags.append("--enforce-eager")
        why.append("--enforce-eager: no CUDA graphs and no compilation, the "
                   "fastest start and the slowest answers; for trying")
    else:
        flags += ["-cc", json.dumps({"cudagraph_capture_sizes": sizes})]
        why.append("-cc cudagraph_capture_sizes %s: CUDA graphs only for the "
                   "batch sizes up to --max-num-seqs %d, so fewer are "
                   "captured at each start" % (sizes, seqs))
    if a.weights == "volume":
        flags += ["--safetensors-load-strategy", "prefetch"]
        why.append("--safetensors-load-strategy prefetch: the weights on a "
                   "network volume are read ahead in large blocks instead "
                   "of mapped and read lazily")
    elif a.weights == "stream":
        flags += ["--load-format", "runai_streamer",
                  "--model-loader-extra-config",
                  json.dumps({"concurrency": 16})]
        why.append("--load-format runai_streamer (vllm[runai]): many "
                   "parallel reads of the safetensors files, also from S3 "
                   "or GCS; concurrency 16")
    if a.sleep:
        flags.append("--enable-sleep-mode")
        why.append("--enable-sleep-mode: POST /sleep frees the GPU and POST "
                   "/wake_up brings the model back in seconds (more for "
                   "large models) instead of a cold start; needs VLLM_SERVER_DEV_MODE=1, "
                   "so keep the server off the open internet")
    mods = []
    if items:
        rank = max([adapter_rank(abs_path(a, e["path"]), e) or 16
                    for _, e in items])
        flags += ["--enable-lora", "--max-lora-rank", str(vllm_rank(rank)),
                  "--max-loras", str(len(items)), "--lora-modules"]
        for n, e in items:
            src = abs_path(a, e["path"])
            dst = out / "adapters" / n
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                "checkpoint-*", "initial-adapter", "train.py",
                "train_config.json", "axolotl.yml", "dataset.manifest.json",
                "training_args.bin"))
            mods.append("%s=/models/adapters/%s" % (served_name(n, e), n))
        flags += mods
    env = {"VLLM_CACHE_ROOT": a.cache_dir, "HF_HUB_OFFLINE": "1"}
    if a.sleep:
        env["VLLM_SERVER_DEV_MODE"] = "1"
    start = ["#!/bin/sh", "# written by neural_bridge: vLLM for cold starts",
             "set -e"] + ["export %s=%s" % (k, shlex.quote(v))
                          for k, v in env.items()]
    start.append("exec vllm serve %s %s \"$@\"" % (model, " ".join(
        shlex.quote(f) for f in flags)))
    (out / "start.sh").write_text("\n".join(start) + "\n", encoding="utf-8")
    docker = ["# written by neural_bridge: the weights inside the image, so a",
              "# cold machine reads them from its own disk",
              "FROM %s" % VLLM_IMAGE]
    if a.weights == "stream":
        docker.append("RUN pip install --no-cache-dir 'vllm[runai]'")
    if a.weights == "image":
        if local_base:
            docker.append("COPY model/ /models/base/")
        else:
            docker += ["RUN --mount=type=secret,id=hf_token \\",
                       "    HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null"
                       ") hf download %s --local-dir /models/base" % base]
    else:
        docker.append("# the weights are mounted at /models/base")
    docker += ["COPY adapters/ /models/adapters/",
               "COPY start.sh /start.sh",
               "ENTRYPOINT [\"sh\", \"/start.sh\"]"]
    (out / "Dockerfile").write_text("\n".join(docker) + "\n",
                                    encoding="utf-8")
    first = served_name(*items[0]) if items else served
    warm = ["#!/bin/sh",
            "# written by neural_bridge: run once on a machine with the same",
            "# GPU and the cache volume at %s, so the compile cache is filled"
            % a.cache_dir,
            "# and later cold starts skip compiling",
            "set -e", "sh /start.sh &", "PID=$!",
            "python3 - <<'EOF'",
            "import json, time, urllib.request",
            "while True:",
            "    try:",
            "        urllib.request.urlopen('http://localhost:8000/health')",
            "        break",
            "    except OSError:",
            "        time.sleep(2)",
            "body = json.dumps({'model': %r, 'prompt': 'hi', "
            "'max_tokens': 4}).encode()" % first,
            "urllib.request.urlopen(urllib.request.Request(",
            "    'http://localhost:8000/v1/completions', body,",
            "    {'Content-Type': 'application/json'}))",
            "EOF",
            "kill $PID", "echo warm: the cache in %s is ready" % a.cache_dir]
    (out / "warmup.sh").write_text("\n".join(warm) + "\n", encoding="utf-8")
    readme = ["# Serving %s with vLLM, for cold starts" % base, "",
              "Written by Tawjih's training bridge for vLLM 0.32. Check the "
              "flags against the vLLM version you run.", "",
              "## Build and run", "", "```",
              "docker build -t %s-serve %s%s ." % (
                  tag.lower(), "--secret id=hf_token,env=HF_TOKEN "
                  if a.weights == "image" and not local_base else "",
                  "" if True else ""),
              "docker run --gpus all -p 8000:8000 -v cache:%s %s-serve" % (
                  a.cache_dir, tag.lower()), "```", ""]
    if a.weights == "image" and local_base:
        readme += ["Copy the model folder into `model/` here first: `cp -r "
                   "%s model`." % base, ""]
    readme += ["## What shortens the start", "",
               "- Weights: %s" % {
                   "image": "inside the image (`Dockerfile`), read from the "
                   "machine's own disk; nothing is downloaded at start "
                   "(`HF_HUB_OFFLINE=1`).",
                   "volume": "on a network volume mounted at `/models/base`, "
                   "read ahead with `--safetensors-load-strategy prefetch`.",
                   "stream": "streamed with the Run:ai model streamer; to "
                   "stream from storage, put its s3:// or gs:// path in "
                   "place of `/models/base` in `start.sh`."}[
                       a.weights],
               "- `VLLM_CACHE_ROOT=%s`: keep this on a volume that outlives "
               "the machine. vLLM keeps its torch.compile results there; "
               "after `warmup.sh` has run once on the same GPU type with the "
               "same flags, a cold machine loads them instead of compiling "
               "for minutes." % a.cache_dir] + ["- " + w for w in why] + [
               "- `--max-model-len %d`: no longer than the paths need; a "
               "smaller KV cache to set up." % a.max_model_len, "",
               "Also on the service's side: keep one worker warm, or a "
               "longer idle time before it stops, if a cold start is too "
               "slow for the users; that costs money while idle.", "",
               "## Tawjih", "",
               "Point a provider at the server (`provider add vllm --kind "
               "openai --base-url https://ENDPOINT/v1`); each file's "
               "`model:` line picks an adapter: %s." % (
                   ", ".join(m.split("=")[0] for m in mods) or base), ""]
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print("serverless vLLM for %s written to %s:" % (base, out))
    for f in ("start.sh", "Dockerfile", "warmup.sh", "README.md"):
        print("  %s" % (out / f))
    print(show_cmd(["vllm", "serve", model] + flags))
    for w in why:
        print("  - " + w)
    return 0


# ------------------------------------------------------------------ runtime
def job_base(a, base):
    """A job runs inside its own folder, so a model folder given by a
    relative path is made absolute. On another computer the folder must be
    there under the same path, or use a Hugging Face model id."""
    if not base or not Path(base).exists():
        return base
    _, cfg = runtimes.get(a.runtime)
    told = a.__dict__.setdefault("_told_base", set())
    if not runtimes.is_local(cfg) and base not in told:
        told.add(base)
        print("note: %s is a folder on this computer; it is not sent with "
              "the job, so it must exist on %s under the same path (or give "
              "a Hugging Face model id)" % (base, a.runtime))
    return str(Path(base).resolve())


def job(a, what, prog, args, files, texts, collect_to, extra):
    """Make a job folder and, with --run, start it on --runtime."""
    name, cfg = runtimes.get(a.runtime)
    py = cfg.get("python") or (a.python if cfg["kind"] == "local"
                               else "python3")
    command = prog or [py] + args
    jdir = runtimes.make_job(a.project, what, name, command, files, texts,
                             collect_to)
    runtimes.record(jdir, extra)
    print("job %s on %s (%s): %s" % (jdir.name, name, cfg["kind"],
                                     " ".join(command)))
    for line in runtimes.plan(jdir, name, cfg):
        print("  " + line)
    if not a.run:
        print("(plan only; add --run to start it%s)" % (
            "" if runtimes.is_local(cfg) else ", and --yes since it leaves "
            "this computer"))
        return 0
    runtimes.allowed(a.project, name, cfg, getattr(a, "yes", False))
    code = runtimes.submit(a.project, jdir, name, cfg)
    if cfg["kind"] == "local":
        if code == 0:
            runtimes.finish(a.project, jdir, cfg)
        return code
    print("started; follow it with: tawjih runtime status %s, tawjih "
          "runtime logs %s, then tawjih runtime collect %s" % (
              jdir.name, jdir.name, jdir.name))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="neural_bridge.py",
                                description=__doc__.split("\n")[0])
    p.add_argument("-p", "--project", default=".")
    p.add_argument("--python", default=sys.executable,
                   help="the Python that has the training libraries "
                   "(default: this one)")
    p.add_argument("--llama-cpp", metavar="FOLDER",
                   help="a clone of llama.cpp (scripts, build/bin)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def on_runtime(s):
        s.add_argument("--runtime", metavar="NAME", help="run it as a job "
                       "on this runtime (runtime list)")
        s.add_argument("--yes", action="store_true", help="allow a job to "
                       "leave this computer")

    def runnable(s):
        s.add_argument("--run", action="store_true",
                       help="start it (otherwise only the plan is shown)")

    s = sub.add_parser("compile", help="examples as SFT, eval and DPO JSONL")
    s.add_argument("-o", "--out", default="datasets")
    s.add_argument("--branch")
    s.add_argument("--format", choices=["chat", "prompt"], default="chat")
    s.add_argument("--with-rules", action="store_true",
                   help="each file's instructions before its questions")
    s.add_argument("--repeat-by-weight", action="store_true")
    s.set_defaults(fn=cmd_compile)

    s = sub.add_parser("env", help="engines and GPU of the training Python")
    s.set_defaults(fn=cmd_env)

    def training(s):
        """The training options, shared by train and pipeline."""
        s.add_argument("--base-model", required=True,
                       help="Hugging Face id or local folder")
        s.add_argument("--method", choices=["sft", "dpo"], default="sft")
        s.add_argument("--engine", choices=ENGINES, default="torch")
        s.add_argument("--qlora", action="store_true", help="4-bit base model")
        s.add_argument("--peft", choices=list(PEFT_METHODS), default="lora",
                       help="the PEFT method (default lora): " + "; ".join(
                           PEFT_METHODS.values()))
        s.add_argument("--target", metavar="WHERE", help="the layers: "
                       "all-linear, attention, mlp, default (PEFT's choice for "
                       "the model) or names separated by commas")
        s.add_argument("--virtual-tokens", type=int, metavar="N",
                       help="prompt and ptuning: how many (default 20)")
        s.add_argument("--peft-arg", action="append", metavar="KEY=VALUE",
                       help="any argument of the method's PEFT configuration, "
                       "e.g. init_lora_weights=gaussian (repeatable)")
        s.add_argument("--rank", type=int, default=16)
        s.add_argument("--alpha", type=int, help="default: 2 x rank")
        s.add_argument("--dropout", type=float, default=0.05)
        s.add_argument("--lr", type=float, default=2e-4)
        s.add_argument("--epochs", type=float, default=3)
        s.add_argument("--batch-size", type=int, default=2)
        s.add_argument("--grad-accum", type=int, default=4)
        s.add_argument("--max-seq-len", type=int, default=2048)
        s.add_argument("--beta", type=float, default=0.1, help="DPO beta")
        s.add_argument("--seed", type=int, default=42)
        s.add_argument("--overwrite", action="store_true")

    def hub(s):
        s.add_argument("--repo", metavar="OWNER/NAME",
                       help="the repository on the Hugging Face Hub")
        s.add_argument("--public", action="store_true",
                       help="a public repository (default private)")
        s.add_argument("--message", help="the commit message")
        s.add_argument("--hub-endpoint", metavar="URL",
                       help="another Hub (a mirror or your own); default "
                       "huggingface.co")

    s = sub.add_parser("train", help="train a PEFT adapter: LoRA, QLoRA, "
                       "DoRA, LoHa, LoKr, IA3, VeRA, OFT, prompt tuning ...")
    s.add_argument("name")
    s.add_argument("--dataset", default="datasets/sft_train.jsonl")
    s.add_argument("--eval", help="evaluation data (sft_eval.jsonl)")
    training(s)
    on_runtime(s)
    runnable(s)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("setup", help="install what the work needs into the "
                       "training Python, here or on a runtime")
    s.add_argument("--for", dest="for_", action="append",
                   metavar="WORK", help="%s (default train,push; "
                   "comma-separated or repeated)" % ", ".join(SETUP_SETS))
    s.add_argument("--pip-arg", action="append", metavar="ARG",
                   help="passed to pip, e.g. --pip-arg=--index-url=URL")
    on_runtime(s)
    runnable(s)
    s.set_defaults(fn=cmd_setup)

    s = sub.add_parser("push", help="upload an adapter, its merged model "
                       "or a dataset to the Hugging Face Hub")
    s.add_argument("name", nargs="?", help="the adapter (lora list)")
    s.add_argument("--what", choices=["adapter", "merged"],
                   default="adapter")
    s.add_argument("--dataset", dest="dataset_dir", metavar="FOLDER",
                   help="upload the .jsonl files of this folder as a "
                   "dataset instead")
    hub(s)
    s.add_argument("--yes", action="store_true",
                   help="allow the files to leave this computer")
    runnable(s)
    s.set_defaults(fn=cmd_push)

    s = sub.add_parser("pipeline", help="compile, train, merge and push in "
                       "one go, here or as one job on a runtime")
    s.add_argument("name")
    training(s)
    s.add_argument("--data-dir", default="datasets",
                   help="where compile writes (default datasets)")
    s.add_argument("--branch", help="compile only this branch")
    s.add_argument("--format", choices=["chat", "prompt"], default="chat")
    s.add_argument("--with-rules", action="store_true")
    s.add_argument("--merge", action="store_true",
                   help="merge the adapter into its base after training")
    s.add_argument("--push", choices=["adapter", "merged", "both"],
                   help="upload to --repo after training")
    s.add_argument("--fetch-merged", action="store_true",
                   help="on a runtime: bring the merged model back too")
    hub(s)
    on_runtime(s)
    runnable(s)
    s.set_defaults(fn=cmd_pipeline)

    s = sub.add_parser("merge", help="merge an adapter into its base")
    s.add_argument("name")
    s.add_argument("--base-model")
    s.add_argument("-o", "--out")
    on_runtime(s)
    runnable(s)
    s.set_defaults(fn=cmd_merge)

    s = sub.add_parser("convert", help="to GGUF with llama.cpp")
    s.add_argument("source", help="a model folder, a merged adapter's name, "
                   "or with --lora an adapter's name")
    s.add_argument("--lora", action="store_true")
    s.add_argument("--base-model")
    s.add_argument("--outtype", default="f16",
                   choices=["f32", "f16", "bf16", "q8_0"])
    s.add_argument("-o", "--out")
    runnable(s)
    s.set_defaults(fn=cmd_convert)

    s = sub.add_parser("quantize", help="quantize a GGUF file")
    s.add_argument("source")
    s.add_argument("--type", default="Q4_K_M")
    s.add_argument("-o", "--out")
    runnable(s)
    s.set_defaults(fn=cmd_quantize)

    s = sub.add_parser("lora", help="add, rm, list, bind, unbind, route, "
                       "apply, fuse, serve, lineage")
    s.add_argument("action", choices=["add", "rm", "list", "bind", "unbind",
                                      "route", "apply", "fuse", "serve",
                                      "lineage"])
    s.add_argument("args", nargs="*")
    s.add_argument("--base-model")
    s.add_argument("--served-as", help="add: the model name a server "
                   "offers it under (default: its name)")
    s.add_argument("--force", action="store_true")
    s.add_argument("--yes", action="store_true", help="apply: write")
    s.add_argument("--overwrite", action="store_true",
                   help="apply: also over a file's own model")
    s.add_argument("--how", choices=FUSE_HOW, default="linear",
                   help="fuse: PEFT's combination type")
    s.add_argument("--density", type=float, default=0.5,
                   help="fuse: for ties, dare_* and magnitude_prune")
    s.add_argument("--server", choices=["vllm", "ollama", "llama"],
                   default="vllm")
    s.add_argument("--ollama-base", help="serve ollama: the Ollama model "
                   "of the base, for FROM")
    s.add_argument("--serverless", action="store_true",
                   help="serve vllm: a folder (start.sh, Dockerfile, "
                   "warmup.sh, README.md) set up for cold starts")
    s.add_argument("--weights", choices=["image", "volume", "stream"],
                   default="image", help="--serverless: the weights inside "
                   "the image, on a network volume, or streamed")
    s.add_argument("--max-model-len", type=int, default=4096)
    s.add_argument("--max-num-seqs", type=int, default=16,
                   help="--serverless: requests at once; CUDA graphs are "
                   "captured only up to this")
    s.add_argument("--eager", action="store_true",
                   help="--serverless: no CUDA graphs, the fastest start")
    s.add_argument("--sleep", action="store_true",
                   help="--serverless: sleep mode, to wake in seconds")
    s.add_argument("--cache-dir", default="/cache/vllm",
                   help="--serverless: VLLM_CACHE_ROOT, on a lasting volume")
    s.add_argument("--out-dir", help="--serverless: the folder to write "
                   "(default models/serve/BASE)")
    runnable(s)
    s.set_defaults(fn=cmd_lora)

    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "lora":
        need = {"add": 2, "rm": 1, "bind": 2, "unbind": 1, "route": 1}
        if a.action in need and len(a.args) != need[a.action]:
            print("error: lora %s takes %d argument(s)" % (
                a.action, need[a.action]), file=sys.stderr)
            return 2
    try:
        return a.fn(a) or 0
    except BridgeError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    except tawjih.Die:
        return 1


if __name__ == "__main__":
    sys.exit(main())
