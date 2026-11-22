#!/usr/bin/env python3
"""neural_bridge - from a Tawjih project to trained adapters and back.

A bridge, not a trainer: standard library only. The heavy work (PyTorch,
transformers, PEFT, TRL, Unsloth, axolotl, llama.cpp) is done by programs
this bridge writes or calls, in the Python you name with --python. Every
command that would start such a program only shows its plan unless --run
is given.

    compile    the examples of the project as SFT, evaluation and DPO data
    env        which engines the training Python has, and the GPU
    train      a PEFT adapter: LoRA, QLoRA, DoRA, rsLoRA, PiSSA, OLoRA,
               LoHa, LoKr, IA3, VeRA, OFT, BOFT, FourierFT, HRA, RandLoRA,
               LayerNorm tuning, prompt tuning, P-tuning (--peft; engines:
               torch, unsloth, axolotl)
    merge      an adapter merged into its base model (PyTorch + PEFT)
    convert    a merged model, or an adapter, to GGUF (llama.cpp)
    quantize   a GGUF model to Q4_K_M, Q8_0, ... (llama.cpp)
    lora       the project's adapters: add, rm, list, bind, unbind, route,
               apply (sets each file's model: line), fuse, serve, lineage
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


def cmd_train(a):
    """Write the training script (or axolotl config) and the configuration
    into models/adapters/NAME/, with the dataset's manifest for lineage;
    run it with --run."""
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
    if getattr(a, "runtime", None):
        # a job folder with relative paths, to run here or elsewhere
        job_cfg = dict(cfg, dataset="data/" + ds.name, output_dir="out",
                       base_model=job_base(a, a.base_model),
                       eval="data/eval-" + Path(a.eval).name if a.eval
                       else "")
        files = {"data/" + ds.name: ds}
        if a.eval:
            files["data/eval-" + Path(a.eval).name] = Path(a.eval)
        texts = {"train_config.json": json.dumps(job_cfg, indent=2)}
        if a.engine == "axolotl":
            texts["axolotl.yml"] = axolotl_config(job_cfg)
            prog = ["axolotl", "train", "axolotl.yml"]
        else:
            texts["train.py"] = (adir / "train.py").read_text(
                encoding="utf-8")
            prog = None
        return job(a, "train " + a.name, prog, ["train.py",
                                                 "train_config.json"],
                   files, texts, adir, {"kind": "train",
                                        "adapter": a.name,
                                        "path": rel(a, adir),
                                        "base": a.base_model,
                                        "rank": cfg["lora_r"],
                                        "peft": a.peft})
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
    if not reg["adapters"]:
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
        for base, items in by_base.items():
            mods = ["%s=%s" % (served_name(n, e), abs_path(a, e["path"]))
                    for n, e in items]
            rank = max([adapter_rank(abs_path(a, e["path"]), e) or 16
                        for _, e in items])
            print(show_cmd(["vllm", "serve", base, "--enable-lora",
                            "--max-lora-rank", str(max(rank, 8)),
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
# ------------------------------------------------------------------ runtime
def job_base(a, base):
    """A job runs inside its own folder, so a model folder given by a
    relative path is made absolute. On another computer the folder must be
    there under the same path, or use a Hugging Face model id."""
    if not base or not Path(base).exists():
        return base
    _, cfg = runtimes.get(a.runtime)
    if not runtimes.is_local(cfg):
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

    s = sub.add_parser("train", help="train a PEFT adapter: LoRA, QLoRA, "
                       "DoRA, LoHa, LoKr, IA3, VeRA, OFT, prompt tuning ...")
    s.add_argument("name")
    s.add_argument("--base-model", required=True,
                   help="Hugging Face id or local folder")
    s.add_argument("--dataset", default="datasets/sft_train.jsonl")
    s.add_argument("--eval", help="evaluation data (sft_eval.jsonl)")
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
    on_runtime(s)
    runnable(s)
    s.set_defaults(fn=cmd_train)

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
