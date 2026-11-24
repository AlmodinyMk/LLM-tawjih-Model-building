"""The architecture lab: language-model architectures as Tawjih files.

An architecture is an ordinary three-faced file tagged `architecture`: its
human face holds the hypothesis and the design decisions, its science face
the papers it rests on, and its machine face the specification, as JSON.
A derived architecture is a child of the one it comes from, so the tree is
its lineage, and locks, reviews, releases and trace work on it as on any
file.

Everything here is standard library: checking a specification, counting
its parameters and memory, comparing two, proposing variants, and writing
a configuration that existing libraries build (Hugging Face transformers).
Building and testing a real model is a job (`arch build`) that runs where
PyTorch is, here or on a runtime.

Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""
import json
import sys
from pathlib import Path

from .core import Store, die
from .runtimes import (
    allowed as rt_allowed, collect as rt_collect, get as rt_get,
    make_job, plan as rt_plan, record as rt_record, submit as rt_submit,
)

TAG = "architecture"
CHOICES = {"family": ("decoder",), "ffn": ("swiglu", "gelu", "relu"),
           "norm": ("rmsnorm", "layernorm"),
           "positions": ("rope", "learned", "none")}
INTS = ("layers", "hidden", "heads", "kv_heads", "ffn_hidden", "vocab",
        "context")
ROUTER_LOSS = 0.001        # transformers' default weight for Mixtral
DEFAULT = {"schema": 1, "family": "decoder", "layers": 12, "hidden": 768,
           "heads": 12, "kv_heads": 4, "ffn": "swiglu", "ffn_hidden": 2048,
           "norm": "rmsnorm", "positions": "rope", "rope_theta": 10000,
           "vocab": 32000, "context": 2048, "tie_embeddings": True,
           "bias": False, "moe": None, "dropout": 0.0}
PRESETS = {
    "tiny": dict(DEFAULT, layers=4, hidden=256, heads=4, kv_heads=2,
                 ffn_hidden=688, vocab=32000, context=1024),
    "small": dict(DEFAULT),
    "llama-7b-like": dict(DEFAULT, layers=32, hidden=4096, heads=32,
                          kv_heads=32, ffn_hidden=11008, context=4096,
                          tie_embeddings=False),
    "gpt2-like": dict(DEFAULT, layers=12, hidden=768, heads=12, kv_heads=12,
                      ffn="gelu", ffn_hidden=3072, norm="layernorm",
                      positions="learned", vocab=50257, context=1024,
                      tie_embeddings=True, bias=True),
    "moe-small": dict(DEFAULT, moe={"experts": 8, "top_k": 2}),
}
KEYS = set(DEFAULT) | {"derived_from", "name", "notes"}
HUMAN_TEMPLATE = ("## Hypothesis\n\n## Design decisions\n\n"
                  "## Known limitations\n")


# ------------------------------------------------------------ the spec
def spec_of(n):
    """The specification in a file's machine face, or die."""
    try:
        spec = json.loads(n.instructions)
    except ValueError as e:
        die("%s [%s]: the machine face is not a JSON specification (%s)"
            % (n.title, n.id, e))
    if not isinstance(spec, dict):
        die("%s [%s]: the specification is not a JSON object"
            % (n.title, n.id))
    return spec


def problems(spec):
    """What is wrong with a specification: [] when it can be built."""
    out = []
    unknown = sorted(set(spec) - KEYS)
    if unknown:
        out.append("unknown key(s): %s (known: %s)" % (
            ", ".join(unknown), ", ".join(sorted(KEYS))))
    for k in INTS:
        v = spec.get(k)
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            out.append("%s must be a whole number above 0 (it is %r)"
                       % (k, v))
    for k, allowed in CHOICES.items():
        if spec.get(k) not in allowed:
            out.append("%s must be one of %s (it is %r)" % (
                k, ", ".join(allowed), spec.get(k)))
    for k in ("tie_embeddings", "bias"):
        if not isinstance(spec.get(k), bool):
            out.append("%s must be true or false" % k)
    d = spec.get("dropout", 0)
    if not isinstance(d, (int, float)) or not 0 <= d < 1:
        out.append("dropout must be from 0 up to (not including) 1")
    if out:
        return out
    h, heads, kv = spec["hidden"], spec["heads"], spec["kv_heads"]
    if h % heads:
        out.append("hidden (%d) must divide evenly by heads (%d)"
                   % (h, heads))
    elif spec["positions"] == "rope" and (h // heads) % 2:
        out.append("with rope the head size (hidden / heads = %d) must be "
                   "even" % (h // heads))
    if kv > heads or heads % kv:
        out.append("kv_heads (%d) must divide heads (%d): heads for MHA, "
                   "1 for MQA, a divisor for GQA" % (kv, heads))
    moe = spec.get("moe")
    if moe is not None:
        if not isinstance(moe, dict) or \
                set(moe) - {"experts", "top_k", "router_loss",
                            "router_jitter"}:
            out.append('moe must be null or {"experts": N, "top_k": K} '
                       '(and "router_loss": C, "router_jitter": J)')
        else:
            e, k = moe.get("experts"), moe.get("top_k")
            if not (isinstance(e, int) and isinstance(k, int)
                    and 1 <= k <= e and e >= 2):
                out.append("moe needs experts of 2 or more and top_k from 1 "
                           "to experts")
            c = moe.get("router_loss", ROUTER_LOSS)
            if isinstance(c, bool) or not isinstance(c, (int, float)) or \
                    not 0 <= c <= 1:
                out.append("moe router_loss is the weight of the load "
                           "balancing loss, from 0 (off) to 1")
            j = moe.get("router_jitter", 0)
            if isinstance(j, bool) or not isinstance(j, (int, float)) or \
                    not 0 <= j < 1:
                out.append("moe router_jitter is the noise on the router's "
                           "input in training, from 0 (none) to below 1")
    if spec["positions"] == "rope":
        t = spec.get("rope_theta")
        if not isinstance(t, (int, float)) or t <= 0:
            out.append("rope_theta must be a number above 0")
    return out


def attention_kind(spec):
    return "MHA" if spec["kv_heads"] == spec["heads"] else \
        "MQA" if spec["kv_heads"] == 1 else "GQA"


def count(spec):
    """Parameters, by part; 'active' is per token (with MoE, top_k experts
    of each layer)."""
    h, L, v = spec["hidden"], spec["layers"], spec["vocab"]
    hd = h // spec["heads"]
    kvd = spec["kv_heads"] * hd
    bias = spec["bias"]
    embed = v * h + (spec["context"] * h if spec["positions"] == "learned"
                     else 0)
    attn = 2 * h * h + 2 * h * kvd + ((2 * h + 2 * kvd) if bias else 0)
    f = spec["ffn_hidden"]
    if spec["ffn"] == "swiglu":
        mlp = 3 * h * f + ((2 * f + h) if bias else 0)
    else:
        mlp = 2 * h * f + ((f + h) if bias else 0)
    moe = spec.get("moe")
    mlp_all = mlp * moe["experts"] + h * moe["experts"] if moe else mlp
    mlp_act = mlp * moe["top_k"] + h * moe["experts"] if moe else mlp
    norm = h if spec["norm"] == "rmsnorm" else 2 * h
    head = 0 if spec["tie_embeddings"] else v * h
    total = embed + L * (attn + mlp_all + 2 * norm) + norm + head
    active = embed + L * (attn + mlp_act + 2 * norm) + norm + head
    return {"embedding": embed, "attention": L * attn, "mlp": L * mlp_all,
            "norms": L * 2 * norm + norm, "head": head, "total": total,
            "active": active}


def sizes(spec, params):
    """Estimates, in bytes: weights at several precisions, the KV cache per
    token of context, and training memory for weights, gradients and Adam
    (16 bytes a parameter, before activations)."""
    hd = spec["hidden"] // spec["heads"]
    kv_token = 2 * spec["layers"] * spec["kv_heads"] * hd * 2
    return {"fp32": params["total"] * 4, "bf16": params["total"] * 2,
            "int8": params["total"], "int4": params["total"] // 2,
            "kv_per_token_bf16": kv_token,
            "kv_full_context_bf16": kv_token * spec["context"],
            "train_adam_estimate": params["total"] * 16,
            "flops_per_token_forward": 2 * params["active"],
            "flops_per_token_train": 6 * params["active"]}


def human(n):
    for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= size:
            return "%.2f%s" % (n / size, unit)
    return str(n)


def gb(b):
    return "%.2f GB" % (b / 1e9) if b >= 1e8 else "%.1f MB" % (b / 1e6)


# --------------------------------------------------------------- export
def hf_config(spec):
    """A Hugging Face transformers config.json for the specification, or
    die naming what has no equivalent there."""
    moe, bias = spec.get("moe"), spec["bias"]
    if spec["norm"] == "rmsnorm" and spec["ffn"] == "swiglu" and \
            spec["positions"] == "rope":
        cfg = {"hidden_size": spec["hidden"],
               "intermediate_size": spec["ffn_hidden"],
               "num_hidden_layers": spec["layers"],
               "num_attention_heads": spec["heads"],
               "num_key_value_heads": spec["kv_heads"],
               "vocab_size": spec["vocab"],
               "max_position_embeddings": spec["context"],
               "rope_theta": spec["rope_theta"], "rms_norm_eps": 1e-5,
               "tie_word_embeddings": spec["tie_embeddings"],
               "hidden_act": "silu", "attention_dropout": spec["dropout"],
               "torch_dtype": "bfloat16"}
        if moe:
            if bias:
                die("Mixtral (MoE) has no bias; set bias to false")
            # the router's load-balancing loss is added to the training
            # loss only when the router's logits are returned: without it
            # a router may send every token to the same few experts
            c = moe.get("router_loss", ROUTER_LOSS)
            cfg.update(model_type="mixtral",
                       architectures=["MixtralForCausalLM"],
                       num_local_experts=moe["experts"],
                       num_experts_per_tok=moe["top_k"],
                       output_router_logits=bool(c),
                       router_aux_loss_coef=c,
                       router_jitter_noise=moe.get("router_jitter", 0))
        else:
            cfg.update(model_type="llama",
                       architectures=["LlamaForCausalLM"],
                       attention_bias=bias, mlp_bias=bias)
        return cfg
    if spec["norm"] == "layernorm" and spec["ffn"] == "gelu" and \
            spec["positions"] == "learned" and not moe:
        if not bias or not spec["tie_embeddings"] or \
                spec["kv_heads"] != spec["heads"]:
            die("GPT-2 has bias, tied embeddings and MHA (kv_heads = heads)")
        return {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"],
                "n_embd": spec["hidden"], "n_layer": spec["layers"],
                "n_head": spec["heads"], "n_inner": spec["ffn_hidden"],
                "vocab_size": spec["vocab"], "n_positions": spec["context"],
                "activation_function": "gelu_new",
                "resid_pdrop": spec["dropout"], "embd_pdrop": spec["dropout"],
                "attn_pdrop": spec["dropout"], "tie_word_embeddings": True}
    die("no Hugging Face family matches this combination; supported: "
        "rmsnorm + swiglu + rope (Llama, or Mixtral with moe), and "
        "layernorm + gelu + learned positions (GPT-2)")


def expert_load_line(load):
    """How evenly a built MoE spread its tokens over the experts: per
    layer, the shares of the busiest and the quietest expert against the
    even share, and experts that got no token."""
    parts = []
    for n, shares in enumerate(load, 1):
        idle = sum(1 for x in shares if x == 0)
        parts.append("layer %d %.0f%%-%.0f%%%s" % (
            n, 100 * min(shares), 100 * max(shares),
            ", %d idle" % idle if idle else ""))
    even = 100.0 / len(load[0]) if load and load[0] else 0
    return ("experts on random tokens (an untrained router; even share "
            "%.0f%%): %s" % (even, "; ".join(parts)))


BUILD_SCRIPT = r'''# written by Tawjih: build the architecture and test it on random tokens.
import json, sys, time
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
raw = json.load(open("config.json", encoding="utf-8"))
expected = json.load(open("expected.json", encoding="utf-8"))
kind = raw.pop("model_type")
raw.pop("architectures", None)
cfg = AutoConfig.for_model(kind, **raw)
torch.manual_seed(0)
model = AutoModelForCausalLM.from_config(cfg)
n = sum(p.numel() for p in model.parameters())
length = min(64, expected["context"])
x = torch.randint(0, expected["vocab"], (2, length))
t0 = time.time()
out = model(input_ids=x, labels=x)
aux = getattr(out, "aux_loss", None)
load = None
if getattr(out, "router_logits", None):
    # the share of tokens each expert received, per layer (top_k choices)
    k, e = raw["num_experts_per_tok"], raw["num_local_experts"]
    load = []
    for layer in out.router_logits:
        picked = layer.topk(k, dim=-1).indices.flatten()
        counts = torch.bincount(picked, minlength=e).float()
        load.append([round(float(c), 4) for c in counts / counts.sum()])
out.loss.backward()
seconds = time.time() - t0
grads = sum(1 for p in model.parameters() if p.grad is not None)
report = {"parameters": n, "expected_parameters": expected["total"],
          "parameters_match": n == expected["total"],
          "logits_shape": list(out.logits.shape),
          "loss": float(out.loss), "loss_is_finite": bool(torch.isfinite(
              out.loss)), "tensors_with_gradients": grads,
          "forward_backward_seconds": round(seconds, 3),
          "router_loss": None if aux is None else float(aux),
          "router_loss_in_training": aux is not None and
          raw.get("router_aux_loss_coef", 0) > 0,
          "expert_load": load,
          "device": "cpu", "torch": torch.__version__,
          "transformers": transformers.__version__}
json.dump(report, open("out/validation.json", "w"), indent=2)
print(json.dumps(report, indent=2))
ok = report["parameters_match"] and report["loss_is_finite"] and \
    report["logits_shape"] == [2, length, expected["vocab"]] and \
    (not raw.get("output_router_logits") or (
        aux is not None and bool(torch.isfinite(aux))))
sys.exit(0 if ok else 1)
'''


# -------------------------------------------------------------- commands
def _arch(st, ref):
    nid = st.resolve(ref)
    n = st.nodes[nid]
    if TAG not in n.tags:
        die("%s [%s] is not an architecture (arch create makes one)"
            % (n.title, nid))
    return n


def _write_spec(st, n, spec):
    n.instructions = json.dumps(spec, ensure_ascii=False, indent=2)
    n.built_sha = n.source_digest()
    st.save(n)


def _results_dir(st, nid):
    return st.root / "models" / "architectures" / nid


def _measured(st, nid):
    f = _results_dir(st, nid) / "validation.json"
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _new(st, title, parent, spec, human=HUMAN_TEMPLATE):
    if parent and st.nodes[parent].level >= 6:
        # the deepest level: a derived design goes beside its source, and
        # its derived-from link keeps the lineage
        parent = st.nodes[parent].parent
    level = 1 if not parent else st.nodes[parent].level + 1
    n = st.new_node(title, level, parent)
    n.tags = [TAG]
    n.human = human
    n.instructions = json.dumps(spec, ensure_ascii=False, indent=2)
    n.built_sha = n.source_digest()
    st.save(n)
    return n


def _summary(st, n, spec):
    p = count(spec)
    s = sizes(spec, p)
    moe = spec.get("moe")
    print("%s  [%s]  %s" % (st.number(n.id), n.id, n.title))
    print("  %s, %d layers, hidden %d, %d heads (%s, kv_heads %d), "
          "%s %d%s" % (spec["family"], spec["layers"], spec["hidden"],
                       spec["heads"], attention_kind(spec), spec["kv_heads"],
                       spec["ffn"], spec["ffn_hidden"],
                       ", MoE %d experts, top %d, router loss %g%s" % (
                           moe["experts"], moe["top_k"],
                           moe.get("router_loss", ROUTER_LOSS),
                           ", jitter %g" % moe["router_jitter"]
                           if moe.get("router_jitter") else "")
                       if moe else ""))
    print("  %s, %s positions, vocab %d, context %d%s%s" % (
        spec["norm"], spec["positions"], spec["vocab"], spec["context"],
        ", tied embeddings" if spec["tie_embeddings"] else "",
        ", bias" if spec["bias"] else ""))
    print("  parameters %s (%d)%s" % (human(p["total"]), p["total"],
                                       "; active per token %s" % human(
                                           p["active"]) if moe else ""))
    print("    embedding %s, attention %s, mlp %s, norms %s, head %s" % tuple(
        human(p[k]) for k in ("embedding", "attention", "mlp", "norms",
                              "head")))
    print("  weights: bf16 %s, int8 %s, int4 %s; KV cache %s per token, "
          "%s for the full context" % (gb(s["bf16"]), gb(s["int8"]),
                                       gb(s["int4"]),
                                       gb(s["kv_per_token_bf16"])
                                       if s["kv_per_token_bf16"] >= 1e8
                                       else "%d KB" % (
                                           s["kv_per_token_bf16"] // 1024),
                                       gb(s["kv_full_context_bf16"])))
    print("  training with Adam: about %s before activations (estimate)"
          % gb(s["train_adam_estimate"]))
    if spec.get("derived_from"):
        src = st.nodes.get(spec["derived_from"])
        print("  derived from: %s" % ("%s [%s]" % (src.title, src.id)
                                      if src else spec["derived_from"]))
    m = _measured(st, n.id)
    if m:
        print("  measured (arch build): %d parameters (%s), loss %.3f, "
              "forward+backward %.2f s on %s, torch %s" % (
                  m["parameters"], "as counted" if m.get("parameters_match")
                  else "NOT as counted", m["loss"],
                  m["forward_backward_seconds"], m["device"], m["torch"]))
    else:
        print("  not built yet (arch build %s --run)" % st.number(n.id))


def _set(spec, pairs):
    out = dict(spec)
    for pair in pairs:
        if "=" not in pair:
            die("set takes key=value (for example layers=16)")
        k, v = pair.split("=", 1)
        k = k.strip()
        if k not in KEYS:
            die("unknown key %s (known: %s)" % (k, ", ".join(sorted(KEYS))))
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def _variants(spec, count_):
    """One change each: the usual factors of a design space."""
    s = spec
    out = []

    def add(label, **ch):
        v = dict(s, **ch)
        if v != s and not problems(v):
            out.append((label, v))
    add("layers %d" % (s["layers"] + max(1, s["layers"] // 4)),
        layers=s["layers"] + max(1, s["layers"] // 4))
    if s["layers"] > 2:
        add("layers %d" % max(1, s["layers"] - max(1, s["layers"] // 4)),
            layers=max(1, s["layers"] - max(1, s["layers"] // 4)))
    for kv in sorted({1, max(1, s["kv_heads"] // 2)}):
        if kv < s["kv_heads"] and s["heads"] % kv == 0:
            add("kv_heads %d (%s)" % (kv, "MQA" if kv == 1 else "GQA"),
                kv_heads=kv)
    add("ffn_hidden %d" % int(s["ffn_hidden"] * 1.5),
        ffn_hidden=int(s["ffn_hidden"] * 1.5) // 8 * 8)
    add("ffn_hidden %d" % (int(s["ffn_hidden"] * 0.75) // 8 * 8),
        ffn_hidden=int(s["ffn_hidden"] * 0.75) // 8 * 8)
    if not s.get("moe"):
        add("moe 8 experts, top 2", moe={"experts": 8, "top_k": 2})
    if s["ffn"] != "swiglu":
        add("ffn swiglu", ffn="swiglu",
            ffn_hidden=int(s["ffn_hidden"] * 2 / 3) // 8 * 8)
    if s["norm"] != "rmsnorm":
        add("norm rmsnorm", norm="rmsnorm")
    if s["positions"] != "rope":
        add("positions rope", positions="rope")
    return out[:count_]


def cmd_arch(a):
    """Architectures as Tawjih files: create, derive, list, show, validate,
    set, diff, components, mutate, export, build, rank."""
    st = Store(a.project)
    act, args = a.action, a.args
    if act == "components":
        for k, v in CHOICES.items():
            print("%-10s %s" % (k, ", ".join(v)))
        print("%-10s MHA (kv_heads = heads), GQA (a divisor), MQA (1)"
              % "attention")
        print("%-10s null, or {\"experts\": N, \"top_k\": K, "
              "\"router_loss\": C, \"router_jitter\": J} (C: the "
              "load-balancing loss's weight, default %g, 0 for none; J: "
              "noise on the router's input in training, default 0)"
              % ("moe", ROUTER_LOSS))
        print("presets    %s" % ", ".join(PRESETS))
        return 0
    if act == "create":
        if len(args) != 1:
            die("arch create TITLE [--preset NAME] [--parent REF]")
        if a.preset not in PRESETS:
            die("presets: %s" % ", ".join(PRESETS))
        spec = _set(dict(PRESETS[a.preset], name=args[0]), a.pairs or [])
        probs = problems(spec)
        if probs:
            die("not created: " + "; ".join(probs))
        parent = st.resolve(a.parent) if a.parent else ""
        n = _new(st, args[0], parent, spec)
        print("architecture %s [%s] (preset %s): its hypothesis in the "
              "human face, papers in the science face, the specification "
              "in the machine face" % (st.number(n.id), n.id, a.preset))
        _summary(st, n, spec)
        return 0
    if act == "derive":
        if len(args) != 2:
            die("arch derive SOURCE TITLE [key=value ...]")
        src = _arch(st, args[0])
        spec = _set(dict(spec_of(src), name=args[1], derived_from=src.id),
                    a.pairs or [])
        probs = problems(spec)
        if probs:
            die("the derived specification cannot be built: " +
                "; ".join(probs))
        n = _new(st, args[1], src.id, spec,
                 human="Derived from %s [%s].\n\n%s" % (
                     src.title, src.id, HUMAN_TEMPLATE))
        n.links = [(src.id, "derived-from")]
        st.save(n)
        print("derived %s [%s] from %s [%s]" % (st.number(n.id), n.id,
                                                src.title, src.id))
        _summary(st, n, spec)
        return 0
    if act == "list":
        rows = [n for n in st.nodes.values() if TAG in n.tags]
        if not rows:
            print("no architectures (arch create TITLE)")
        for n in sorted(rows, key=lambda n: st.number(n.id)):
            try:
                spec = json.loads(n.instructions)
                p = count(spec) if not problems(spec) else None
            except ValueError:
                p = None
            m = _measured(st, n.id)
            print("%-10s [%s] %-28s %10s%s" % (
                st.number(n.id), n.id, n.title,
                human(p["total"]) if p else "invalid",
                "  built" if m and m.get("parameters_match") else ""))
        return 0
    if act in ("show", "validate", "export", "build") and len(args) != 1:
        die("arch %s REF" % act)
    if act == "show":
        n = _arch(st, args[0])
        spec = spec_of(n)
        probs = problems(spec)
        if probs:
            print("%s [%s]: the specification cannot be built:\n  %s" % (
                n.title, n.id, "\n  ".join(probs)))
            return 1
        _summary(st, n, spec)
        return 0
    if act == "validate":
        n = _arch(st, args[0])
        probs = problems(spec_of(n))
        if probs:
            print("%s [%s]: %d problem(s)\n  %s" % (
                n.title, n.id, len(probs), "\n  ".join(probs)))
            return 1
        print("%s [%s]: the specification is valid (%s parameters); "
              "arch build checks a real model" % (
                  n.title, n.id, human(count(spec_of(n))["total"])))
        return 0
    if act == "set":
        if len(args) < 2:
            die("arch set REF key=value ...")
        n = _arch(st, args[0])
        spec = _set(spec_of(n), args[1:])
        probs = problems(spec)
        if probs and not a.force:
            die("not changed: " + "; ".join(probs) + " (--force keeps it "
                "anyway)")
        _write_spec(st, n, spec)
        _summary(st, n, spec)
        return 0
    if act == "diff":
        if len(args) != 2:
            die("arch diff A B")
        x, y = _arch(st, args[0]), _arch(st, args[1])
        sx, sy = spec_of(x), spec_of(y)
        print("%s [%s]  ->  %s [%s]" % (x.title, x.id, y.title, y.id))
        same = True
        for k in sorted((set(sx) | set(sy)) - {"name", "notes"}):
            if sx.get(k) != sy.get(k):
                same = False
                print("  %-15s %s -> %s" % (k, json.dumps(sx.get(k)),
                                             json.dumps(sy.get(k))))
        if same:
            print("  the same specification")
        if not (problems(sx) or problems(sy)):
            px, py = count(sx)["total"], count(sy)["total"]
            print("  parameters %s -> %s (%+.1f%%)" % (
                human(px), human(py), (py - px) * 100.0 / px))
        return 0
    if act == "mutate":
        if len(args) != 1:
            die("arch mutate REF [--count N] [--yes]")
        src = _arch(st, args[0])
        spec = spec_of(src)
        vs = _variants(spec, a.count)
        for label, v in vs:
            print("  %-26s %s parameters" % (label, human(count(v)["total"])))
        if not a.yes:
            print("%d variant(s); add --yes to make each a derived "
                  "architecture under %s" % (len(vs), src.title))
            return 0
        for label, v in vs:
            title = "%s: %s" % (src.title, label)
            n = _new(st, title, src.id, dict(v, name=title,
                                             derived_from=src.id),
                     human="Variant of %s [%s]: %s.\n\n%s" % (
                         src.title, src.id, label, HUMAN_TEMPLATE))
            n.links = [(src.id, "derived-from")]
            st.save(n)
            print("made %s [%s] %s" % (st.number(n.id), n.id, title))
        return 0
    if act == "export":
        n = _arch(st, args[0])
        spec = spec_of(n)
        probs = problems(spec)
        if probs:
            die("not exported: " + "; ".join(probs))
        cfg = hf_config(spec)
        out = Path(a.out) if a.out else _results_dir(st, n.id) / "hf"
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n",
                                         encoding="utf-8")
        print("wrote %s (%s); transformers builds it with "
              "AutoModelForCausalLM.from_config" % (out / "config.json",
                                                    cfg["model_type"]))
        return 0
    if act == "build":
        n = _arch(st, args[0])
        spec = spec_of(n)
        probs = problems(spec)
        if probs:
            die("not built: " + "; ".join(probs))
        cfg = hf_config(spec)
        p = count(spec)
        big = p["total"] > 3e8
        if big and not a.force:
            die("%s parameters is large for a test build (it needs about %s "
                "of memory in fp32); add --force, or build on a runtime "
                "with enough memory" % (human(p["total"]),
                                        gb(p["total"] * 4 * 3)))
        name, rcfg = rt_get(a.runtime)
        py = a.python or rcfg.get("python") or (
            sys.executable if rcfg["kind"] == "local" else "python3")
        expected = dict(p, vocab=spec["vocab"], context=spec["context"])
        jdir = make_job(
            st.root, "build " + n.title, name, [py, "build.py"],
            texts={"config.json": json.dumps(cfg, indent=2),
                   "expected.json": json.dumps(expected, indent=2),
                   "build.py": BUILD_SCRIPT},
            collect_to=_results_dir(st, n.id))
        rt_record(jdir, {"kind": "arch-build", "architecture": n.id})
        print("job %s: build %s [%s] (%s, %s parameters) on %s" % (
            jdir.name, n.title, n.id, cfg["model_type"], human(p["total"]),
            name))
        for line in rt_plan(jdir, name, rcfg):
            print("  " + line)
        if not a.run:
            print("(plan only; add --run to build and test it; it needs "
                  "torch and transformers in %s)" % py)
            return 0
        rt_allowed(st.root, name, rcfg, a.yes)
        code = rt_submit(st.root, jdir, name, rcfg)
        if rcfg["kind"] != "local":
            print("started; then: tawjih runtime collect %s" % jdir.name)
            return code
        rt_collect(jdir, rcfg)
        m = _measured(st, n.id)
        if m is None:
            die("the build did not finish; see %s" % (jdir / "log.txt"))
        print("built: %d parameters (%s), logits %s, loss %.3f, "
              "forward+backward %.2f s" % (
                  m["parameters"], "as counted" if m["parameters_match"]
                  else "counted %d" % m["expected_parameters"],
                  m["logits_shape"], m["loss"],
                  m["forward_backward_seconds"]))
        if m.get("router_loss") is not None:
            print("router load-balancing loss %.3f, %s" % (
                m["router_loss"], "added to the training loss" if
                m.get("router_loss_in_training") else "not used in training "
                "(router_loss 0)"))
        if m.get("expert_load"):
            print(expert_load_line(m["expert_load"]))
        return code
    if act == "rank":
        rows = []
        for n in st.nodes.values():
            if TAG not in n.tags:
                continue
            try:
                spec = json.loads(n.instructions)
            except ValueError:
                continue
            if problems(spec):
                continue
            p, m = count(spec), _measured(st, n.id)
            rows.append((n, p, sizes(spec, p), m))
        key = {"params": lambda r: r[1]["total"],
               "kv": lambda r: r[2]["kv_per_token_bf16"],
               "speed": lambda r: (r[3] or {}).get(
                   "forward_backward_seconds", float("inf"))}[a.by]
        print("%-10s %-28s %10s %10s %10s %s" % ("No.", "Title", "params",
                                                 "active", "KV/token",
                                                 "measured"))
        for n, p, s, m in sorted(rows, key=key):
            print("%-10s %-28s %10s %10s %9dK %s" % (
                st.number(n.id), n.title[:28], human(p["total"]),
                human(p["active"]), s["kv_per_token_bf16"] // 1024,
                ("%.2f s, %s" % (m["forward_backward_seconds"], "ok" if
                                 m.get("parameters_match") else "mismatch"))
                if m else "-"))
        print("(counts are exact for the formulas used; memory and FLOPs "
              "are estimates; 'measured' comes from arch build)")
        return 0
    die("unknown action %s" % act)


__all__ = ['cmd_arch']
