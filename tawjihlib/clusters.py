"""Model servers on Kubernetes and on your own machines through Ansible.

Tawjih writes what these tools read (a Deployment and a Service; a
playbook) from a server in servers.json, and runs kubectl and
ansible-playbook for you, as they are installed on this computer. Nothing
here replaces them: anything beyond a model server's life is theirs.

Part of Tawjih. Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from .core import Store, die
from .providers import POLICY_FILE, load_policy
from .serving import ENGINES, _note, command, get_server, health_path

LABEL = "app.kubernetes.io/name"
MANAGED = "app.kubernetes.io/managed-by"


# --------------------------------------------------------------------- yaml
def _scalar(v):
    if v is True or v is False:
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return str(v)
    return json.dumps(str(v), ensure_ascii=False)   # JSON strings are YAML


def to_yaml(obj, indent=0):
    """Plain YAML for dicts, lists and scalars (all strings quoted)."""
    pad = "  " * indent
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                lines.append("%s%s:" % (pad, k))
                lines.append(to_yaml(v, indent + 1))
            else:
                lines.append("%s%s: %s" % (pad, k, _scalar(v) if not
                                           isinstance(v, (dict, list)) else
                                           "{}" if isinstance(v, dict)
                                           else "[]"))
    else:
        for v in obj:
            if isinstance(v, dict) and v:
                inner = to_yaml(v, indent + 1).split("\n")
                lines.append("%s- %s" % (pad, inner[0].lstrip()))
                lines += inner[1:]
            elif isinstance(v, list) and v:
                lines.append("%s-" % pad)
                lines.append(to_yaml(v, indent + 1))
            else:
                lines.append("%s- %s" % (pad, _scalar(v)))
    return "\n".join(lines)


def _gpu_count(cfg, gpus):
    if gpus is not None:
        return gpus
    g = str(cfg.get("gpus") or "")
    if not g or g == "auto":
        return 0
    return len([x for x in g.split(",") if x.strip()])


def _image(name, cfg, image):
    image = image or cfg.get("image") or ENGINES[cfg["engine"]].get("image")
    if not image:
        die("%s: its engine (%s) has no image of its own; give --image"
            % (name, cfg["engine"]))
    return image


# --------------------------------------------------------------- kubernetes
def manifest(name, cfg, image=None, replicas=1, namespace=None, gpus=None,
             claim=None):
    """[Deployment, Service] of a server, as dicts."""
    c = dict(cfg, host="0.0.0.0", python="python3", exe=None)
    argv, env = command(c)
    port = int(cfg["port"])
    app = "tawjih-" + name
    labels = {LABEL: app, MANAGED: "tawjih"}
    meta = {"name": app, "labels": labels}
    if namespace:
        meta["namespace"] = namespace
    secret_vars = sorted({w[1:] for w in argv if w.startswith("$")} |
                         {v[1:] for v in env.values() if v.startswith("$")})
    envs = []
    for k, v in sorted(env.items()):
        if k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
            continue          # the cluster gives the pod its GPUs
        if v.startswith("$"):
            envs.append({"name": k, "valueFrom": {"secretKeyRef": {
                "name": app + "-key", "key": v[1:]}}})
        else:
            envs.append({"name": k, "value": v})
    for var in secret_vars:
        if not any(e["name"] == var for e in envs):
            envs.append({"name": var, "valueFrom": {"secretKeyRef": {
                "name": app + "-key", "key": var}}})
    args = ["$(%s)" % w[1:] if w.startswith("$") else w for w in argv]
    n = _gpu_count(cfg, gpus)
    box = {"name": cfg["engine"], "image": _image(name, cfg, image),
           "command": args, "ports": [{"containerPort": port,
                                       "name": "http"}]}
    if envs:
        box["env"] = envs
    if n:
        box["resources"] = {"limits": {"nvidia.com/gpu": n}}
    box["readinessProbe"] = {"httpGet": {"path": health_path(cfg).split(
        "?")[0] if not health_path(cfg).startswith(":") else "/",
        "port": port}, "initialDelaySeconds": 10, "periodSeconds": 10,
        "failureThreshold": 60}
    mounts = [{"name": "shm", "mountPath": "/dev/shm"}]
    volumes = [{"name": "shm", "emptyDir": {"medium": "Memory"}}]
    model = cfg.get("model") or ""
    if claim:
        where = model if model.startswith("/") else "/models"
        mounts.append({"name": "models", "mountPath": where})
        volumes.append({"name": "models", "persistentVolumeClaim": {
            "claimName": claim}})
    box["volumeMounts"] = mounts
    deploy = {"apiVersion": "apps/v1", "kind": "Deployment",
              "metadata": meta,
              "spec": {"replicas": int(replicas or 1),
                       "selector": {"matchLabels": {LABEL: app}},
                       "template": {"metadata": {"labels": labels},
                                    "spec": {"containers": [box],
                                             "volumes": volumes}}}}
    service = {"apiVersion": "v1", "kind": "Service",
               "metadata": dict(meta),
               "spec": {"selector": {LABEL: app},
                        "ports": [{"name": "http", "port": port,
                                   "targetPort": port}]}}
    return [deploy, service]


def manifest_text(name, cfg, **kw):
    docs = manifest(name, cfg, **kw)
    head = "# Tawjih server %s (%s), from servers.json\n" % (name,
                                                             cfg["engine"])
    keys = [e["valueFrom"]["secretKeyRef"]["key"] for e in
            docs[0]["spec"]["template"]["spec"]["containers"][0].get(
                "env", []) if "valueFrom" in e]
    if keys:
        head += ("# the key comes from a Secret, made once with:\n"
                 "#   kubectl create secret generic tawjih-%s-key %s\n" % (
                     name, " ".join("--from-literal=%s=..." % k
                                    for k in keys)))
    model = cfg.get("model") or ""
    if model and Path(model).exists() and not kw.get("claim"):
        head += ("# %s is a folder on this computer: the pod needs it in "
                 "its image or on a volume (--claim NAME)\n" % model)
    return head + "\n---\n".join(to_yaml(d) for d in docs) + "\n"


def _tool(env, name, hint):
    exe = os.environ.get(env) or shutil.which(name)
    if not exe:
        die("%s is not installed here (%s); or set %s to its path" % (
            name, hint, env))
    return exe


def _network_allowed(root, what):
    if load_policy(root)["allow_network"] is not True:
        die("the project's policy (%s) does not allow the network; %s "
            "reaches other machines" % (POLICY_FILE, what))


def _kubectl(a, *args, data=None, capture=False):
    argv = [_tool("KUBECTL", "kubectl",
                  "https://kubernetes.io/docs/tasks/tools/")]
    if a.context:
        argv += ["--context", a.context]
    if a.namespace:
        argv += ["--namespace", a.namespace]
    argv += list(args)
    print("$ " + " ".join(shlex.quote(x) for x in ["kubectl"] + argv[1:]),
          file=sys.stderr)
    r = subprocess.run(argv, input=data, text=True,
                       capture_output=capture)
    return r


def k8s_summary(data):
    """Lines from `kubectl get deployment,service,pod -o json`."""
    lines = []
    for it in data.get("items", []):
        kind, meta = it.get("kind"), it.get("metadata", {})
        st, sp = it.get("status", {}), it.get("spec", {})
        if kind == "Deployment":
            lines.append("deployment %-24s %s/%s ready, %s updated" % (
                meta.get("name"), st.get("readyReplicas", 0),
                sp.get("replicas", 0), st.get("updatedReplicas", 0)))
        elif kind == "Service":
            ports = ",".join(str(p.get("port")) for p in sp.get("ports", []))
            lines.append("service    %-24s %s:%s" % (
                meta.get("name"), sp.get("clusterIP", "?"), ports))
        elif kind == "Pod":
            cs = st.get("containerStatuses") or [{}]
            restarts = sum(c.get("restartCount", 0) for c in cs)
            why = ""
            for c in cs:
                w = (c.get("state") or {}).get("waiting") or {}
                if w.get("reason"):
                    why = " (%s)" % w["reason"]
            lines.append("pod        %-24s %s%s, %d restarts%s" % (
                meta.get("name"), st.get("phase", "?"),
                ", ready" if all(c.get("ready") for c in cs) else "",
                restarts, why))
    return lines or ["nothing of Tawjih's here"]


def cmd_k8s(a):
    st = Store(a.project)
    act = a.action
    if act != "status" and not a.name:
        die("k8s %s needs a server NAME" % act)
    cfg = get_server(st.root, a.name) if a.name else None
    app = "tawjih-" + a.name if a.name else None
    if act == "manifest":
        text = manifest_text(a.name, cfg, image=a.image, replicas=a.replicas,
                             namespace=a.namespace, gpus=a.gpus,
                             claim=a.claim)
        if a.out:
            Path(a.out).write_text(text, encoding="utf-8")
            print("written to %s (apply it: kubectl apply -f %s, or "
                  "tawjih k8s apply %s)" % (a.out, a.out, a.name))
        else:
            sys.stdout.write(text)
        return 0
    _network_allowed(st.root, "kubectl")
    sel = "%s=%s" % (LABEL, app) if app else "%s=tawjih" % MANAGED
    if act == "status":
        r = _kubectl(a, "get", "deployment,service,pod", "-l", sel,
                     "-o", "json", capture=True)
        if r.returncode:
            die("kubectl: %s" % (r.stderr or r.stdout).strip()[:300])
        try:
            data = json.loads(r.stdout or "{}")
        except ValueError:
            die("kubectl did not answer JSON")
        print("\n".join(k8s_summary(data)))
        return 0
    if act == "logs":
        return _kubectl(a, "logs", "deployment/" + app, "--tail",
                        str(a.lines)).returncode
    if act == "forward":
        port = int(cfg["port"])
        local = a.local_port or port
        eng = ENGINES[cfg["engine"]]
        print("# while this runs the server answers at "
              "http://127.0.0.1:%d%s; as a provider:\n#   tawjih provider "
              "add %s --kind %s --base-url http://127.0.0.1:%d%s" % (
                  local, eng.get("base") or "", a.name,
                  eng.get("kind") or "openai", local, eng.get("base") or ""),
              file=sys.stderr)
        return _kubectl(a, "port-forward", "service/" + app,
                        "%d:%d" % (local, port)).returncode
    changes = {"apply": "applied", "scale": "scaled", "rollback":
               "rolled back", "delete": "deleted"}
    if act == "scale" and a.replicas is None:
        die("k8s scale needs --replicas N")
    if not a.yes:
        if act == "apply":
            text = manifest_text(a.name, cfg, image=a.image,
                                 replicas=a.replicas or 1,
                                 namespace=a.namespace, gpus=a.gpus,
                                 claim=a.claim)
            r = _kubectl(a, "apply", "--dry-run=server", "-f", "-",
                         data=text)
            print("checked by the cluster, nothing changed; add --yes to "
                  "apply" if r.returncode == 0 else "the cluster refused it")
            return r.returncode
        what = {"scale": "set %s to %s replicas" % (app, a.replicas),
                "rollback": "go back to %s's previous version" % app,
                "delete": "delete %s and its service" % app}[act]
        print("this would %s; add --yes" % what)
        return 1
    if act == "apply":
        text = manifest_text(a.name, cfg, image=a.image,
                             replicas=a.replicas or 1, namespace=a.namespace,
                             gpus=a.gpus, claim=a.claim)
        r = _kubectl(a, "apply", "-f", "-", data=text)
    elif act == "scale":
        r = _kubectl(a, "scale", "deployment/" + app,
                     "--replicas=%d" % a.replicas)
    elif act == "rollback":
        r = _kubectl(a, "rollout", "undo", "deployment/" + app)
    else:
        r = _kubectl(a, "delete", "deployment,service", "-l", sel)
    if r.returncode == 0:
        _note(st.root, "k8s " + act, a.name,
              **{k: v for k, v in (("replicas", a.replicas),
                                   ("namespace", a.namespace),
                                   ("context", a.context)) if v})
        print("%s %s" % (app, changes[act]))
    return r.returncode


# ------------------------------------------------------------------ ansible
ENGINE_PACKAGES = {"vllm": "vllm", "sglang": "sglang[all]",
                   "lmdeploy": "lmdeploy", "aphrodite": "aphrodite-engine",
                   "litellm": "litellm[proxy]"}


def remote_unit(name, cfg, user, listen=None):
    """A systemd unit for another machine: the engine found on its PATH."""
    argv, env = command(dict(cfg, exe=None, python="python3",
                             host=listen or cfg.get("host") or "127.0.0.1"))
    var = cfg.get("api_key_env")
    lines = ["[Unit]", "Description=Tawjih server %s (%s)" % (
        name, cfg["engine"]), "After=network-online.target", "",
        "[Service]", "Type=simple", "WorkingDirectory=/var/lib/tawjih"]
    for k, v in sorted(env.items()):
        if not v.startswith("$"):
            lines.append("Environment=%s=%s" % (k, v))
    if var or any(v.startswith("$") for v in env.values()):
        lines.append("EnvironmentFile=/etc/tawjih/%s.env" % name)
    words = ["${%s}" % w[1:] if w.startswith("$") else shlex.quote(w)
             for w in argv]
    lines += ["ExecStart=/usr/bin/env " + " ".join(words),
              "Restart=on-failure", "RestartSec=10", "User=%s" % user, "",
              "[Install]", "WantedBy=multi-user.target", ""]
    return "\n".join(lines)


def playbook(name, cfg, hosts="tawjih", user="root", install=False,
             state="started", listen=None):
    """A playbook that puts a server on each host as a systemd service."""
    unit = "tawjih-%s.service" % name
    eng = ENGINES[cfg["engine"]]
    tasks = []
    if install:
        pkg = ENGINE_PACKAGES.get(cfg["engine"])
        if not pkg:
            die("no pip package is known for %s; install it on the hosts "
                "yourself, or leave out --install" % cfg["engine"])
        tasks.append({"name": "install %s" % pkg,
                      "ansible.builtin.pip": {"name": pkg}})
    tasks.append({"name": "a folder for the server",
                  "ansible.builtin.file": {"path": "/var/lib/tawjih",
                                           "state": "directory",
                                           "owner": user}})
    keyvars = sorted({v[1:] for v in command(cfg)[1].values()
                      if v.startswith("$")} |
                     ({cfg["api_key_env"]} if cfg.get("api_key_env") else
                      set()))
    for var in keyvars:
        tasks.append({
            "name": "the key, from %s on this computer" % var,
            "ansible.builtin.copy": {
                "dest": "/etc/tawjih/%s.env" % name, "mode": "0600",
                "content": "%s={{ lookup('env', '%s') }}\n" % (var, var)},
            "no_log": True})
    tasks.append({"name": "the unit",
                  "ansible.builtin.copy": {
                      "dest": "/etc/systemd/system/" + unit,
                      "content": remote_unit(name, cfg, user, listen)}})
    tasks.append({"name": "%s %s" % (unit, state),
                  "ansible.builtin.systemd": {
                      "name": unit, "daemon_reload": True,
                      "enabled": state != "stopped", "state": state}})
    if state != "stopped":
        path = health_path(cfg)
        if not path.startswith(":"):
            tasks.append({
                "name": "wait until it answers",
                "ansible.builtin.uri": {
                    "url": "http://127.0.0.1:%s%s" % (cfg["port"], path)},
                "register": "health",
                "until": "health.status == 200",
                "retries": 60, "delay": 10})
    return [{"name": "Tawjih server %s (%s)" % (name, cfg["engine"]),
             "hosts": hosts, "become": True, "tasks": tasks,
             "vars": {"tawjih_server": name, "tawjih_port": cfg["port"],
                      "tawjih_kind": eng.get("kind") or ""}}]


def _inventory(a, root):
    if a.inventory:
        return a.inventory
    if not a.hosts:
        die("give --hosts h1,h2 or --inventory FILE")
    hosts = [h.strip() for h in a.hosts.split(",") if h.strip()]
    for h in hosts:
        if not re.fullmatch(r"[\w.\-]+(@[\w.\-]+)?", h):
            die("not a host: %r" % h)
    d = Path(root) / "servers" / "ansible"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "hosts.ini"
    lines = ["[tawjih]"]
    for h in hosts:
        user, _, host = h.rpartition("@")
        lines.append(host + (" ansible_user=%s" % user if user else ""))
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(f)


def cmd_ansible(a):
    st = Store(a.project)
    cfg = get_server(st.root, a.name)
    text = "# Tawjih server %s, from servers.json\n" % a.name + to_yaml(
        playbook(a.name, cfg, a.group, a.user, a.install, a.state,
                 a.listen)) + "\n"
    if a.action == "playbook":
        if a.out:
            Path(a.out).write_text(text, encoding="utf-8")
            print("written to %s (run it: ansible-playbook -i HOSTS %s, or "
                  "tawjih ansible run %s --hosts ...)" % (a.out, a.out,
                                                         a.name))
        else:
            sys.stdout.write(text)
        return 0
    _network_allowed(st.root, "ansible-playbook")
    inv = _inventory(a, st.root)
    d = Path(st.root) / "servers" / "ansible"
    d.mkdir(parents=True, exist_ok=True)
    book = d / ("%s.yml" % a.name)
    book.write_text(text, encoding="utf-8")
    argv = [_tool("ANSIBLE_PLAYBOOK", "ansible-playbook",
                  "pip install ansible-core"), "-i", inv, str(book)]
    if a.limit:
        argv += ["--limit", a.limit]
    if not a.yes:
        argv += ["--check", "--diff"]
    print("$ " + " ".join(shlex.quote(x) for x in
                          ["ansible-playbook"] + argv[1:]), file=sys.stderr)
    r = subprocess.run(argv)
    if not a.yes:
        print("a check only: nothing changed on the hosts; add --yes to "
              "do it" if r.returncode == 0 else "the check failed")
    elif r.returncode == 0:
        _note(st.root, "ansible " + a.state, a.name, inventory=inv,
              **({"limit": a.limit} if a.limit else {}))
    return r.returncode


def add_cluster_parsers(sub):
    s = sub.add_parser("k8s", aliases=["عنقود"], help="a model server on "
                       "Kubernetes: its manifest, apply, status, logs, "
                       "scale, rollback, forward, delete (through kubectl)")
    s.add_argument("action", choices=["manifest", "apply", "status", "logs",
                                      "scale", "rollback", "forward",
                                      "delete"])
    s.add_argument("name", nargs="?", help="a server in servers.json")
    s.add_argument("--image", help="the container image (by default the "
                   "engine's)")
    s.add_argument("--replicas", type=int)
    s.add_argument("--gpus", type=int, metavar="N", help="GPUs per pod "
                   "(by default as many as the server's --gpus)")
    s.add_argument("--claim", metavar="PVC", help="a PersistentVolumeClaim "
                   "holding the model, mounted at its path (or /models)")
    s.add_argument("--namespace", "-n")
    s.add_argument("--context", help="a kubectl context")
    s.add_argument("--lines", type=int, default=100)
    s.add_argument("--local-port", type=int)
    s.add_argument("-o", "--out", help="manifest: write it here")
    s.add_argument("--yes", action="store_true", help="apply, scale, "
                   "rollback, delete: do it (without: apply is checked by "
                   "the cluster, the others say what they would do)")
    s.set_defaults(fn=cmd_k8s)

    s = sub.add_parser("ansible", aliases=["أنسبل"], help="a model server "
                       "on your own machines as a systemd service, through "
                       "ansible-playbook")
    s.add_argument("action", choices=["playbook", "run"])
    s.add_argument("name", help="a server in servers.json")
    s.add_argument("--hosts", help="host,user@host,... (written to "
                   "servers/ansible/hosts.ini)")
    s.add_argument("--inventory", "-i", help="an inventory of your own")
    s.add_argument("--group", default="tawjih", help="the hosts' group in "
                   "the inventory (tawjih)")
    s.add_argument("--limit", help="only these hosts of the inventory")
    s.add_argument("--user", default="root", help="the user the service "
                   "runs as (root)")
    s.add_argument("--install", action="store_true", help="pip install the "
                   "engine on the hosts first")
    s.add_argument("--listen", metavar="ADDRESS", help="the address the "
                   "server listens on there (by default the server's host: "
                   "127.0.0.1 answers only on that machine; 0.0.0.0 on "
                   "every network, so give it a key)")
    s.add_argument("--state", default="started",
                   choices=["started", "restarted", "stopped"])
    s.add_argument("-o", "--out", help="playbook: write it here")
    s.add_argument("--yes", action="store_true", help="run: do it "
                   "(without: ansible-playbook --check --diff)")
    s.set_defaults(fn=cmd_ansible)


__all__ = ["ENGINE_PACKAGES", "add_cluster_parsers", "cmd_ansible",
           "cmd_k8s", "k8s_summary", "manifest", "manifest_text",
           "playbook", "remote_unit", "to_yaml"]
