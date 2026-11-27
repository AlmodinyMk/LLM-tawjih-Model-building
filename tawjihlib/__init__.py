"""tawjih - personal LLM command-directing (توجيه أوامر الآلة).

The package behind tawjih.py, in modules that each import only from the ones before
them: core, providers, tracing, engine, commands, library, manage, outputs, evaluate, governance, runtimes, serving, arch, script, mcpserver, home, monitor, dashboard, cli.
"""

from importlib import import_module as _import

MODULES = ("core", "providers", "tracing", "engine", "commands", "library", "manage", "outputs", "evaluate", "governance", "runtimes", "serving", "clusters", "arch", "script", "mcpserver", "home", "monitor", "dashboard", "health", "cli")  # noqa: E501
for _name in MODULES:
    _m = _import("%s.%s" % (__name__, _name))
    globals().update({_k: getattr(_m, _k) for _k in _m.__all__})
