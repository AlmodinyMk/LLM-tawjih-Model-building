"""tawjih - personal LLM command-directing (توجيه أوامر الآلة).

The package behind tawjih.py, in modules that each import only from the ones before
them: core, providers, engine, commands, library, manage, outputs, evaluate, governance, script, mcpserver, cli.
"""

from importlib import import_module as _import

MODULES = ("core", "providers", "engine", "commands", "library", "manage", "outputs", "evaluate", "governance", "script", "mcpserver", "cli")  # noqa: E501
for _name in MODULES:
    _m = _import("%s.%s" % (__name__, _name))
    globals().update({_k: getattr(_m, _k) for _k in _m.__all__})
