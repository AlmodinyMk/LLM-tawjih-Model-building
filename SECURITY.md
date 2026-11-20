# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's
private reporting instead: the **Security** tab of this repository, then
**Report a vulnerability**. Describe what is affected, how to reproduce it,
and the version (`tawjih --version`).

You should receive an answer within two weeks. Fixes are released as a new
version on GitHub and PyPI, with the problem described in the release notes
once a fix is available.

## Supported versions

Only the latest release receives fixes.

## Scope

Of particular interest: anything that lets a project file, a release or
pack file, a path, or a client of the MCP server reach files outside the
project, run code without `--allow-code`, bypass a lock or a person's
approval, or send data to a provider that was not chosen. Code steps run
with the user's permissions by design (see "Code steps" in the README); the
sandbox limits memory, time and file size but not the network.
