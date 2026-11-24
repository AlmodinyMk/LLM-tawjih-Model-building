# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report a
vulnerability privately through GitHub:
https://github.com/AlmodinyMk/LLM-tawjih-Model-building/security/advisories/new
(or the **Security** tab of this repository, then **Report a
vulnerability**). Describe what is affected, how to reproduce it, and the
version (`tawjih --version`). Only you and the maintainer see the report.

## What happens next

- You receive an answer within 14 days, saying whether the report is
  accepted as a vulnerability.
- An accepted vulnerability is fixed in a new version on GitHub and PyPI,
  aimed for within 90 days of the report.
- Disclosure is coordinated: the vulnerability is disclosed in a GitHub
  security advisory and the release notes once a fix is available, or after
  90 days if no fix is ready, unless we agree otherwise. You are credited
  in the advisory unless you ask not to be.

## Supported versions

Only the latest release receives fixes.

## Scope

Of particular interest: anything that lets a project file, a release or
pack file, a path, or a client of the MCP server reach files outside the
project, run code without `--allow-code`, bypass a lock or a person's
approval, or send data to a provider that was not chosen. Code steps run
with the user's permissions by design (see "Code steps" in the README); the
sandbox limits memory, time and file size but not the network.
