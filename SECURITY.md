# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.10.x  | Yes       |
| < 0.10  | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in embacle, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email **security@dravr.ai** with a description of the vulnerability
3. Include steps to reproduce, if possible
4. You will receive an acknowledgment within 48 hours

We will work with you to understand the issue and coordinate a fix before any public disclosure.

## Security Model

Embacle delegates LLM execution to CLI tools installed on the host. The security boundary is:

- **Subprocess isolation** — each request spawns a sandboxed child process with environment whitelisting and working directory control
- **No secrets in core** — the library stores no API keys or tokens; authentication is handled by the underlying CLI tools
- **Bearer auth** — `embacle-server` supports optional `EMBACLE_API_KEY` for request authentication
- **Input validation** — all user inputs are validated at system boundaries before passing to subprocesses
- **No shell expansion** — commands are built with explicit argument arrays, never shell strings
