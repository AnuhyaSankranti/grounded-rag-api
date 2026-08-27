# Security

- Keep `OPENAI_API_KEY` in a secret manager or environment variable.
- Treat retrieved documents as untrusted data. Mini SIA wraps context as data and
  instructs the model not to follow instructions found inside it.
- Add authentication, tenant-aware authorization, per-user quotas, malware scanning,
  and encrypted storage before exposing document upload on the public internet.
- Report vulnerabilities privately to the repository owner rather than opening a
  public issue.

