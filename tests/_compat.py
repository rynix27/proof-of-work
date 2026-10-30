"""openai>=3 depends on `httpx2`; older openai releases (still allowed by
requirements.txt) depend on `httpx`. Tests that build HTTP request/response
objects use whichever the installed openai SDK uses."""
try:
    import httpx2 as httpx  # noqa: F401
except ImportError:  # pragma: no cover - depends on installed openai version
    import httpx  # noqa: F401
