import os

# llm_judge raises at import if the key is missing; tests never make real API calls.
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")
