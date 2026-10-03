import litellm
import os
import time
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

_original_completion = litellm.completion


def _retry_delay(e):
    """Seconds to wait if this error is worth retrying, else None."""
    if isinstance(e, (litellm.RateLimitError, litellm.ServiceUnavailableError)):
        return 20
    # Groq returns a 400 when the model's tool-call output degenerates; a retry usually fixes it
    if isinstance(e, litellm.BadRequestError) and "tool_use_failed" in str(e):
        return 3
    return None


def _patched_completion(*args, **kwargs):
    if "messages" in kwargs:
        for m in kwargs["messages"]:
            if isinstance(m, dict):
                m.pop("cache_breakpoint", None)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            return _original_completion(*args, **kwargs)
        except Exception as e:
            base = _retry_delay(e)
            if base is None or attempt == max_retries - 1:
                raise
            wait_time = base * (attempt + 1)
            print(f"[Groq issue ({type(e).__name__}) - waiting {wait_time}s before retry {attempt + 1}/{max_retries}]")
            time.sleep(wait_time)


litellm.completion = _patched_completion


def get_llm():
    return LLM(
        model="groq/qwen/qwen3.8-27b",
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=2000,
    )