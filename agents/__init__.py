# HR Resume Shortlisting Agent - Agents
import os

def get_langfuse_callbacks():
    """Returns Langfuse CallbackHandler if configured, else empty list."""
    callbacks = []
    if os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"):
        try:
            from langfuse.callback import CallbackHandler  # type: ignore
            langfuse_handler = CallbackHandler(
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            )
            callbacks.append(langfuse_handler)
        except ImportError:
            pass
    return callbacks
