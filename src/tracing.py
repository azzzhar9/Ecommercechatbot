"""Langfuse tracing utilities for observability."""

import os
from typing import Optional, Dict, Any, Callable
from functools import wraps
from datetime import datetime

# Try to import Langfuse, provide fallback if not installed
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None

from dotenv import load_dotenv
load_dotenv()


class LangfuseTracer:
    """Langfuse tracer for observability and monitoring."""
    
    _instance: Optional['LangfuseTracer'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.enabled = False
        self.client = None
        
        # Check if Langfuse credentials are available
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        if LANGFUSE_AVAILABLE and secret_key and public_key:
            try:
                self.client = Langfuse(
                    secret_key=secret_key,
                    public_key=public_key,
                    host=host
                )
                self.enabled = True
                print("[Langfuse] Tracing enabled")
            except Exception as e:
                print(f"[Langfuse] Failed to initialize: {e}")
                self.enabled = False
        else:
            if not LANGFUSE_AVAILABLE:
                print("[Langfuse] Package not installed. Run: pip install langfuse")
            else:
                print("[Langfuse] Credentials not configured. Tracing disabled.")
        
        self._initialized = True
    
    def trace(
        self,
        name: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[list] = None
    ):
        """Create a new trace."""
        if not self.enabled or not self.client:
            return DummyTrace(name)
        
        try:
            trace = self.client.trace(
                name=name,
                session_id=session_id,
                user_id=user_id,
                metadata=metadata or {},
                tags=tags or []
            )
            return TraceWrapper(trace)
        except Exception as e:
            print(f"[Langfuse] Error creating trace: {e}")
            return DummyTrace(name)
    
    def span(
        self,
        trace,
        name: str,
        input: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """Create a span within a trace."""
        if not self.enabled or isinstance(trace, DummyTrace):
            return DummySpan(name)
        
        try:
            span = trace.trace.span(
                name=name,
                input=input,
                metadata=metadata or {}
            )
            return SpanWrapper(span)
        except Exception as e:
            print(f"[Langfuse] Error creating span: {e}")
            return DummySpan(name)
    
    def generation(
        self,
        trace,
        name: str,
        model: str,
        input: Any,
        output: Optional[Any] = None,
        usage: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """Log an LLM generation."""
        if not self.enabled or isinstance(trace, DummyTrace):
            return DummyGeneration(name)
        
        try:
            gen = trace.trace.generation(
                name=name,
                model=model,
                input=input,
                output=output,
                usage=usage,
                metadata=metadata or {}
            )
            return GenerationWrapper(gen)
        except Exception as e:
            print(f"[Langfuse] Error creating generation: {e}")
            return DummyGeneration(name)
    
    def flush(self):
        """Flush all pending events."""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:
                print(f"[Langfuse] Error flushing: {e}")


class TraceWrapper:
    """Wrapper for Langfuse trace."""
    def __init__(self, trace):
        self.trace = trace
    
    def end(self, output: Optional[Any] = None):
        try:
            if output:
                self.trace.update(output=output)
        except:
            pass


class SpanWrapper:
    """Wrapper for Langfuse span."""
    def __init__(self, span):
        self.span = span
    
    def end(self, output: Optional[Any] = None):
        try:
            self.span.end(output=output)
        except:
            pass


class GenerationWrapper:
    """Wrapper for Langfuse generation."""
    def __init__(self, generation):
        self.generation = generation
    
    def end(self, output: Optional[Any] = None, usage: Optional[Dict] = None):
        try:
            self.generation.end(output=output, usage=usage)
        except:
            pass


class DummyTrace:
    """Dummy trace when Langfuse is disabled."""
    def __init__(self, name: str):
        self.name = name
    
    def end(self, output=None):
        pass


class DummySpan:
    """Dummy span when Langfuse is disabled."""
    def __init__(self, name: str):
        self.name = name
    
    def end(self, output=None):
        pass


class DummyGeneration:
    """Dummy generation when Langfuse is disabled."""
    def __init__(self, name: str):
        self.name = name
    
    def end(self, output=None, usage=None):
        pass


def traced(name: Optional[str] = None):
    """Decorator for tracing functions."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            trace_name = name or func.__name__
            
            # Get session_id from kwargs or first arg if it's a chatbot
            session_id = kwargs.get('session_id')
            if not session_id and args and hasattr(args[0], 'session_id'):
                session_id = args[0].session_id
            
            trace = tracer.trace(
                name=trace_name,
                session_id=session_id,
                metadata={"function": func.__name__}
            )
            
            try:
                result = func(*args, **kwargs)
                trace.end(output={"result": str(result)[:500] if result else None})
                return result
            except Exception as e:
                trace.end(output={"error": str(e)})
                raise
            finally:
                tracer.flush()
        
        return wrapper
    return decorator


# Singleton accessor
_tracer: Optional[LangfuseTracer] = None

def get_tracer() -> LangfuseTracer:
    """Get the singleton Langfuse tracer."""
    global _tracer
    if _tracer is None:
        _tracer = LangfuseTracer()
    return _tracer
