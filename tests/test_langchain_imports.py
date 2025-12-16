"""Test that all langchain imports work with new versions."""
import pytest


def test_langchain_core_imports():
    """Test langchain-core imports."""
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.language_models import BaseChatModel
    assert AIMessage is not None
    assert BaseMessage is not None
    assert HumanMessage is not None
    assert SystemMessage is not None


def test_langgraph_imports():
    """Test langgraph imports."""
    from langgraph.graph import END, START, StateGraph
    from langgraph.checkpoint.memory import MemorySaver
    assert END is not None
    assert START is not None
    assert StateGraph is not None
    assert MemorySaver is not None


def test_langsmith_imports():
    """Test langsmith imports."""
    try:
        import langsmith
        assert langsmith is not None
    except ImportError:
        pytest.skip("LangSmith not installed or not configured")


def test_langchain_integration_imports():
    """Test langchain integration package imports."""
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_aws import ChatBedrock
    from langchain_community.vectorstores import MongoDBAtlasVectorSearch

    assert ChatOpenAI is not None
    assert ChatGoogleGenerativeAI is not None
    assert ChatBedrock is not None
    assert MongoDBAtlasVectorSearch is not None
