"""Tests for the Isaacus Reranker component. Run `pytest --asyncio-mode=auto` to execute all tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.rankers.isaacus import IsaacusRanker

MODEL = "kanon-2-reranker"


def _make_reranker(**kwargs) -> IsaacusRanker:
    kwargs.setdefault("api_key", Secret.from_token("x"))
    kwargs.setdefault("model", MODEL)
    return IsaacusRanker(**kwargs)


def _build_response(top_k: int) -> SimpleNamespace:
    return SimpleNamespace(
        results=[SimpleNamespace(index=idx, score=0.5) for idx in range(top_k)]
    )


def _fake_create(*args, **kwargs):
    top_n = kwargs.get("top_n") or len(kwargs.get("texts"))
    return _build_response(top_n)


def _attach_sync_client(reranker: IsaacusRanker) -> Mock:
    create = Mock(side_effect=_fake_create)
    reranker._client = SimpleNamespace(rerankings=SimpleNamespace(create=create))
    return create


def _attach_async_client(reranker: IsaacusRanker) -> AsyncMock:
    create = AsyncMock(side_effect=_fake_create)
    reranker._aclient = SimpleNamespace(rerankings=SimpleNamespace(create=create))
    return create


# Unit tests
def test_initialization_kanon_2_reranker() -> None:
    """Test reranking model initialization."""
    rr = _make_reranker()
    assert rr.top_k is None
    assert rr.model == "kanon-2-reranker"
    assert rr.scoring_method == "auto"


# Integration tests
def test_haystack_isaacus_rerank_documents() -> None:
    """Test Isaacus Reranks."""
    query = "foo"
    documents = [Document(content="foo bar")]
    rerank = _make_reranker()
    create = _attach_sync_client(rerank)
    output = rerank.run(query, documents)
    create.assert_called_once_with(
        model=MODEL,
        query=query,
        texts=[doc.content for doc in documents],
        top_n=None,
        scoring_method="auto",
    )
    assert len(output["documents"]) == 1


def test_haystack_isaacus_rerank_documents_multiple() -> None:
    """Test Isaacus Reranks."""
    query = "foo"
    documents = [
        Document(content="foo bar"),
        Document(content="bar foo"),
        Document(content="foo"),
    ]
    rerank = _make_reranker()
    create = _attach_sync_client(rerank)
    output = rerank.run(query, documents)
    create.assert_called_once_with(
        model=MODEL,
        query=query,
        texts=[doc.content for doc in documents],
        top_n=None,
        scoring_method="auto",
    )
    assert len(output["documents"]) == 3


async def test_haystack_isaacus_async_rerank_documents_multiple() -> None:
    """Test Isaacus Reranks."""
    query = "foo"
    documents = [
        Document(content="foo bar"),
        Document(content="bar foo"),
        Document(content="foo"),
    ]
    rerank = _make_reranker()
    create = _attach_async_client(rerank)
    output = await rerank.run_async(query, documents)
    create.assert_awaited_once_with(
        model=MODEL,
        query=query,
        texts=[doc.content for doc in documents],
        top_n=None,
        scoring_method="auto",
    )
    assert len(output["documents"]) == 3


def test_haystack_isaacus_rerank_documents_with_top_k() -> None:
    """Test Isaacus Reranks."""
    query = "foo"
    documents = [
        Document(content="foo bar"),
        Document(content="bar foo"),
        Document(content="foo"),
    ]
    rerank = _make_reranker(top_k=2)
    create = _attach_sync_client(rerank)
    output = rerank.run(query, documents)
    create.assert_called_once_with(
        model=MODEL,
        query=query,
        texts=[doc.content for doc in documents],
        top_n=2,
        scoring_method="auto",
    )
    assert len(output["documents"]) == 2


def test_haystack_isaacus_rerank_with_overridden_top_k() -> None:
    query = "foo"
    documents = [
        Document(content="foo bar"),
        Document(content="bar foo"),
        Document(content="foo"),
    ]
    rerank = _make_reranker(top_k=3)
    create = _attach_sync_client(rerank)
    output = rerank.run(query, documents, top_k=2)
    create.assert_called_once_with(
        model=MODEL,
        query=query,
        texts=[doc.content for doc in documents],
        top_n=2,
        scoring_method="auto",
    )
    assert len(output["documents"]) == 2


async def test_haystack_isaacus_rerank_with_overridden_top_k_async() -> None:
    query = "foo"
    documents = [
        Document(content="foo bar"),
        Document(content="bar foo"),
        Document(content="foo"),
    ]
    rerank = _make_reranker(top_k=3)
    create = _attach_async_client(rerank)
    output = await rerank.run_async(query, documents, top_k=2)
    create.assert_awaited_once_with(
        model=MODEL,
        query=query,
        texts=[doc.content for doc in documents],
        top_n=2,
        scoring_method="auto",
    )
    assert len(output["documents"]) == 2
