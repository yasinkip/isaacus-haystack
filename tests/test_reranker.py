from __future__ import annotations

from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.rankers.isaacus import IsaacusRanker

MODEL = "kanon-2-reranker"


# Unit tests
def test_initialization_kanon_2_reranker() -> None:
    """Test reranking model initialization."""
    rr = IsaacusRanker(api_key=Secret.from_token("x"), model="kanon-2-reranker")
    assert rr.top_k is None
    assert rr.model == "kanon-2-reranker"
    assert rr.scoring_method == "auto"


# Integration tests
def test_haystack_isaacus_rerank_documents() -> None:
    """Test Isaacus Reranks."""
    query = "foo"
    documents = [Document(content="foo bar")]
    rerank = IsaacusRanker(model=MODEL)
    output = rerank.run(query, documents)
    assert len(output["documents"]) == 1


def test_haystack_isaacus_rerank_documents_multiple() -> None:
    """Test Isaacus Reranks."""
    query = "foo"
    documents = [
        Document(content="foo bar"),
        Document(content="bar foo"),
        Document(content="foo"),
    ]
    rerank = IsaacusRanker(model=MODEL)
    output = rerank.run(query, documents)
    assert len(output["documents"]) == 3


async def test_haystack_isaacus_async_rerank_documents_multiple() -> None:
    """Test Isaacus Reranks."""
    query = "foo"
    documents = [
        Document(content="foo bar"),
        Document(content="bar foo"),
        Document(content="foo"),
    ]
    rerank = IsaacusRanker(model=MODEL)
    output = await rerank.run_async(query, documents)
    assert len(output["documents"]) == 3


def test_haystack_isaacus_rerank_documents_with_top_k() -> None:
    """Test Isaacus Reranks."""
    query = "foo"
    documents = [
        Document(content="foo bar"),
        Document(content="bar foo"),
        Document(content="foo"),
    ]
    rerank = IsaacusRanker(model=MODEL, top_k=2)
    output = rerank.run(query, documents)
    assert len(output["documents"]) == 2


def test_haystack_isaacus_rerank_with_overridden_top_k() -> None:
    query = "foo"
    documents = [
        Document(content="foo bar"),
        Document(content="bar foo"),
        Document(content="foo"),
    ]
    rerank = IsaacusRanker(model=MODEL, top_k=3)
    output = rerank.run(query, documents, top_k=2)
    assert len(output["documents"]) == 2


async def test_haystack_isaacus_rerank_with_overridden_top_k_async() -> None:
    query = "foo"
    documents = [
        Document(content="foo bar"),
        Document(content="bar foo"),
        Document(content="foo"),
    ]
    rerank = IsaacusRanker(model=MODEL, top_k=3)
    output = await rerank.run_async(query, documents, top_k=2)
    assert len(output["documents"]) == 2
