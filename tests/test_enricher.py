"""Tests for the Isaacus Enricher component. Run `pytest --asyncio-mode=auto` to execute all tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.enrichers.isaacus import IsaacusEnricher

MODEL = "kanon-2-enricher"


def _make_enricher(**kwargs) -> IsaacusEnricher:
    kwargs.setdefault("api_key", Secret.from_token("x"))
    kwargs.setdefault("model", MODEL)
    return IsaacusEnricher(**kwargs)


def _build_response(documents: list[Document]) -> SimpleNamespace:
    doc = SimpleNamespace(
        **{kind: [kind] for kind in IsaacusEnricher._ENRICHMENT_KINDS}
    )
    results = [
        SimpleNamespace(index=index, document=doc) for index, _ in enumerate(documents)
    ]
    return SimpleNamespace(results=results)


def _attach_sync_client(enricher: IsaacusEnricher, documents: list[Document]) -> Mock:
    create = Mock(return_value=_build_response(documents))
    enricher._client = SimpleNamespace(enrichments=SimpleNamespace(create=create))
    return create


def _attach_async_client(
    enricher: IsaacusEnricher, documents: list[Document]
) -> AsyncMock:
    create = AsyncMock(return_value=_build_response(documents))
    enricher._aclient = SimpleNamespace(enrichments=SimpleNamespace(create=create))
    return create


# Unit tests
def test_initialization_kanon_2_enricher() -> None:
    """Test enriching model initialization."""
    rr = _make_enricher()
    assert rr.exclude == []
    assert rr.model == "kanon-2-enricher"
    assert rr.overflow_strategy == "auto"


def test_initialization_filters_unknown_exclude_values() -> None:
    rr = _make_enricher(exclude=["persons", "invalid_kind"])
    assert rr.exclude == ["persons"]


def test_build_result_preserves_document_fields() -> None:
    doc = Document(
        content="foo bar",
        id="doc-1",
        score=0.42,
        embedding=[1.0, 2.0],
        meta={"source": "test"},
    )

    result_doc = SimpleNamespace(
        **{kind: [kind] for kind in IsaacusEnricher._ENRICHMENT_KINDS}
    )
    response = SimpleNamespace(
        results=[SimpleNamespace(index=0, document=result_doc)],
    )
    enricher = _make_enricher()

    output = enricher._build_result(response, [doc])["documents"][0]

    assert output.id == doc.id
    assert output.score == doc.score
    assert output.embedding == doc.embedding
    assert output.meta["source"] == "test"
    assert "persons" in output.meta


# Integration tests
def test_haystack_isaacus_enricher_single_document() -> None:
    """Test Isaacus Enrichers."""
    documents = [Document(content="foo bar")]
    enricher = _make_enricher()
    create = _attach_sync_client(enricher, documents)
    output = enricher.run(documents)
    create.assert_called_once_with(
        model=MODEL,
        texts=["foo bar"],
        overflow_strategy="auto",
    )
    assert len(output["documents"]) == 1


def test_haystack_isaacus_enricher_documents_multiple() -> None:
    """Test Isaacus enrichers."""
    documents = [
        Document(content="foo bar"),
        Document(content="bar foo"),
        Document(content="foo"),
    ]
    enricher = _make_enricher()
    create = _attach_sync_client(enricher, documents)
    output = enricher.run(documents)
    create.assert_called_once_with(
        model=MODEL,
        texts=["foo bar", "bar foo", "foo"],
        overflow_strategy="auto",
    )
    assert len(output["documents"]) == 3


async def test_haystack_isaacus_async_enricher_documents_multiple() -> None:
    """Test Isaacus enrichers."""
    documents = [
        Document(content="foo bar"),
        Document(content="bar foo"),
        Document(content="foo"),
    ]
    enricher = _make_enricher()
    create = _attach_async_client(enricher, documents)
    output = await enricher.run_async(documents)
    create.assert_awaited_once_with(
        model=MODEL,
        texts=["foo bar", "bar foo", "foo"],
        overflow_strategy="auto",
    )
    assert len(output["documents"]) == 3


def test_haystack_isaacus_enrichments_has_all_enrichments() -> None:
    """Test Isaacus enrichers."""
    documents = [
        Document(content="foo bar"),
        Document(content="bar foo"),
        Document(content="foo"),
    ]
    enricher = _make_enricher()
    _attach_sync_client(enricher, documents)
    output = enricher.run(documents)
    for doc in output["documents"]:
        assert set(IsaacusEnricher._ENRICHMENT_KINDS).issubset(set(doc.meta.keys()))
    assert len(output["documents"]) == 3


def test_haystack_isaacus_enricher_with_excluded_enrichments() -> None:
    documents = [
        Document(content="foo bar"),
        Document(content="bar foo"),
        Document(content="foo"),
    ]
    exclude = ["persons", "locations", "emails"]
    include = set(IsaacusEnricher._ENRICHMENT_KINDS) - set(exclude)
    enricher = _make_enricher(exclude=exclude)
    _attach_sync_client(enricher, documents)
    output = enricher.run(documents)
    for doc in output["documents"]:
        assert not any(ex in doc.meta.keys() for ex in exclude)
        assert include.issubset(set(doc.meta.keys()))
    assert len(output["documents"]) == 3


async def test_haystack_isaacus_enricher_with_excluded_enrichments_async() -> None:
    documents = [
        Document(content="foo bar"),
        Document(content="bar foo"),
        Document(content="foo"),
    ]
    exclude = ["persons", "locations", "emails"]
    include = set(IsaacusEnricher._ENRICHMENT_KINDS) - set(exclude)
    enricher = _make_enricher(exclude=exclude)
    _attach_async_client(enricher, documents)
    output = await enricher.run_async(documents)
    for doc in output["documents"]:
        assert not any(ex in doc.meta.keys() for ex in exclude)
        assert include.issubset(set(doc.meta.keys()))
    assert len(output["documents"]) == 3
