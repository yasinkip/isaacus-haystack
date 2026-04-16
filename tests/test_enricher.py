from __future__ import annotations

from types import SimpleNamespace

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.enrichers.isaacus import IsaacusEnricher

MODEL = "kanon-2-enricher"


# Unit tests
def test_initialization_kanon_2_enricher() -> None:
    """Test enriching model initialization."""
    rr = IsaacusEnricher(api_key=Secret.from_token("x"), model="kanon-2-enricher")
    assert rr.exclude == []
    assert rr.model == "kanon-2-enricher"
    assert rr.overflow_strategy == "auto"


def test_initialization_filters_unknown_exclude_values() -> None:
    rr = IsaacusEnricher(
        model="kanon-2-enricher",
        exclude=["persons", "invalid_kind"],
    )
    assert rr.exclude == ["persons"]
    assert rr.to_dict()["init_parameters"]["exclude"] == ["persons"]


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
    enricher = IsaacusEnricher(api_key=Secret.from_token("x"), model=MODEL)

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
    enricher = IsaacusEnricher(model=MODEL)
    output = enricher.run(documents)
    assert len(output["documents"]) == 1


def test_haystack_isaacus_enricher_documents_multiple() -> None:
    """Test Isaacus enrichers."""
    documents = [
        Document(content="foo bar"),
        Document(content="bar foo"),
        Document(content="foo"),
    ]
    enricher = IsaacusEnricher(model=MODEL)
    output = enricher.run(documents)
    assert len(output["documents"]) == 3


@pytest.mark.asyncio
async def test_haystack_isaacus_async_enricher_documents_multiple() -> None:
    """Test Isaacus enrichers."""
    documents = [
        Document(content="foo bar"),
        Document(content="bar foo"),
        Document(content="foo"),
    ]
    enricher = IsaacusEnricher(model=MODEL)
    output = await enricher.run_async(documents)
    assert len(output["documents"]) == 3


def test_haystack_isaacus_enrichments_has_all_enrichments() -> None:
    """Test Isaacus enrichers."""
    documents = [
        Document(content="foo bar"),
        Document(content="bar foo"),
        Document(content="foo"),
    ]
    enricher = IsaacusEnricher(model=MODEL)
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
    enricher = IsaacusEnricher(model=MODEL, exclude=exclude)
    output = enricher.run(documents)
    for doc in output["documents"]:
        assert not any(ex in doc.meta.keys() for ex in exclude)
        assert include.issubset(set(doc.meta.keys()))
    assert len(output["documents"]) == 3


@pytest.mark.asyncio
async def test_haystack_isaacus_enricher_with_excluded_enrichments_async() -> None:
    documents = [
        Document(content="foo bar"),
        Document(content="bar foo"),
        Document(content="foo"),
    ]
    exclude = ["persons", "locations", "emails"]
    include = set(IsaacusEnricher._ENRICHMENT_KINDS) - set(exclude)
    enricher = IsaacusEnricher(model=MODEL, exclude=exclude)
    output = await enricher.run_async(documents)
    for doc in output["documents"]:
        assert not any(ex in doc.meta.keys() for ex in exclude)
        assert include.issubset(set(doc.meta.keys()))
    assert len(output["documents"]) == 3
