from __future__ import annotations

from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.enrichers.isaacus import IsaacusEnricher

MODEL = "kanon-2-enricher"


# Unit tests
def test_initialization_kanon_2_enricher() -> None:
    """Test enrichering model initialization."""
    rr = IsaacusEnricher(api_key=Secret.from_token("x"), model="kanon-2-enricher")
    assert rr.exclude == []
    assert rr.model == "kanon-2-enricher"
    assert rr.overflow_strategy == "auto"


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
