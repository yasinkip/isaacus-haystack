[![PyPI - Version](https://img.shields.io/pypi/v/isaacus-haystack.svg)](https://pypi.org/project/isaacus-haystack)
## Overview
[Isaacus](https://isaacus.com/) is a foundational legal AI research company building AI models, apps, and tools for the legal tech ecosystem.

Isaacus' offering includes:
- [Kanon 2 Embedder](https://isaacus.com/blog/introducing-kanon-2-embedder), the world's best legal embedding model (as measured on the [Massive Legal Embedding Benchmark](https://isaacus.com/blog/introducing-mleb)),
- [Kanon 2 Reranker](https://isaacus.com/blog/kanon-2-reranker), the most powerful reranking model for legal RAG,
- [Kanon 2 Enricher](https://isaacus.com/blog/kanon-2-enricher), the world's first hierarchical graphitization model,
- [legal zero-shot classification](https://docs.isaacus.com/models/introduction#universal-classification) and [legal extractive question answering models](https://docs.isaacus.com/models/introduction#answer-extraction).

Isaacus offers first-class support for Haystack through the `isaacus-haystack` integration package.

## Installation
```bash
pip install isaacus-haystack
```

## Components
- `IsaacusTextEmbedder` – embeds query text into a vector.
- `IsaacusDocumentEmbedder` – embeds Haystack `Document`s and writes to `document.embedding`.
- `IsaacusRanker` – orders Haystack `Document`s by relevance to a query and writes scores to `document.score`.
- `IsaacusEnricher` – enriches the metadata of `Document`s and writes to `document.meta`.

## Examples

### Using the Isaacus embedders and rerankers
Below is how you would execute a quick retrieval-augmented generation (RAG) flow in Haystack using the
`isaacus-haystack` package.

```python
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.utils import Secret
from haystack_integrations.components.embedders.isaacus import IsaacusTextEmbedder, IsaacusDocumentEmbedder
from haystack_integrations.components.rankers.isaacus import IsaacusRanker

store = InMemoryDocumentStore(embedding_similarity_function="dot_product")
embedder = IsaacusDocumentEmbedder(
    api_key=Secret.from_env_var("ISAACUS_API_KEY"),
    model="kanon-2-embedder",          # choose any supported Isaacus embedding model.
    # dimensions=1792,                 # optionally set to match your vector DB.
)
ranker = IsaacusRanker(
    model="kanon-2-reranker",
    # top_k=1,                       # optionally specify the number of documents you want returned.
) 

raw_docs = [Document(content="Isaacus releases Kanon 2 Embedder: the world's best legal embedding model."),
            Document(content="Isaacus also offers legal zero-shot classification and extractive question answering models.")]
store.write_documents(embedder.run(raw_docs)["documents"])

pipe = Pipeline()
pipe.add_component("q", IsaacusTextEmbedder(api_key=Secret.from_env_var("ISAACUS_API_KEY"), model="kanon-2-embedder"))
pipe.add_component("ret", InMemoryEmbeddingRetriever(document_store=store))
pipe.add_component("rank", IsaacusRanker(model="kanon-2-reranker"))
pipe.connect("q.embedding", "ret.query_embedding")
pipe.connect("ret.documents", "rank.documents")

query = "What is the best legal embedding model?"
print(pipe.run({"q": {"text": query}, "rank": {"query": query}}))
```

### Using the Isaacus enrichers
Here we demonstrate how you would run the enricher.

```python
from haystack import Document
from haystack_integrations.components.enrichers.isaacus import IsaacusEnricher

enricher = IsaacusEnricher(model="kanon-2-enricher")

docs = [Document(content="Elon Musk founded SpaceX in 2002 in Hawthorne, California. You can reach them at info@spacex.com or visit spacex.com.")]

result = enricher.run(documents=docs)
doc = result["documents"][0]

print("Persons:", doc.meta.get("persons"))
print("Locations:", doc.meta.get("locations"))
print("Emails:", doc.meta.get("emails"))
print("Websites:", doc.meta.get("websites"))
print("Dates:", doc.meta.get("dates"))
# ...
```

## Docs
- Isaacus Embedding docs: https://docs.isaacus.com/capabilities/embedding
- Isaacus Reranking docs: https://docs.isaacus.com/capabilities/reranking
- Isaacus Enrichment docs: https://docs.isaacus.com/capabilities/enrichment
- Haystack: https://haystack.deepset.ai/

## License
Apache-2.0