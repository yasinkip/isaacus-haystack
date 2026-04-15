import logging
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

from isaacus import AsyncIsaacus, Isaacus

logger = logging.getLogger(__name__)


@component
class IsaacusEnricher:
    """
    Enriches documents by converting them into structured knowledge graphs using [Isaacus enrichment models](https://docs.isaacus.com/capabilities/enrichment).
    All enrichments are stored in the documents metadata.

    Usage example:
    ```python
        from haystack import Document
        from haystack_integrations.components.enrichers.isaacus import IsaacusEnricher

        enricher = IsaacusEnricher(model="kanon-2-enricher")

        docs = [Document(content="Apple Inc. was founded on April 1, 1976 in Cupertino, California.")]
        results = enricher.run(documents=docs)
        docs = results["documents"]
        print(docs[0].meta["persons"])  # Persons identified in the document.
    ```
    """

    _ENRICHMENT_KINDS = [
        "crossreferences",
        "locations",
        "persons",
        "emails",
        "websites",
        "phone_numbers",
        "id_numbers",
        "terms",
        "external_documents",
        "quotes",
        "dates",
    ]

    def __init__(
        self,
        model: str,
        *,
        exclude: list[str] | None = None,
        overflow_strategy: str = "auto",
        api_key: Secret = Secret.from_env_var(["ISAACUS_API_KEY"]),
        api_base_url: str = "https://api.isaacus.com/v1",
    ) -> None:
        """Creates an instance of 'IsaacusEnricher' for enriching documents into structured knowledge graphs.

        Args:
            model (str): Isaacus model to use for enrichment.

            exclude (list[str], optional): List of [enrichment kinds](https://docs.isaacus.com/capabilities/enrichment#3-rendering) to exclude.

            overflow_strategy (str, optional): Strategy to use when a document exceeds the model's local context window. Defaults to `auto`.

            api_key (Secret, optional): Isaacus API key.

            api_base_url (str, optional): Isaacus API base URL.
        """
        self.model = model
        self.overflow_strategy = overflow_strategy
        self.exclude = exclude if exclude else []
        self._include = [
            kind for kind in self._ENRICHMENT_KINDS if kind not in self.exclude
        ]

        unknown = [kind for kind in self.exclude if kind not in self._ENRICHMENT_KINDS]
        if unknown:
            logger.warning(
                "The following exclude values are not valid enrichment kinds and will be ignored: %s. "
                "Valid kinds are: %s",
                unknown,
                self._ENRICHMENT_KINDS,
            )
        self.api_key = api_key
        self.api_base_url = api_base_url

        self._client = None
        self._aclient = None

    def _get_client(self):
        if not self._client:
            self._client = Isaacus(
                api_key=self.api_key.resolve_value(), base_url=self.api_base_url
            )

        return self._client

    def _get_aclient(self):
        if not self._aclient:
            self._aclient = AsyncIsaacus(
                api_key=self.api_key.resolve_value(), base_url=self.api_base_url
            )

        return self._aclient

    def to_dict(self) -> dict[str, Any]:
        """Serializes the component to a dictionary."""
        return default_to_dict(
            self,
            model=self.model,
            exclude=self.exclude,
            overflow_strategy=self.overflow_strategy,
            api_key=self.api_key.to_dict(),
            api_base_url=self.api_base_url,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IsaacusEnricher":
        """Deserializes the component from a dictionary."""

        init_params = data.get("init_parameters", {})
        if init_params:
            deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])

        return default_from_dict(cls, data)

    def _build_result(
        self, response: Any, documents: list[Document]
    ) -> dict[str, list[Document]]:
        """Builds the result given an API response and list of documents."""
        enriched_docs = []

        for res in response.results:
            doc = documents[res.index]
            result = res.document
            new_enrichments = {kind: getattr(result, kind) for kind in self._include}
            meta = {**doc.meta, **new_enrichments}
            enriched_docs.append(Document(content=doc.content, meta=meta))

        return {"documents": enriched_docs}

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """Use Isaacus Enricher to enrich the input documents into structured knowledge graphs.

        Args:
            documents (list[Document]): The list of documents to enrich.
        """
        if not documents:
            return {"documents": []}

        client = self._get_client()
        response = client.enrichments.create(
            model=self.model,
            texts=[doc.content for doc in documents],
            overflow_strategy=self.overflow_strategy,
        )
        return self._build_result(response, documents)

    @component.output_types(documents=list[Document])
    async def run_async(self, documents: list[Document]) -> dict[str, list[Document]]:
        """Use Isaacus Enricher to enrich the input documents into structured knowledge graphs.

        Args:
            documents (list[Document]): The list of documents to enrich.
        """
        if not documents:
            return {"documents": []}

        client = self._get_aclient()
        response = await client.enrichments.create(
            model=self.model,
            texts=[doc.content for doc in documents],
            overflow_strategy=self.overflow_strategy,
        )
        return self._build_result(response, documents)
