from dataclasses import replace
from typing import Any, Literal

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

from isaacus import AsyncIsaacus, Isaacus


@component
class IsaacusRanker:
    """
    Ranks documents based on their relevance to the query using [Isaacus rerankers](https://docs.isaacus.com/capabilities/reranking).

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.rankers.isaacus import IsaacusRanker

    ranker = IsaacusRanker(model="kanon-2-reranker")

    docs = [Document(content="The sky is blue."), Document(content="The sun is yellow.")]
    query = "What colour is the sky?"
    results = ranker.run(query=query, documents=docs)
    docs = results["documents"]
    print(docs[0].content) # The most relevant document.
    ```
    """

    def __init__(
        self,
        model: str,
        *,
        top_k: int | None = None,
        scoring_method: Literal["auto", "chunk_max", "chunk_avg", "chunk_min"] = "auto",
        api_key: Secret = Secret.from_env_var(["ISAACUS_API_KEY"]),
        api_base_url: str = "https://api.isaacus.com/v1",
    ) -> None:
        """Creates an instance of 'IsaacusRanker' for ordering documents based on their relevance to a query.

        Args:
            model (str): Isaacus model to use for reranking.

            top_k (int, optional): The number of documents to return.

            scoring_method (enum, optional): Scoring method to use for texts that exceed the model's local context window. Defaults to `auto`.

            api_key (Secret, optional): Isaacus API key.

            api_base_url (str, optional): Isaacus API base URL.
        """
        self.model = model
        self.top_k = top_k
        self.scoring_method = scoring_method
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
            top_k=self.top_k,
            scoring_method=self.scoring_method,
            api_key=self.api_key.to_dict(),
            api_base_url=self.api_base_url,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IsaacusRanker":
        """Deserializes the component from a dictionary."""

        init_params = data.get("init_parameters", {})
        if init_params:
            deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])

        return default_from_dict(cls, data)

    @staticmethod
    def _build_result(
        response: Any, documents: list[Document]
    ) -> dict[str, list[Document]]:
        """Builds the result given an API response and list of documents."""
        indices = [res.index for res in response.results]
        scores = [res.score for res in response.results]
        sorted_docs = []
        for idx, score in zip(indices, scores):
            doc = documents[idx]
            sorted_docs.append(replace(doc, score=score))
        return {"documents": sorted_docs}

    @component.output_types(documents=list[Document])
    def run(
        self, query: str, documents: list[Document], top_k: int | None = None
    ) -> dict[str, list[Document]]:
        """Use Isaacus Reranker to order the input documents according to their relevance to the query.

        Args:
            query (str): Query string.

            documents (list[Document]): The list of documents to rank.

            top_k (int, optional): The number of documents to return. Falls back to `self.top_k` if None.
        """
        top_k = top_k or self.top_k
        client = self._get_client()
        response = client.rerankings.create(
            model=self.model,
            query=query,
            top_n=top_k,
            scoring_method=self.scoring_method,
            texts=[doc.content for doc in documents],
        )
        return self._build_result(response, documents)

    @component.output_types(documents=list[Document])
    async def run_async(
        self, query: str, documents: list[Document], top_k: int | None = None
    ) -> dict[str, list[Document]]:
        """Use Isaacus Reranker to order the input documents according to their relevance to the query.

        Args:
            query (str): Query string.

            documents (list[Document]): The list of documents to rank.

            top_k (int, optional): The number of documents to return. Falls back to `self.top_k` if None.
        """
        top_k = top_k or self.top_k
        client = self._get_aclient()
        response = await client.rerankings.create(
            model=self.model,
            query=query,
            top_n=top_k,
            scoring_method=self.scoring_method,
            texts=[doc.content for doc in documents],
        )
        return self._build_result(response, documents)
