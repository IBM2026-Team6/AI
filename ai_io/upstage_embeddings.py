from typing import List
from langchain_core.embeddings import Embeddings
from openai import OpenAI


class UpstageEmbeddings(Embeddings):
    """Custom Embeddings wrapper for Upstage OpenAI-compatible API.

    Uses the official OpenAI client with Upstage base_url to avoid
    invalid input issues observed with langchain_openai.
    """

    def __init__(self, api_key: str, model: str = "embedding-passage", base_url: str = "https://api.upstage.ai/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = [str(t or "") for t in texts]
        resp = self.client.embeddings.create(model=self.model, input=inputs)
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> List[float]:
        input_text = str(text or "")
        resp = self.client.embeddings.create(model=self.model, input=input_text)
        return resp.data[0].embedding
