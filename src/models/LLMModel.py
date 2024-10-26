from langchain_ollama import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.embeddings.cloudflare_workersai import (
    CloudflareWorkersAIEmbeddings,
)


class LLMModel:

    def get_model(self):
        return OllamaLLM(model=self.name, base_url=self.base_url)

    def get_embedding_function(self):
        # return OllamaEmbeddings(model=self.name, base_url=self.base_url)
        return CloudflareWorkersAIEmbeddings(
            account_id="46dbfc08c93ceeed07d0708cb6d62a4a",
            api_token="dkW9wWg0fqr3R-cnXY1snt10s7FMHv9vOABQalCl",
            model_name="@cf/baai/bge-small-en-v1.5",
        )
