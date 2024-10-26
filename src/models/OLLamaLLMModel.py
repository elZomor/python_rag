from src.models.LLMModel import LLMModel
from langchain_ollama import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings


class OLLamaLLMModel(LLMModel):
    def __init__(
        self, name: str = 'llama3.2', base_url: str = 'host.docker.internal:11400'
    ):
        super().__init__()
        self.name = name
        self.base_url = base_url

    def get_model(self):
        return OllamaLLM(model=self.name, base_url=self.base_url)

    def get_embedding_function(self):
        return OllamaEmbeddings(model=self.name, base_url=self.base_url)
