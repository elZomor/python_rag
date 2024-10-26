from src.models.LLMModel import LLMModel
from langchain_community.llms.cloudflare_workersai import CloudflareWorkersAI
from langchain_community.embeddings.cloudflare_workersai import (
    CloudflareWorkersAIEmbeddings,
)
import os


class CloudFlareLLMModel(LLMModel):
    def __init__(self):
        super().__init__()

    def get_model(self):
        return CloudflareWorkersAI(
            account_id=os.getenv("CF_ACCOUNT_ID"),
            api_token=os.getenv("CF_API_TOKEN"),
            model_name="@cf/baai/bge-small-en-v1.5",
        )

    def get_embedding_function(self):
        return CloudflareWorkersAIEmbeddings(
            account_id=os.getenv("CF_ACCOUNT_ID"),
            api_token=os.getenv("CF_API_TOKEN"),
            model_name="@cf/baai/bge-small-en-v1.5",
        )
