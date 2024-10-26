from langchain.prompts import ChatPromptTemplate

from src.models.LLMModel import LLMModel
from src.utils import (
    load_documents,
    split_docs_to_chunks,
    add_text_to_chroma,
    split_string_to_chunks,
    get_chroma_db,
    add_document_to_chroma,
)


class RAGModel:
    def __init__(
        self,
        k: int = 3,
        chunk_size: int = 100,
        chunk_overlap: int = 100,
        model: LLMModel = LLMModel(),
        data_path: str = "./data",
        prompt_template: str = """ 
        Answer the question based only on the following context and be very focused
        and always use bullet points:

        {context}

        ---

        Answer the question based on the above context: {question}
        """,
    ):
        self.k = k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model
        self.data_path = data_path
        self.prompt_template = prompt_template
        self.load_data()

    def load_data(self):
        documents = load_documents(self.data_path)
        chunk_dict_with_ids = split_docs_to_chunks(
            docs=documents, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        add_document_to_chroma(
            chunk_doc_id_dict=chunk_dict_with_ids,
            embedding_function=self.model.get_embedding_function(),
        )

    def add_context(self, text: str):
        chunk_id_dict = split_string_to_chunks(
            text=text, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        add_text_to_chroma(
            chunk_text_id_dict=chunk_id_dict,
            embedding_function=self.model.get_embedding_function(),
        )

    def query_rag(self, query_text: str):
        db = get_chroma_db(embedding_function=self.model.get_embedding_function())

        results = db.similarity_search_with_score(query_text, k=self.k)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
        prompt = prompt_template.format(context=context_text, question=query_text)

        return self.model.get_model(), prompt
