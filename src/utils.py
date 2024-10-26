import hashlib
from typing import List, Dict
from langchain.schema.document import Document

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma


def load_documents(data_path: str) -> List[Document]:
    return PyPDFDirectoryLoader(data_path).load()


def split_docs_to_chunks(
    chunk_size: int,
    chunk_overlap: int,
    docs: List[Document] = None,
    is_separator_regex: bool = True,
    separators: List[str] = (r"\n\n+", r"\.\s+", r"\nChapter \d+"),
) -> Dict[str, Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=is_separator_regex,
        separators=separators,
    )
    chunks = text_splitter.split_documents(docs)
    return calculate_docs_chunk_ids(chunks)


def calculate_docs_chunk_ids(chunks) -> Dict[str, Document]:
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0
    chunk_id_dict = {}

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk_id_dict[chunk_id] = chunk
    return chunk_id_dict


def split_string_to_chunks(
    chunk_size: int,
    chunk_overlap: int,
    text: str,
    is_separator_regex: bool = True,
    separators: List[str] = (r"\n\n+", r"\.\s+", r"\nChapter \d+"),
) -> Dict[str, str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=is_separator_regex,
        separators=separators,
    )
    chunk_list = text_splitter.split_text(text)
    return add_string_id_list(chunk_list)


def add_string_id_list(text_list: List[str]) -> Dict[str, str]:
    chunk_id_dict = {}
    for text in text_list:
        chunk_id_dict[hashlib.sha256(text.encode()).hexdigest()] = text
    return chunk_id_dict


def add_text_to_chroma(
    chunk_text_id_dict: Dict[str, str], embedding_function: Embeddings
):
    db = get_chroma_db(embedding_function=embedding_function)

    existing_ids = set(db.get(include=[])["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    new_chunk_ids_dict = {}
    for chunk_id, chunk in chunk_text_id_dict.items():
        if chunk_id not in existing_ids:
            new_chunk_ids_dict[chunk_id] = chunk
    if len(new_chunk_ids_dict):
        db.add_texts(
            list(new_chunk_ids_dict.values()), ids=list(new_chunk_ids_dict.keys())
        )


def add_document_to_chroma(
    embedding_function: Embeddings,
    chunk_doc_id_dict: Dict[str, Document],
):
    db = get_chroma_db(embedding_function=embedding_function)

    existing_ids = set(db.get(include=[])["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    new_chunk_doc_id_dict = {}
    for chunk_id, chunk in chunk_doc_id_dict.items():
        if chunk_id not in existing_ids:
            new_chunk_doc_id_dict[chunk_id] = chunk
    if len(new_chunk_doc_id_dict):
        print(f"👉 Adding new documents: {len(chunk_doc_id_dict)}")
        db.add_documents(
            list(new_chunk_doc_id_dict.values()), ids=list(new_chunk_doc_id_dict.keys())
        )
    else:
        print("✅ No new documents to add")


def get_chroma_db(
    embedding_function: Embeddings,
    collection_name: str = "chroma",
    persist_directory: str = "chroma",
):
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_function,
    )
