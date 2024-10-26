# Python RAG (Retrieval-Augmented Generation)

This repository demonstrates a **Retrieval-Augmented Generation (RAG)** system using **Python** and **ChromaDB** for
managing embeddings and querying external data. The RAG model combines retrieval techniques with generative models to
provide accurate and context-aware responses by searching a knowledge base before generating text.

## Features

- **ChromaDB**: Vector database to store and manage document embeddings.
- **Cloudflare Workers AI / Ollama Integrations**: Supports large language models (LLMs) for generating text from
  queries.
- **Streaming Response**: Asynchronous streaming of model outputs.
- **FastAPI**: RESTful API to handle user queries.
- **Docker & Docker Compose**: Containerized deployment for easy setup.

---

## Setup Instructions

### 0. Pre-requisites

* Docker

### 1. Clone the Repository

```bash
git clone https://github.com/elZomor/python_rag.git
cd python_rag
```

### 2. Setup env variables

```bash
sudo cp .env.example .env
```

Then update the .env with your credentials

### 3. Run docker

```bash
docker compose up -d --build
```

## API Endpoints

### **POST /question**

- **Description**: Receives a query and returns a streamed response from the model.
- **Request** (JSON):
    ```json
    {
        "question": "What is Python?"
    }
    ```
- **Response** (Streamed Text):
    ```
    Python is a high-level programming language...
    ```

### **POST /upload**

- **Description**: Accepts PDF files and stores them in the `data/` directory for embedding.
- **Request**:
    - **Multipart/form-data**: Upload a PDF file using the `file` field.
- **Response**:
    ```json
    {
        "message": "File uploaded successfully"
    }
    ```

### **POST /context**

- **Description**: Receives text string and add it to the context.
- **Request** (JSON):
    ```json
    {
        "data": "elzomor is a software developer."
    }
    ```
- **Response** (JSON):
    ```json
    {
    "status": "SUCCESS"
    }
    ```

## Notes

* In the first build of the docker, the app will download LLama3.2 locally (About 2 GB)
* If you don't want to download it, you can remove the entry point line from the docker compose file

  Note that: You can not use the local LLM unless a model is downloaded
* The project is using LLMs for two things:
    * Embeddings
    * Responding with prompt
      They both can be done locally using OLLama or remote using Cloudflare
* To use CloudFlare you have to create api token on the cloudflare dashboard
* You can change the LLM model from the class file for both aproaches
* OLLama is using (llama3.2)
* Cloudflare is using (@cf/baai/bge-small-en-v1.5)
* To change the current model, in the main file line: 12