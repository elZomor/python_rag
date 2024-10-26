from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.RAGModel import RAGModel
from src.models.OLLamaLLMModel import OLLamaLLMModel

load_dotenv()

app = FastAPI()
llm_model = OLLamaLLMModel()
model = RAGModel(
    chunk_size=20, chunk_overlap=10, k=5, model=llm_model, data_path="../data"
)


class ContextRequest(BaseModel):
    data: str


@app.post("/context")
def add_context(request: ContextRequest):
    model.add_context(request.data)
    return {"status": "SUCCESS"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return {"error": "File type not supported. Please upload a PDF file."}

    file_location = f"data/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    model.load_data()
    return {"status": "SUCCESS"}


async def generate_text(return_model, prompt):
    for chunk in return_model.stream(prompt):
        yield chunk


class QuestionRequest(BaseModel):
    question: str


@app.post("/question")
def get_answer(request: QuestionRequest):
    generated_model, prompt = model.query_rag(request.question)
    return StreamingResponse(
        generate_text(generated_model, prompt), media_type="text/plain"
    )
