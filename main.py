from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn
from dotenv import load_dotenv

load_dotenv() # Carrega as variáveis do .env

from .models import EmailProcessResponse
from . import services

app = FastAPI(title="Email Classifier API")

# Configuração do CORS para permitir requisições do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Endereço do seu frontend React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-email/", response_model=EmailProcessResponse)
async def process_email(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    if not text and not file:
        raise HTTPException(status_code=400, detail="No text or file provided.")

    email_content = ""
    if file:
        email_content = await services.get_text_from_file(file)
    elif text:
        email_content = text

    if not email_content.strip():
        raise HTTPException(status_code=400, detail="Email content is empty.")

    category = services.classify_email(email_content)
    suggested_response = services.generate_response(email_content, category)

    return EmailProcessResponse(category=category, suggested_response=suggested_response)
