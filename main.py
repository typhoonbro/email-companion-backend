from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import asyncio
import os

# Importa os modelos e serviços
from models import EmailProcessResponse
from services import classify_email, generate_response, extract_text_from_pdf

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

app = FastAPI()

# Configuração do CORS para permitir requisições do frontend
origins = [
    "http://localhost:5173", # Porta padrão do Vite pra desenvolvimento
    "http://localhost:4173", # Porta padrão do Vite pra testar a versão de produção localmente
    "https://teste-auto-u-frontend-ytv8.vercel.app" # URL do frontend hospedado no Vercel TROCAR PARA SUA URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    
)

@app.post("/process-email/", response_model=EmailProcessResponse)
async def process_email_endpoint(
    email_text: str = Form(""),
    file: UploadFile | None = File(None)
):
    """
    Processa um email (texto ou PDF), classifica-o e gera uma resposta.
    """
    final_text = email_text
    file_content = ""

    
    # Se houver um arquivo PDF, extrai o texto e o adiciona ao texto do email
    if file:
        if file.content_type == 'application/pdf':
            pdf_content = await file.read()
            file_content = "\n\nTexto contido no PDF anexado:" + extract_text_from_pdf(pdf_content)
        else:
            raise HTTPException(status_code=400, detail="Tipo de arquivo inválido. Apenas PDFs são aceitos.")
    # Verifica se o corpo do email está vazio
    if not final_text.strip():
        raise HTTPException(status_code=400, detail="Nenhum texto de email fornecido.")
    # Executa as funções síncronas e bloqueantes em um thread separado
    category = await asyncio.to_thread(classify_email, final_text, file_content)
    suggested_response = await asyncio.to_thread(generate_response, final_text, category, file_content)

    return EmailProcessResponse(category=category, suggested_response=suggested_response)

    

@app.get("/")
def read_root():
    return {"message": "API do Classificador de Email está no ar!"}
