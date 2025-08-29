import os
import spacy
from pypdf import PdfReader
from io import BytesIO
from fastapi import UploadFile
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Carrega o modelo do SpaCy
nlp = spacy.load("en_core_web_sm")

# Carrega a API Key do .env
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Configuração do modelo de Geração de Texto (LLM)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.7,
)

async def get_text_from_file(file: UploadFile) -> str:
    """Extrai texto de um arquivo .txt ou .pdf."""
    content = await file.read()
    if file.filename.endswith(".pdf"):
        reader = PdfReader(BytesIO(content))
        text = "".join(page.extract_text() for page in reader.pages)
    else: # Assume .txt
        text = content.decode("utf-8")
    return text

def classify_email(email_text: str) -> str:
    """Classifica o email como Produtivo ou Improdutivo usando um prompt."""
    # Usamos um LLM para classificação zero-shot, que é mais flexível.
    # O prompt instrui o modelo a fazer a classificação.
    classification_prompt = PromptTemplate.from_template(
        """
        Classify the following email text as either "Productive" or "Unproductive".
        Only return the single word classification.

        Email: "{text}"
        Classification:
        """
    )
    chain = classification_prompt | llm | StrOutputParser()
    result = chain.invoke({"text": email_text})
    
    # Limpeza simples para garantir que temos apenas a palavra esperada
    if "productive" in result.lower():
        return "Productive"
    return "Unproductive"

def generate_response(email_text: str, category: str) -> str:
    """Gera uma resposta sugerida com base na categoria."""
    template = """
    Based on the following email, which was classified as '{category}', write a brief and appropriate suggested response.

    Email: "{text}"
    Suggested Response:
    """
    response_prompt = PromptTemplate.from_template(template)
    chain = response_prompt | llm | StrOutputParser()
    return chain.invoke({"text": email_text, "category": category})
