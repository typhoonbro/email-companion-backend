from pypdf import PdfReader
from io import BytesIO
from fastapi import UploadFile, HTTPException
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuração do modelo de Geração de Texto (LLM)
# 1. Crie o endpoint que se conecta à API do Hugging Face
endpoint = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b", 
    temperature=0.7,
    provider="cerebras",
)

# 2. Use o wrapper ChatHuggingFace para tratar o endpoint como um modelo de chat
llm = ChatHuggingFace(llm=endpoint)

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extrai texto de um conteúdo de PDF em bytes."""
    if not pdf_content:
        return ""
    reader = PdfReader(BytesIO(pdf_content))
    print(f"--- Extração de PDF: Número de páginas no PDF: {"".join(page.extract_text() or "" for page in reader.pages)}")
    return "".join(page.extract_text() or "" for page in reader.pages)

def classify_email(email_text: str, file_content: str) -> str:
    """Classifica o email como Produtivo ou Improdutivo usando um prompt."""
    classification_prompt = PromptTemplate.from_template(
        """

        Classifique o seguinte texto de email como "Produtivo" ou "Improdutivo".
        Email: "{text}"
        Conteúdo do PDF anexado (se houver): "{file_content}"
        Responda apenas uma única palavra com a classificação. Não forneça explicações adicionais ou a linha de raciocínio, apenas a classificação.
        Classificação:
        """
    )
    chain = classification_prompt | llm | StrOutputParser()

    result = None
    try:
        print("--- Classificação: Tentando invocar o modelo...")
        result = chain.invoke({"text": email_text, "file_content": file_content})
        print(f"--- Classificação: Modelo retornou: '{result}' (Tipo: {type(result)})")
    except Exception as e:
        print(f"--- Classificação: Ocorreu uma exceção durante a chamada do modelo: {e}")
        # Re-lança a exceção como um erro HTTP claro para o frontend
        raise HTTPException(status_code=503, detail=f"O serviço de classificação falhou: {e}")

    if not result or not isinstance(result, str):
        print(f"--- Classificação: Resultado inválido ou vazio recebido do modelo: {result}")
        raise HTTPException(
            status_code=500,
            detail="Falha ao obter uma classificação válida do modelo.",
        )
    clean_result = result.strip()
    if "Produtivo" in clean_result:
        return "Produtivo"
    elif "Improdutivo" in clean_result:
        return "Improdutivo"
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Classificação inesperada recebida do modelo: '{result}'",
        )


def generate_response(email_text: str, category: str, file_content: str) -> str:
    """Gera uma resposta sugerida com base na categoria."""
    template = """
    Você é um assistente de redação de e-mails da minha empresa. Voce sera um assistente para analistas que irão efetivamente responder solicitações feitas por email para a minha empresa. Sua única tarefa é escrever uma resposta breve e profissional para o e-mail fornecido, com base na categoria. Lembre-se de que você está respondendo em nome de uma empresa, então mantenha um tom formal e profissional. 

    Considerando que somos uma empresa que realiza negócios online, é crucial que nossas respostas sejam claras, concisas e transmitam profissionalismo. Não devemos incluir informações pessoais ou informais. Mas devemos considerar o contato de possíveis clientes e parceiros, então a resposta deve ser acolhedora e convidativa. 

    Em caso de dúvidas sobre o conteúdo do e-mail, responda de forma genérica, solicitando mais informações se necessário.

    **REGRAS ESTRITAS:**
    - NÃO inclua sua própria assinatura.
    - NÃO adicione qualquer texto explicativo ou prefixos como "O usuário pediu...".
    - Forneça APENAS o corpo da resposta do e-mail em sua resposta.
    - NÃO diga as instruções novamente.
    - APENAS escreva a resposta em si do e-mail.

    [E-MAIL ORIGINAL]
    {text}
    [CONTEÚDO DO PDF ANEXADO, SE HOUVER]
    {file_content}
    [CATEGORIA]
    {category}

    
    """
    response_prompt = PromptTemplate.from_template(template)
    chain = response_prompt | llm | StrOutputParser()

    result = None
    try:
        print("--- Geração de Resposta: Tentando invocar o modelo...")
        result = chain.invoke({"text": email_text, "category": category, "file_content": file_content})
        print(f"--- Geração de Resposta: Modelo retornou: '{result}' (Tipo: {type(result)})")
    except Exception as e:
        print(f"--- Geração de Resposta: Ocorreu uma exceção durante a chamada do modelo: {e}")
        raise HTTPException(status_code=503, detail=f"O serviço de geração de resposta falhou: {e}")

    # Validação rigorosa do resultado
    if not result or not isinstance(result, str):
        print(f"--- Geração de Resposta: Resultado inválido ou vazio recebido do modelo: {result}")
        raise HTTPException(
            status_code=500,
            detail="Falha ao gerar uma resposta válida do modelo.",
        )

    return result.strip()
