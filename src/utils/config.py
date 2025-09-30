import os 
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY não está definida no .env")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
TOP_K = int(os.getenv("TOP_K", 3))


DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
PROCESSED_DIR = DATA_DIR / "processed"


FAISS_INDEX_PATH = PROCESSED_DIR / "faiss_index"
TEXTS_FILE_PATH = PROCESSED_DIR / "texts.pkl"
EMBEDDINGS_FILE_PATH = PROCESSED_DIR / "embeddings.npy"
METADATA_FILE_PATH = PROCESSED_DIR / "metadata.json"

DEBUG = os.getenv("DEBUG", "False").lower() == "true"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

def ensure_directories():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if DEBUG:
        print(f"Diretórios verificados/criados: {DOCUMENTS_DIR}, {PROCESSED_DIR}")

ensure_directories()

SYSTEM_PROMPT = """
Você é um assistente especializado em responder perguntas sobre documentos.

INSTRUÇÕES:
1. Use APENAS as informações do contexto fornecido
2. Se a informação não estiver no contexto, diga "Não encontrei essa informação nos documentos"
3. Seja preciso e direto nas respostas
4. Cite trechos relevantes quando possível

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:
"""

SIMILARITY_THRESHOLD = 0.7


PROJECT_NAME = "RAG TESTE"
VERSION = "0.1.0"
AUTHOR = "byPedro"


print (f"{PROJECT_NAME} v{VERSION} iniciado. Configurado")
if DEBUG:
    print(f"🔑 API Key configurada: {'✅' if GOOGLE_API_KEY else '❌'}")
    print(f"📏 Chunk size: {CHUNK_SIZE}")
    print(f"🔍 Top K results: {TOP_K}")