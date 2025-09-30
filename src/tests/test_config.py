try:
    from src.utils.config import GOOGLE_API_KEY, CHUNK_SIZE, DOCUMENTS_DIR
    print("✅ Config funcionando!")
    print(f"📏 Chunk size: {CHUNK_SIZE}")
    print(f"📁 Documentos: {DOCUMENTS_DIR}")
    print(f"🔑 API Key: {'✅ Configurada' if GOOGLE_API_KEY else '❌ Faltando'}")
except Exception as e:
    print(f"❌ Erro: {e}")