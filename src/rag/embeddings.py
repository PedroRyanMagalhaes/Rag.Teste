import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Tuple
import pickle 
import logging 
from pathlib import Path
import json
from datetime import datetime

from src.utils.config import(
    EMBEDDING_MODEL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PROCESSED_DIR,
    EMBEDDINGS_FILE_PATH,
    TEXTS_FILE_PATH,
    METADATA_FILE_PATH,
    DEBUG
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Classe responsavel por criar e gerenciar embeddings de textos.
    """

    def __init__(self):
        logger.info("Inicializando o modelo de embeddings...")

        try: 
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"Modelo de embeddings carregado: {EMBEDDING_MODEL}")

            self.embeddings_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Dimensão dos embeddings: {self.embeddings_dim}")
        except Exception as e:
            logger.error(f"Erro ao carregar o modelo de embeddings: {e}")
            raise

    def split_text_into_chunks(self, text:str, source_name: str = "documento") -> List[Dict]:
        """
        Divide texto em chunks com sobreposição
        
        Args:
            text (str): Texto para dividir
            source_name (str): Nome da fonte do texto
            
        Returns:
            List[Dict]: Lista de chunks com metadados
        """

        if len(text) <= CHUNK_SIZE:
            return [{"text": text, "source": source_name, "chunk_id": 0, "start_pos":0 , "end_pos": len(text)}]
        
        chunks = []
        start = 0
        chunk_id = 0

        logger.info(f"Dividindo texto em chunks (tamanho: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP})")

        while start < len(text):
            end = start + CHUNK_SIZE

            if end < len(text):
                chunk_preview = text[start:end]

                last_period = chunk_preview.rfind('.')
                last_newline = chunk_preview.rfind('\n')
                last_space = chunk_preview.rfind(' ')

                min_size = int(CHUNK_SIZE * 0.80)
                best_break = -1

                for break_pos in [last_period, last_newline, last_space]:
                    if break_pos > min_size:
                        best_break = max(best_break, break_pos)

                if best_break > 0:
                    end = start + best_break + 1

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_info = {
                    "text": chunk_text,
                    "source": source_name,
                    "chunk_id": chunk_id,
                    "start_pos": start,
                    "end_pos": end,
                    'length': len(chunk_text),
                }
                chunks.append(chunk_info)
                chunk_id += 1
            
            start = end - CHUNK_OVERLAP

            if start >= len(text) - CHUNK_OVERLAP:
                break

        logger.info(f"Texto dividido em {len(chunks)} chunks.")
        return chunks   
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Cria embeddings para uma lista de textos
        
        Args:
            texts (List[str]): Lista de textos
            
        Returns:
            np.ndarray: Array de embeddings
        """
        if not texts:
            logger.warning("Lista de textos vazia. Nenhum embedding criado.")
            return np.array([])
        
        logger.info(f"Criando embeddings para {len(texts)} textos...")

        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=True if len(texts) > 10 else False,
                batch_size=32
            )

            logger.info(f"Embeddings criados com sucesso. Shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Erro ao criar embeddings: {e}")
            raise

    def process_document(self, text:str, source_name: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Processa um documento completo: divide em chunks e cria embeddings
        """
        logger.info (f"Processando documento: {source_name}")

        # 1. Divide em chunks
        chunks = self.split_text_into_chunks(text, source_name)
        
        # 2. Extrai só os textos para gerar embeddings
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # 3. Gera embeddings
        embeddings = self.create_embeddings(chunk_texts)

        # 4. Adiciona informações dos embeddings aos chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding_index'] = i
            chunk['embeddings_dim'] = self.embeddings_dim
        
        logger.info(f"✅ Documento processado: {len(chunks)} chunks, {embeddings.shape} embeddings")
        return chunks, embeddings

    def save_embeddings(self, chunks: List[Dict], embeddings: np.ndarray, additional_metadata: Dict = None) -> None:
        """
        Salva os chunks e embeddings em arquivos
        
        Args:
            chunks (List[Dict]): Lista de chunks com metadados
            embeddings (np.ndarray): Array de embeddings
        """
        try:
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

            with open(TEXTS_FILE_PATH, 'wb') as f:
                pickle.dump(chunks, f)
            logger.info(f"Chunks salvos em {TEXTS_FILE_PATH}")

            np.save(EMBEDDINGS_FILE_PATH, embeddings)
            logger.info(f"Embeddings salvos em {EMBEDDINGS_FILE_PATH}")

            metadata = {
                "timestamp": datetime.now().isoformat(),
                "model_name": EMBEDDING_MODEL,
                "embedding_dim": self.embeddings_dim,
                "num_chunks": len(chunks),
                "num_embeddings": embeddings.shape[0],
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "sources": list(set(chunk['source'] for chunk in chunks)),
            }

            if additional_metadata:
                metadata.update(additional_metadata)

            with open(METADATA_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Metadata salva em {METADATA_FILE_PATH}")

        except Exception as e:
            logger.error(f"Erro ao salvar embeddings e metadados: {e}")
            raise

    def load_embeddings(self) -> Tuple[List[Dict], np.ndarray, Dict]:
        """
        Carrega chunks, embeddings e metadados de arquivos
        
        Returns:
            Tuple[List[Dict], np.ndarray, Dict]: (chunks, embeddings, metadata)
        """
        logger.info("Carregando embeddings e metadados...")

        try:
            if not TEXTS_FILE_PATH.exists():
                raise FileNotFoundError(f"Arquivo de textos não encontrado: {TEXTS_FILE_PATH}")
            if not EMBEDDINGS_FILE_PATH.exists():
                raise FileNotFoundError(f"Arquivo de embeddings não encontrado: {EMBEDDINGS_FILE_PATH}")
            with open(TEXTS_FILE_PATH, 'rb') as f:
                chunks = pickle.load(f)
            embeddings = np.load(EMBEDDINGS_FILE_PATH)
            
            metadata = {}
            if METADATA_FILE_PATH.exists():
                with open(METADATA_FILE_PATH, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

            logger.info(f"✅ Carregados: {len(chunks)} chunks, {embeddings.shape} embeddings")
            return chunks, embeddings, metadata
        except Exception as e:
            logger.error(f"Erro ao carregar embeddings e metadados: {e}")
            raise

    def embeddings_exist(self) -> bool:
        return (TEXTS_FILE_PATH.exists() and EMBEDDINGS_FILE_PATH.exists())
    
    
def process_documents_from_dict(documents: Dict[str, str]) -> Tuple[List[Dict], np.ndarray]:
    """
    Processa múltiplos documentos de uma vez
    
    Args:
        documents (Dict[str, str]): Dicionário nome_arquivo -> texto
        
    Returns:
        Tuple[List[Dict], np.ndarray]: (todos os chunks, todos os embeddings)
    """

    generator = EmbeddingGenerator()

    all_chunks = []
    all_embeddings = []

    for doc_name, text in documents.items():
        logger.info(f"--- Processando: {doc_name} ---")
        chunks, embeddings = generator.process_document(text, doc_name)

        all_chunks.extend(chunks)
        all_embeddings.append(embeddings)

    if all_embeddings:
        final_embeddings = np.vstack(all_embeddings)
    else:
        final_embeddings = np.array([])

    logger.info(f"✅ Todos os documentos processados: {len(all_chunks)} chunks totais")
    return all_chunks, final_embeddings

if __name__ == "__main__":
    from src.utils.pdf_extractor import extract_text_from_directory
    from src.utils.config import DOCUMENTS_DIR

    print ("Testando geração de embeddings...")

    # 1. Extrai textos dos PDFs
    documents = extract_text_from_directory(DOCUMENTS_DIR)
    
    if not documents:
        print("❌ Nenhum documento encontrado para processar")
        exit()
    
    # 2. Processa documentos
    try:
        all_chunks, all_embeddings = process_documents_from_dict(documents)
        
        # 3. Salva resultados
        generator = EmbeddingGenerator()
        generator.save_embeddings(all_chunks, all_embeddings)
        
        print(f"✅ Teste concluído!")
        print(f"📊 Total: {len(all_chunks)} chunks, {all_embeddings.shape} embeddings")
        
        # 4. Testa carregamento
        chunks, embeddings, metadata = generator.load_embeddings()
        print(f"✅ Teste de carregamento OK: {len(chunks)} chunks")
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
    

