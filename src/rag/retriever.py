import numpy as np
from typing import List, Dict, Tuple
import logging
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.embeddings import EmbeddingGenerator
from src.utils.config import TOP_K, SIMILARITY_THRESHOLD, DEBUG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentRetriever:
    """
     Classe responsável por buscar documentos relevantes usando embeddings
    """

    def __init__(self):
        logger.info("🔍 Inicializando sistema de busca...")
        
        self.generator = EmbeddingGenerator()
        
        try:
            self.chunks, self.embeddings, self.metadata = self.generator.load_embeddings()
            logger.info(f"✅ Sistema carregado: {len(self.chunks)} chunks disponíveis")
            
            if DEBUG:
                logger.info(f"📊 Dimensão dos embeddings: {self.embeddings.shape}")
                sources = set(chunk['source'] for chunk in self.chunks)
                logger.info(f"📚 Fontes: {list(sources)}")
                
        except Exception as e:
            logger.error(f"❌ Erro ao carregar embeddings: {e}")
            logger.error("💡 Execute primeiro: python -m src.rag.embeddings")
            raise
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Busca documentos relevantes para uma consulta
        """
        
        if top_k is None:
            top_k = TOP_K
            
        logger.info(f"🔍 Buscando: '{query}' (top {top_k})")
        
        # 1. Transforma a pergunta em embedding
        query_embedding = self.generator.create_embeddings([query])
        
        if query_embedding.size == 0:
            logger.warning("⚠️ Não foi possível criar embedding da consulta")
            return []
        
        # 2. Calcula similaridade com todos os chunks
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # 3. Encontra os índices dos mais similares
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 4. Filtra por threshold se especificado
        results = []
        for idx in top_indices:
            similarity_score = float(similarities[idx])
            
            if similarity_score >= SIMILARITY_THRESHOLD:
                chunk = self.chunks[idx].copy()  
                chunk['similarity_score'] = similarity_score
                chunk['rank'] = len(results) + 1
                results.append(chunk)
                
                if DEBUG:
                    logger.info(f"📄 Resultado {len(results)}: {similarity_score:.3f} - {chunk['text'][:100]}...")
        
        logger.info(f"✅ Encontrados {len(results)} resultados relevantes")
        return results
    
    def search_with_context(self, query: str, top_k: int = None) -> Tuple[List[Dict], str]:
        """
        Busca documentos e retorna também o contexto concatenado
        """
        
        results = self.search(query, top_k)
        
        if not results:
            return results, "Nenhum contexto relevante encontrado."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Fonte {i}: {result['source']}]\n{result['text']}")
        
        full_context = "\n\n".join(context_parts)
        
        if DEBUG:
            logger.info(f"📝 Contexto criado: {len(full_context)} caracteres")
        
        return results, full_context
    
    def get_statistics(self) -> Dict:
        """
        Retorna estatísticas do sistema de busca
        """
        
        if not hasattr(self, 'chunks') or not self.chunks:
            return {"error": "Sistema não inicializado"}
        
        sources = {}
        total_chars = 0
        
        for chunk in self.chunks:
            source = chunk['source']
            if source not in sources:
                sources[source] = {'chunks': 0, 'characters': 0}
            
            sources[source]['chunks'] += 1
            sources[source]['characters'] += chunk.get('length', 0)
            total_chars += chunk.get('length', 0)
        
        return {
            'total_chunks': len(self.chunks),
            'total_characters': total_chars,
            'embedding_dimension': self.embeddings.shape[1] if len(self.embeddings.shape) > 1 else 0,
            'sources': sources,
            'model_name': self.metadata.get('model_name', 'unknown'),
            'created_at': self.metadata.get('timestamp', 'unknown')
        }
    
def search_documents(query: str, top_k: int = None) -> List[Dict]:
    """
    Função simples para buscar documentos
    """
    retriever = DocumentRetriever()
    return retriever.search(query, top_k)

def get_context_for_query(query: str, top_k: int = None) -> str:
    """
    Função simples para obter contexto para uma pergunta
    """
    retriever = DocumentRetriever()
    _, context = retriever.search_with_context(query, top_k)
    return context

#TESTE SIMPLES
if __name__ == "__main__":
    print("🧪 Testando sistema de busca...")
    
    try:
        # Inicializa o retriever
        retriever = DocumentRetriever()
        
        # Mostra estatísticas
        stats = retriever.get_statistics()
        print(f"📊 Estatísticas do sistema:")
        print(f"   📄 Total de chunks: {stats['total_chunks']}")
        print(f"   📝 Total de caracteres: {stats['total_characters']:,}")
        print(f"   🤖 Modelo: {stats['model_name']}")
        
        # Testes de busca
        test_queries = [
            "Quem ganhou o campeonato brasileiro em 2020?",
            "Quantos times participam do brasileirão?",
            "Quql o primeiro ano que teve o brasileirao?",
            "Corinthians ganhou quantas vezes ?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testando: '{query}'")
            results = retriever.search(query, top_k=3)
            
            if results:
                print(f"✅ Encontrados {len(results)} resultados:")
                for i, result in enumerate(results, 1):
                    print(f"   {i}. Score: {result['similarity_score']:.3f}")
                    print(f"      Fonte: {result['source']}")
                    print(f"      Texto: {result['text'][:150]}...")
            else:
                print("❌ Nenhum resultado encontrado")
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")