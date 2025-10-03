# test_scores.py (arquivo temporário)
from src.rag.retriever import DocumentRetriever

retriever = DocumentRetriever()

test_queries = [
    "Quantos times participam do brasileirão?",
    "Quando começou o brasileirão?", 
    "Corinthians campeão"
]

for query in test_queries:
    print(f"\n🔍 Pergunta: '{query}'")
    
    # Busca sem threshold (para ver todos os scores)
    query_embedding = retriever.generator.create_embeddings([query])
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_embedding, retriever.embeddings)[0]
    
    # Mostra os 5 melhores scores
    top_5 = similarities.argsort()[::-1][:5]
    for i, idx in enumerate(top_5):
        score = similarities[idx]
        text = retriever.chunks[idx]['text'][:100]
        print(f"   {i+1}. Score: {score:.3f} - {text}...")

        python -m src.tests.test_retriever