from qdrant_client import QdrantClient

from embeddings.sentence_embedding import get_embeddings

client = QdrantClient(url="http://localhost:6333")

def get_search_result(text, limit=2):
    embedding_vector = get_embeddings(text)

    search_results = client.query_points(
        collection_name="test_collection",
        query=embedding_vector,
        query_filter=None,  # If you don't want any filters for now
        limit=limit,  # 5 the most closest results is enough
    ).points

    return "\n".join([result.payload["text"] for result in search_results])


