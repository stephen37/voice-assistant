from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from pymilvus.model.dense import JinaEmbeddingFunction

from config import Config


class MilvusWrapper:
    def __init__(self, config: Config):
        self.config = config
        self.client = MilvusClient(
            uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}",
        )
        self.ef = JinaEmbeddingFunction(
            "jina-embeddings-v3",
            self.config.JINA_API_KEY,
            task="retrieval.passage",
            dimensions=1024,
        )
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        print("checking if the collection exists in Milvus")
        self.client.drop_collection(self.config.COLLECTION_NAME)

        self.client.create_collection(
            self.config.COLLECTION_NAME,
            dimension=self.config.VECTOR_DIM,
            vector_field_name="embedding",
            auto_id=True,
        )

        self.client.load_collection(self.config.COLLECTION_NAME)

    def search(self, query_vector: list[float], limit: int = 5) -> list[dict]:
        print("Runs a Vector Search in Milvus")
        results = self.client.search(
            collection_name=self.config.COLLECTION_NAME,
            data=[query_vector],
            limit=limit,
            output_fields=["content"],
        )
        print(f"results of the vector search: {results}")

        return [
            {"content": hit["entity"].get("content"), "distance": hit["distance"]}
            for hit in results[0]
        ]

    def add_sample_data(self):
        print("Adding sample data to Milvus collection")

        sample_texts = [
            "In 1950, Alan Turing published his seminal paper, 'Computing Machinery and Intelligence,' proposing the Turing Test as a criterion of intelligence, a foundational concept in the philosophy and development of artificial intelligence.",
            "The Dartmouth Conference in 1956 is considered the birthplace of artificial intelligence as a field; here, John McCarthy and others coined the term 'artificial intelligence' and laid out its basic goals.",
            "In 1951, British mathematician and computer scientist Alan Turing also developed the first program designed to play chess, demonstrating an early example of AI in game strategy.",
            "The invention of the Logic Theorist by Allen Newell, Herbert A. Simon, and Cliff Shaw in 1955 marked the creation of the first true AI program, which was capable of solving logic problems, akin to proving mathematical theorems.",
            "The High-Performance Vector Database Built for Scale. Milvus is an open-source vector database built for GenAI applications. Install with pip, perform high-speed searches, and scale to tens of billions of vectors with minimal performance loss.",
            "Milvus is an open-source project under LF AI & Data Foundation distributed under the Apache 2.0 license. Most contributors are experts from the high-performance computing (HPC) community, specializing in building large-scale systems and optimizing hardware-aware code. Core contributors include professionals from Zilliz, ARM, NVIDIA, AMD, Intel, Meta, IBM, Salesforce, Alibaba, and Microsoft.",
            """What Makes Milvus so Fast？
Milvus was designed from day one to be a highly efficient vector database system. In most cases, Milvus outperforms other vector databases by 2-5x (see the VectorDBBench results). This high performance is the result of several key design decisions:

Hardware-aware Optimization: To accommodate Milvus in various hardware environments, we have optimized its performance specifically for many hardware architectures and platforms, including AVX512, SIMD, GPUs, and NVMe SSD.

Advanced Search Algorithms: Milvus supports a wide range of in-memory and on-disk indexing/search algorithms, including IVF, HNSW, DiskANN, and more, all of which have been deeply optimized. Compared to popular implementations like FAISS and HNSWLib, Milvus delivers 30%-70% better performance.

Search Engine in C++: Over 80% of a vector database’s performance is determined by its search engine. Milvus uses C++ for this critical component due to the language’s high performance, low-level optimization, and efficient resource management. Most importantly, Milvus integrates numerous hardware-aware code optimizations, ranging from assembly-level vectorization to multi-thread parallelization and scheduling, to fully leverage hardware capabilities.

Column-Oriented: Milvus is a column-oriented vector database system. The primary advantages come from the data access patterns. When performing queries, a column-oriented database reads only the specific fields involved in the query, rather than entire rows, which greatly reduces the amount of data accessed. Additionally, operations on column-based data can be easily vectorized, allowing for operations to be applied in the entire columns at once, further enhancing performance.""",
        ]

        embeddings = self.ef.encode_documents(sample_texts)

        data = [
            {"content": sample_texts[i], "embedding": embeddings[i].tolist()}
            for i in range(len(embeddings))
        ]

        result = self.client.insert(
            collection_name=self.config.COLLECTION_NAME, data=data
        )
        print(f"Added {result['insert_count']} sample entries to the collection")

    def search_similar_text(self, query_text: str, limit: int = 3) -> list[dict]:
        print(f"Searching for text similar to: '{query_text}'")
        query_vector = self.ef.encode_queries([query_text])[0]

        results = self.client.search(
            collection_name=self.config.COLLECTION_NAME,
            data=[query_vector],
            limit=limit,
            output_fields=["content"],
        )[0]
        print(f"Found {len(results)} results")
        print(f"Returning results: {results}")
        return [
            {"text": hit["entity"].get("content"), "distance": hit["distance"]}
            for hit in results
        ]


if __name__ == "__main__":
    config = Config()
    milvus_wrapper = MilvusWrapper(config)
    milvus_wrapper.add_sample_data()

    query_text = "What event in 1956 marked the official birth of artificial intelligence as a discipline?"
    results = milvus_wrapper.search_similar_text(query_text)

    print("Search results:")
    for result in results:
        print(f"Content: {result['text']}")
        print(f"Distance: {result['distance']}")
        print("---")
