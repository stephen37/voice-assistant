import requests
from ollama import AsyncClient

from config import Config


class LLMProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.ollama_client = AsyncClient()
        self.headers = {"Authorization": f"Bearer {self.config.JINA_API_KEY}"}
        self.embedding_url = "https://api.jina.ai/v1/embeddings"

    async def generate_embedding(self, text: str) -> list[float]:
        print("Generating Embeddings")
        payload = {
            "input": text,
            "model": "jina-embeddings-v3",
            "task": "text-matching",
            "dimensions": self.config.VECTOR_DIM,
            "late_chunking": False,
            "embedding_type": "float",
        }
        response = requests.post(self.embedding_url, json=payload, headers=self.headers)

        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        else:
            raise Exception(
                f"Error getting embedding: {response.status_code} - {response.text}"
            )

    async def process_query(self, query: str) -> str:
        try:
            system_message = "Please respond in short, concise sentences."
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query},
            ]
            response = ""
            async for chunk in await self.ollama_client.chat(
                model="llama3.2", messages=messages, stream=True
            ):
                if chunk["done"]:
                    break
                response += chunk["message"]["content"]
            return response
        except Exception as e:
            print(f"Error processing query with LLM: {e}")
            return "I'm sorry, I encountered an error while processing your request."
