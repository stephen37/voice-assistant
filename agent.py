import asyncio
from typing import List

import requests

from config import Config
from vector_search import MilvusWrapper
from voice_processor import VoiceProcessor
from web_searcher import WebSearcher


class Agent:
    def __init__(self, config, voice_processor):
        self.config = config
        self.voice_processor = voice_processor

    async def process_voice_query(self, audio_data: bytes):
        query_text = await self.voice_processor.start_transcription()

        # Search in Milvus
        # Note: You'll need to implement the embedding generation for the query
        query_vector = self._get_embedding(query_text)  # Implement this method
        results = self.milvus_client.search(query_vector)

        if not results:
            # If no results found, search the web
            self.voice_processor.text_to_speech(
                "Please wait while I search for information."
            )
            web_results = self.web_searcher.search(query_text)

            # Convert web results to embeddings and store in Milvus
            embeddings = [self._get_embedding(result) for result in web_results]
            self.milvus_client.insert(web_results, embeddings)

            # Search again with the updated database
            results = self.milvus_client.search(query_vector)

        # Synthesize and play the response
        response = self._format_response(results)
        self.voice_processor.text_to_speech(response)

    def _get_embedding(self, text: str) -> List[float]:
        print("Generating Embeddings")
        payload = {"input": text, "model": "jina-embeddings-v3"}
        response = requests.post(self.embedding_url, json=payload, headers=self.headers)

        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        else:
            raise Exception(
                f"Error getting embedding: {response.status_code} - {response.text}"
            )

    def _format_response(self, results: List[dict]) -> str:
        # Implement your response formatting logic here
        # This could include summarizing the results or picking the best answer
        return (
            results[0]["content"]
            if results
            else "I couldn't find any relevant information."
        )
