from typing import List

from duckduckgo_search import DDGS

from config import Config


class WebSearcher:
    def __init__(self, config: Config):
        self.config = config
        self.DDGS = DDGS()

    def search(self, query: str) -> List[str]:
        print(f"run DDG Search for: {query}")
        results = self.DDGS.text(query, max_results=self.config.DUCKDUCKGO_NUM_RESULTS)
        print(f"Results from ddg: {results}")
        return [result["body"] for result in results]
