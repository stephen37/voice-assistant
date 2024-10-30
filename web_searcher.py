from typing import List

from duckduckgo_search import DDGS

from config import Config


class WebSearcher:
    def __init__(self, config: Config):
        self.config = config
        self.DDGS = DDGS()

    def search(self, query: str) -> List[str]:
        results = self.DDGS.text(query, max_results=self.config.DUCKDUCKGO_NUM_RESULTS)
        return [result["body"] for result in results]
