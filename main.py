import asyncio
import keyboard
import signal

from calendar_service import CalendarService
from config import Config
from llm_processor import LLMProcessor
from vector_search import MilvusWrapper
from voice_processor import VoiceProcessor
from web_searcher import WebSearcher


class VoiceAssistant:
    def __init__(self):
        self.config = Config()
        self.voice_processor = VoiceProcessor(self.config)
        self.llm_processor = LLMProcessor(self.config)
        self.milvus_wrapper = MilvusWrapper(self.config)
        self.milvus_wrapper.add_sample_data()
        self.web_searcher = WebSearcher(self.config)
        self.calendar_service = CalendarService(self.config)
        self.running = False
        self.loop = asyncio.get_event_loop()
        self.is_transcribing = False 

    async def process_transcription(self, text: str):
        print("\n👤 User Query:")
        print(f"    \"{text}\"")

        # Search Milvus for similar content
        print("\n🔍 Searching knowledge base...")
        milvus_results = self.milvus_wrapper.search_similar_text(text)
        relevant_results = [result for result in milvus_results if result["distance"] > 0.4]

        if relevant_results:
            print("\n📚 Found relevant information in knowledge base")
            context = "\n".join([result["text"] for result in milvus_results])
            print("    Context snippets:")
            for i, result in enumerate(milvus_results, 1):
                print(f"    {i}. {result['text'][:100]}...")

            augmented_query = f"Context: {context}\n\nUser Query: {text}\n\nPlease answer the user's query based on the given context."
            llm_response = await self.llm_processor.process_query(augmented_query)

        # Check for calendar-related queries
        elif any(keyword in text.lower() for keyword in ["calendar", "schedule", "events", "appointment"]):
            print("\n📅 Processing calendar-related query...")
            events = await self.calendar_service.get_upcoming_events()
            augmented_query = f"Calendar Events:\n{events}\n\nUser Query: {text}\n\nPlease answer the user's query based on their calendar events."
            llm_response = await self.llm_processor.process_query(augmented_query)

        # Fall back to web search
        else:
            print("\n🌐 Searching the web for information...")
            web_results = self.web_searcher.search(text)

            if web_results:
                print("    ✓ Found relevant web results")
                context = "\n".join(web_results[:3])
                print("    Web snippets:")
                for i, result in enumerate(web_results[:3], 1):
                    print(f"    {i}. {result[:100]}...")

                augmented_query = f"Web search results:\n{context}\n\nUser Query: {text}\n\nPlease answer the user's query based on the web search results."
                llm_response = await self.llm_processor.process_query(augmented_query)
            else:
                print("    ⚠️  No web results found, using direct LLM response")
                llm_response = await self.llm_processor.process_query(text)

        print("\n🤖 Assistant Response:")
        print(f"    \"{llm_response}\"")

        print("\n🔊 Converting response to speech...")
        await self.voice_processor.text_to_speech(llm_response)

    def handle_interrupt(self, signum, frame):
        print("\nInterrupt received. Stopping transcription...")
        self.stop()

    def stop(self):
        self.running = False
        self.voice_processor.stop_transcription()

    def transcription_callback(self, text: str):
        asyncio.run_coroutine_threadsafe(self.process_transcription(text), self.loop)

    async def toggle_transcription(self):
        self.is_transcribing = not self.is_transcribing
        if self.is_transcribing:
            print("\n🎙️ Started recording...")
            await self.voice_processor.start_continuous_transcription()
        else:
            print("\n⏹️  Stopped recording...")
            self.voice_processor.stop_transcription()
        
    async def run(self):
        self.voice_processor.set_transcription_callback(self.transcription_callback)
        signal.signal(signal.SIGINT, self.handle_interrupt)

        print("\n🚀 Voice Assistant Started")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("    Press SPACE to toggle recording")
        print("    Press Ctrl+C to exit")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        
    async def run(self):
        self.voice_processor.set_transcription_callback(self.transcription_callback)
        signal.signal(signal.SIGINT, self.handle_interrupt)

        print("Starting voice assistant. Press SPACE to toggle recording. Press Ctrl+C to exit.")
        
        def on_space_press():
            asyncio.run_coroutine_threadsafe(self.toggle_transcription(), self.loop)

        keyboard.on_press_key('space', lambda _: on_space_press())
        
        self.running = True
        while self.running:
            await asyncio.sleep(0.1)


async def main():
    assistant = VoiceAssistant()
    await assistant.run()


if __name__ == "__main__":
    asyncio.run(main())
