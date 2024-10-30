import asyncio
import os
import threading
from queue import Queue

import assemblyai as aai
from elevenlabs import AsyncElevenLabs, play

from config import Config


class VoiceProcessor:
    def __init__(self, config: Config):
        self.config = config
        aai.settings.api_key = config.ASSEMBLY_API_KEY
        self.elevenlabs = AsyncElevenLabs(api_key=config.ELEVENLABS_API_KEY)
        self.transcription_callback = None
        self.final_transcript_queue = Queue()
        self.transcriber = None
        self.is_listening = False
        self.listen_event = threading.Event()
        self.stop_event = threading.Event()
        self.listening_thread = None

    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        print("\n🎤 Speech Recognition Session Started")
        print(f"Session ID: {session_opened.session_id}")

    def on_error(self, error: aai.RealtimeError):
        print("\n❌ Error in Speech Recognition:")
        print(f"    {error}")

    def on_close(self):
        print("\n🔚 Speech Recognition Session Ended")

    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            final_transcript = transcript.text
            print("\n📝 Final Transcript:")
            print(f"    \"{final_transcript}\"")
            if self.transcription_callback:
                self.transcription_callback(final_transcript)
            self.final_transcript_queue.put(final_transcript)
        else:
            # For real-time partial transcripts, overwrite the line
            print(f"\r🎙️ {transcript.text}", end="", flush=True)

    def create_transcriber(self):
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=16_000,
            on_data=self.on_data,
            on_error=self.on_error,
            on_open=self.on_open,
            on_close=self.on_close,
            word_boost=["Milvus, Zilliz"],
        )
        return self.transcriber

    async def start_continuous_transcription(self):
        print("\n🚀 Initializing Speech Recognition...")
        self.stop_transcription()

        try:
            self.transcriber = self.create_transcriber()
            print("    ✓ Created transcriber")
            self.transcriber.connect()
            print("    ✓ Connected to speech recognition service")

            microphone_stream = aai.extras.MicrophoneStream(sample_rate=16_000)
            thread = threading.Thread(
                target=self._stream_microphone,
                args=(microphone_stream, self._stream_callback),
                daemon=True,
            )
            thread.start()
            print("    ✓ Started microphone stream")
            print("\n🎙️ Listening...")

        except Exception as e:
            print("\n❌ Error during initialization:")
            print(f"    {str(e)}")
            self.transcriber = None
            raise

    def _stream_microphone(self, microphone_stream, stream_callback):
        try:
            for chunk in microphone_stream:
                if not self.transcriber:
                    print("\n⚠️  Stopping microphone stream (transcriber closed)")
                    break
                if not stream_callback(chunk):
                    break
        except Exception as e:
            print("\n❌ Microphone Stream Error:")
            print(f"    {str(e)}")
        finally:
            print("\n🔚 Microphone stream ended")

    def stop_transcription(self):
        print("\n⏹️  Stopping Speech Recognition...")
        if self.transcriber:
            try:
                self.transcriber.close()
                print("    ✓ Transcriber closed successfully")
            except Exception as e:
                print("    ❌ Error closing transcriber:")
                print(f"       {str(e)}")
            self.transcriber = None

    def set_transcription_callback(self, callback):
        self.transcription_callback = callback

    def _stream_callback(self, chunk):
        if not self.transcriber:
            return False
        try:
            self.transcriber.stream(chunk)
            return True
        except Exception as e:
            print("\n❌ Stream Processing Error:")
            print(f"    {str(e)}")
            return False

    async def text_to_speech(self, text: str):
        print("\n🔊 Converting text to speech...")
        self.stop_transcription()

        try:
            print("    ⏳ Generating audio...")
            audio_stream = await self.elevenlabs.generate(
                text=text, model="eleven_turbo_v2_5", voice="mZ8K1MPRiT5wDQaasg3i"
            )

            print("    ✓ Audio generated successfully")
            audio_data = b""
            async for chunk in audio_stream:
                audio_data += chunk

            print("    🔈 Playing audio response...")
            play(audio_data)
            print("    ✓ Audio playback completed")

        except Exception as e:
            print("    ❌ Text-to-Speech Error:")
            print(f"       {str(e)}")
        
        print("\n✨ Ready for next input")
        print("   Press SPACE to start recording")