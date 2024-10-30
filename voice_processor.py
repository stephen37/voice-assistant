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
        print("Session opened with ID:", session_opened.session_id)

    def on_error(self, error: aai.RealtimeError):
        print("Error:", error)

    def on_close(self):
        print("Session closed")

    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            final_transcript = transcript.text
            print(f"Final transcript: {final_transcript}")
            if self.transcription_callback:
                self.transcription_callback(final_transcript)
            self.final_transcript_queue.put(final_transcript)
        else:
            print(transcript.text, end="\r")

    def create_transcriber(self):
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=16_000,
            on_data=self.on_data,
            on_error=self.on_error,
            on_open=self.on_open,
            on_close=self.on_close,
            word_boost=["Milvus, Zilliz"]
        )
        return self.transcriber

    async def start_continuous_transcription(self):
        print('Starting continuous transcription...')
        self.stop_transcription()
        
        try:
            self.transcriber = self.create_transcriber()
            print("Connecting transcriber...")
            self.transcriber.connect()
            print("Transcriber connected, starting microphone stream...")
            
            microphone_stream = aai.extras.MicrophoneStream(sample_rate=16_000)
            thread = threading.Thread(
                target=self._stream_microphone,
                args=(microphone_stream, self._stream_callback),
                daemon=True
            )
            thread.start()
            print("Microphone stream started successfully")
            
        except Exception as e:
            print(f"Error in start_continuous_transcription: {e}")
            self.transcriber = None
            raise

    def _stream_microphone(self, microphone_stream, stream_callback):
        print("Starting microphone streaming...")
        try:
            for chunk in microphone_stream:
                if not self.transcriber:
                    print("Transcriber no longer exists, stopping microphone stream")
                    break
                if not stream_callback(chunk):
                    break
        except Exception as e:
            print(f"Error in microphone streaming: {e}")
        finally:
            print("Microphone streaming ended")

    def stop_transcription(self):
        print("Stopping transcription...")
        if self.transcriber:
            try:
                self.transcriber.close()
            except Exception as e:
                print(f"Error closing transcriber: {e}")
            self.transcriber = None
        print("Transcription stopped")

    def set_transcription_callback(self, callback):
        self.transcription_callback = callback
    
    def pause_listening(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None

    async def resume_listening(self):
        print('Resuming listening...')
        if not self.transcriber:
            print('creating trasncriber')
            self.create_transcriber()
        
        try:
            print('Trying to connect')
            self.transcriber.connect()
            print('connected')
            microphone_stream = aai.extras.MicrophoneStream(sample_rate=16_000)
            threading.Thread(
                target=self._stream_microphone,
                args=(microphone_stream, self._stream_callback),
                daemon=True  # Make thread daemon so it exits when main program exits
            ).start()
        except Exception as e:
            print(f"Error resuming listening: {e}")
            self.transcriber = None
            raise

    def _stream_callback(self, chunk):
        if not self.transcriber:
            return False
        try:
            self.transcriber.stream(chunk)
            return True
        except Exception as e:
            print(f"Error in stream callback: {e}")
            return False

    async def text_to_speech(self, text: str):
        print("\n=== Starting text-to-speech ===")
        # Explicitly stop the current transcriber
        self.stop_transcription()
        
        try:
            audio_stream = await self.elevenlabs.generate(
                text=text, model="eleven_turbo_v2_5", voice="dDpKZ6xv1gpboV4okVbc"
            )

            print("Audio generated, playing now...")
            audio_data = b""
            async for chunk in audio_stream:
                audio_data += chunk

            play(audio_data)
            print("Audio playback completed")
            
        except Exception as e:
            print(f"Error in text-to-speech conversion or playback: {e}")
            
        finally:
            print("\n=== Restarting transcription ===")
            # Ensure we wait for audio playback to complete
            await asyncio.sleep(1.0)
            # Start a fresh transcription session
            await self.start_continuous_transcription()