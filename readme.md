# AI Voice Assistant with Milvus Vector Database

## 🚀 Overview
This project is a voice-enabled AI assistant that leverages [Milvus](https://github.com/milvus-io/milvus) vector database for semantic search capabilities, allowing for contextual understanding and intelligent responses to user queries.

⭐ If you find this project helpful, please consider giving it a star to [Milvus](https://github.com/milvus-io/milvus)!


## 🎯 Key Features

### Primary Component: Milvus Vector Database
- High-performance vector similarity search
- Hardware-optimized for efficient querying
- Supports advanced search algorithms (IVF, HNSW, DiskANN)
- Column-oriented architecture for optimized data access
- Integrated with Jina embeddings for text vectorization

### Supporting Stack
- **Voice Processing**: AssemblyAI for real-time speech-to-text
- **Text-to-Speech**: ElevenLabs for natural voice synthesis
- **LLM Integration**: Ollama for text processing and response generation
- **Web Search**: DuckDuckGo for fallback information retrieval
- **Calendar Integration**: Google Calendar API for schedule management

## 🛠 Technical Architecture

The system follows a multi-stage processing pipeline:
1. Voice input is captured and transcribed in real-time
2. Text queries are converted to vectors and searched in Milvus
3. Relevant context is retrieved and augmented with the query
4. LLM processes the augmented query and generates a response
5. Response is converted to speech and played back

## 📋 Prerequisites
- Python 3.11+
- Milvus server running locally (default: localhost:19530)
- Required API keys:
  - ElevenLabs
  - AssemblyAI
  - Jina AI
  - Google Calendar credentials

## 🚀 Quick Start

1. Install dependencies using Poetry:
```bash
poetry install
```

2. Set up environment variables:
```bash
ELEVENLABS_API_KEY=your_key
ASSEMBLY_API_KEY=your_key
JINA_API_KEY=your_key
```
3. Run the assistant:
```bash
python main.py
```


## 💡 Usage
- Press SPACE to start/stop recording
- Press Ctrl+C to exit
- The assistant will:
  - Search Milvus for relevant context
  - Check calendar for schedule-related queries when needed
  - Fall back to web search when needed
  - Respond with synthesized speech

## 🔍 Vector Search Details
The system uses Milvus for efficient vector similarity search with:
- 1024-dimensional vectors
- Jina embeddings for text vectorization
- Configurable similarity threshold (currently 0.4)
- Sample knowledge base included for demonstration

## 📚 Sample Data
The system comes pre-loaded with sample data about:
- AI history
- Milvus architecture
- Vector database performance characteristics