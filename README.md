# Qdrant Agentic Memory Workshop

Build memory systems for AI agents using Qdrant vector database.

## What You'll Learn

1. **Basic Memory Storage** - Store and retrieve memories with vector embeddings
2. **Semantic Memory** - Categorize and filter memories by type and importance
3. **Episodic Memory** - Track conversation history across sessions
4. **Memory Agent** - Build a complete agent with persistent memory
5. **Advanced Memory System** - Production-grade memory with consolidation, decay, and reflection

## Prerequisites

- Python 3.10+
- Qdrant Cloud account (free tier works)
- OpenAI API key (optional, for LLM features)

## Setup

### 1. Clone and install dependencies

```bash
cd qdrant-agentic-memory
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```
OPENAI_API_KEY=<your-openai-api-key>
QDRANT_URL=<your-qdrant-cloud-url>
QDRANT_API_KEY=<your-qdrant-api-key>
```

**Qdrant Cloud Setup:**
1. Create account at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a free cluster
3. Copy the URL and API key to your `.env`

**OpenAI Setup (Optional):**
1. Create account at [platform.openai.com](https://platform.openai.com)
2. Generate an API key
3. Add to `.env` for chat/summarization/reflection features

## What's Free vs Paid

| Feature | Cost |
|---------|------|
| Embeddings (FastEmbed) | FREE (runs locally) |
| Qdrant Cloud | FREE tier available |
| Chat responses | Requires OpenAI API key |
| Summarization | Requires OpenAI API key |
| Reflection cycles | Requires OpenAI API key |

**All core memory features work without any API costs.**

## Workshop Modules

### Quick Start

| Module | Difficulty | What You'll Build |
|--------|------------|-------------------|
| 01_basics | Beginner | Store and search memories |
| 02_semantic_memory | Beginner | Categorized memory with filtering |
| 03_episodic_memory | Intermediate | Conversation history tracking |
| 04_memory_agent | Intermediate | Complete agent with memory |
| **05_advanced_memory_system** | **Advanced** | **Production-grade memory with consolidation, decay, reflection** |

### 01: Basic Memory Storage

Learn the fundamentals of storing and searching memories.

```bash
python -m workshop.01_basics
```

**Concepts covered:**
- Creating Qdrant collections
- Using FastEmbed for local embeddings (FREE)
- Storing memories with metadata
- Vector similarity search

### 02: Semantic Memory

Build a categorized memory system with importance scoring.

```bash
python -m workshop.02_semantic_memory
```

**Concepts covered:**
- Memory types (facts, skills, preferences, context)
- Payload filtering in Qdrant
- Importance-based retrieval
- Access tracking

### 03: Episodic Memory

Store and retrieve conversation history.

```bash
python -m workshop.03_episodic_memory
```

**Concepts covered:**
- Session-based conversation storage
- Retrieving relevant past conversations
- Summarizing episodes with LLM (optional)

### 04: Memory Agent

Build a complete agent with persistent memory.

```bash
python -m workshop.04_memory_agent
```

**Concepts covered:**
- Combining memory types
- Memory-augmented response generation
- Automatic memory extraction from conversations
- Interactive chat with memory

### 05: Advanced Memory System (Production-Grade)

Build a sophisticated memory architecture for production agents.

```bash
python -m workshop.05_advanced_memory_system
```

**This is the real deal.** A three-tier memory system with:

**Memory Tiers:**
- **Working Memory**: Current session buffer (volatile)
- **Episodic Memory**: Past conversation sessions (time-bound)
- **Semantic Memory**: Consolidated knowledge (persistent)
- **Meta Memory**: Reflections and insights about the user

**Advanced Features:**

| Feature | Description |
|---------|-------------|
| **Consolidation** | Automatically compress episodic memories into semantic knowledge |
| **Importance Decay** | Memories fade unless reinforced through access (7-day half-life) |
| **Memory Linking** | Graph-like connections between related memories |
| **Reflection Cycles** | Agent reviews past interactions to extract meta-insights |
| **Forgetting** | Remove low-value memories to prevent unbounded growth |
| **Unified Recall** | Query across all memory tiers with importance-weighted ranking |

**Interactive Commands:**
```
/stats        - Show memory statistics
/recall <q>   - Query all memory tiers
/reflect      - Run reflection cycle
/consolidate  - Compress episodes to semantic
/forget       - Remove decayed memories
/clear        - Reset all memories
```

**Memory Flow:**
```
Working Memory ──────► Episodic Memory ──────► Semantic Memory
   (session)              (commit)            (consolidation)
                              │
                              ▼
                         Meta Memory
                        (reflection)
```

## Architecture

### Basic (Workshops 01-04)
```
┌─────────────────────────────────────────────────────────┐
│                    Memory Agent                          │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────┐   │
│  │         FastEmbed (local) + Qdrant Cloud         │   │
│  └──────────────────────────────────────────────────┘   │
│                          │                               │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐       │
│  │  Semantic   │  │  Episodic   │  │  Working   │       │
│  │   Memory    │  │   Memory    │  │   Memory   │       │
│  └─────────────┘  └─────────────┘  └────────────┘       │
└─────────────────────────────────────────────────────────┘
```

### Advanced (Workshop 05)
```
┌─────────────────────────────────────────────────────────────────┐
│                   Advanced Memory System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User Query                                                     │
│       │                                                          │
│       ▼                                                          │
│   ┌───────────────────────────────────────────────────────┐     │
│   │              Unified Recall Engine                     │     │
│   │    (importance-weighted, decay-adjusted ranking)       │     │
│   └───────────────────────────────────────────────────────┘     │
│       │               │               │               │          │
│       ▼               ▼               ▼               ▼          │
│   ┌───────┐     ┌──────────┐    ┌──────────┐    ┌────────┐     │
│   │Working│────►│ Episodic │───►│ Semantic │    │  Meta  │     │
│   │Memory │     │  Memory  │    │  Memory  │    │ Memory │     │
│   └───────┘     └──────────┘    └──────────┘    └────────┘     │
│    session         commit       consolidate       reflect       │
│     end              │               │               │          │
│                      │               │               │          │
│   ┌──────────────────┴───────────────┴───────────────┘          │
│   │                  Background Jobs                    │        │
│   │  ┌────────────┐ ┌────────────┐ ┌────────────┐      │        │
│   │  │ Consolidate│ │  Reflect   │ │   Forget   │      │        │
│   │  │  (daily)   │ │  (weekly)  │ │  (weekly)  │      │        │
│   │  └────────────┘ └────────────┘ └────────────┘      │        │
│   └─────────────────────────────────────────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Patterns

### Storing a Memory

```python
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

client = QdrantClient(url="https://your-cluster.cloud.qdrant.io", api_key="your-key")
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def embed(text: str) -> list[float]:
    return list(embedder.embed([text]))[0].tolist()

client.upsert(
    collection_name="memories",
    points=[PointStruct(
        id="memory-1",
        vector=embed("User prefers dark mode"),
        payload={"content": "User prefers dark mode", "type": "preference"}
    )]
)
```

### Searching Memories

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.query_points(
    collection_name="memories",
    query=embed("What are user preferences?"),
    query_filter=Filter(
        must=[FieldCondition(key="type", match=MatchValue(value="preference"))]
    ),
    limit=5
)
```

### Memory-Augmented Generation

```python
def chat_with_memory(user_message: str) -> str:
    # 1. Recall relevant memories
    memories = recall(user_message)

    # 2. Build context
    context = "\n".join(f"- {m['content']}" for m in memories)

    # 3. Generate response with context
    response = llm.chat(
        system=f"You have these memories:\n{context}",
        user=user_message
    )

    # 4. Extract and store new memories
    new_memories = extract_memories(user_message, response)
    for mem in new_memories:
        store(mem)

    return response
```

## Advanced: Memory Decay Formula

The advanced system uses this formula for effective importance:

```
effective_importance = base_importance × recency_factor × access_factor

where:
  recency_factor = 0.5^(days_since_access / half_life)  # half_life = 7 days
  access_factor = 1.0 + 0.1 × log(1 + access_count)
```

This means:
- Unused memories decay to 50% importance after 7 days
- Frequently accessed memories get boosted
- Combined with vector similarity for final ranking

## Advanced: Consolidation Strategy

```python
# Consolidation extracts durable knowledge from time-bound episodes
def consolidate():
    # 1. Find old, unconsolidated episodes
    episodes = get_episodes(older_than=1_day, consolidated=False)

    # 2. Use LLM to extract semantic memories
    for episode in episodes:
        facts = llm.extract_facts(episode.summary)
        preferences = llm.extract_preferences(episode.summary)

        for item in facts + preferences:
            store_semantic(item)

    # 3. Mark episodes as processed
    mark_consolidated(episodes)
```

## Production Tips

1. **Run consolidation on a schedule** (daily cron job)
2. **Set memory limits per user** to control costs
3. **Use forgetting aggressively** (threshold=0.2) to prevent bloat
4. **Cache meta-insights** in your prompt template
5. **Batch embedding calls** when storing multiple memories

## Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [FastEmbed](https://github.com/qdrant/fastembed)
- [OpenAI API](https://platform.openai.com/docs)
