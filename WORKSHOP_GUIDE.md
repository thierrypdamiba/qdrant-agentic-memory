# Qdrant Agentic Memory Workshop - Detailed Guide

This guide explains each workshop module in detail and provides a script for presenting the workshop.

---

## Overview: The Memory Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    Human Memory Analogy                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Working Memory     =  Current conversation context             │
│   Episodic Memory    =  "I remember we talked about X last week" │
│   Semantic Memory    =  "I know the user prefers Python"         │
│   Meta Memory        =  "The user seems to be a visual learner"  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module 01: Basic Memory Storage

### What It Teaches
The fundamentals of storing and retrieving memories using vector embeddings.

### Key Concepts

**1. Vector Embeddings**
Text is converted into numerical vectors (arrays of floats) that capture semantic meaning. Similar texts produce similar vectors.

```python
# How embedding works
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
embedding = list(embedder.embed(["User prefers Python"]))[0].tolist()
# Returns: [0.023, -0.156, 0.089, ...] (384 dimensions)
```

**2. Creating Collections**
A collection is like a database table for vectors.

```python
client.create_collection(
    collection_name="memories",
    vectors_config=VectorParams(
        size=384,              # Must match embedding dimensions
        distance=Distance.COSINE  # How similarity is measured
    ),
)
```

**3. Storing Memories**
Each memory is a "point" with:
- A unique ID
- A vector (the embedding)
- A payload (metadata like content, timestamp, category)

```python
client.upsert(
    collection_name="memories",
    points=[PointStruct(
        id="memory-1",
        vector=embed("User prefers dark mode"),
        payload={"content": "User prefers dark mode", "category": "preference"}
    )]
)
```

**4. Searching Memories**
Query with a text, find semantically similar memories:

```python
results = client.query_points(
    collection_name="memories",
    query=embed("What are the user's preferences?"),
    limit=5
)
# Returns memories ranked by cosine similarity
```

### Demo Flow
1. Creates 130 example memories (preferences, tasks, context, technical, skills)
2. Stores each memory with its embedding and metadata
3. Runs 3 semantic searches to show how similar content is found

### Key Takeaway
You don't need exact keyword matches - "What programming language?" finds "User's favorite programming language is Python" through semantic similarity.

---

## Module 02: Semantic Memory System

### What It Teaches
How to categorize memories by type and importance, with filtered retrieval.

### Key Concepts

**1. Memory Types**
Memories are categorized:
- **FACT**: Objective information ("Python uses indentation")
- **SKILL**: How-to knowledge ("Run 'make deploy' to deploy")
- **PREFERENCE**: User preferences ("User prefers functional programming")
- **CONTEXT**: Situational information ("User works at a fintech startup")

**2. Payload Indexing**
Create indexes for faster filtering:

```python
client.create_payload_index(
    collection_name="semantic_memories",
    field_name="type",
    field_schema=PayloadSchemaType.KEYWORD,  # For exact matches
)
client.create_payload_index(
    collection_name="semantic_memories",
    field_name="importance",
    field_schema=PayloadSchemaType.FLOAT,    # For range queries
)
```

**3. Filtered Queries**
Combine semantic search with metadata filters:

```python
results = client.query_points(
    collection_name="semantic_memories",
    query=embed("What does the user like?"),
    query_filter=Filter(
        must=[
            FieldCondition(key="type", match=MatchValue(value="preference")),
            FieldCondition(key="importance", range=Range(gte=0.7))
        ]
    ),
    limit=5
)
```

**4. Access Tracking**
Track when memories are accessed to understand which are useful:

```python
def _update_access(self, memory_id, payload):
    self.client.set_payload(
        collection_name=SEMANTIC_COLLECTION,
        payload={
            "access_count": payload.get("access_count", 0) + 1,
            "last_accessed": datetime.now().isoformat(),
        },
        points=[memory_id],
    )
```

### Demo Flow
1. Stores 140 memories across 4 types with varying importance
2. Shows unfiltered recall
3. Shows type-filtered recall (preferences only)
4. Shows importance-filtered recall (>= 0.7)
5. Displays all high-importance memories in a table

### Key Takeaway
Filtering lets you retrieve contextually appropriate memories - don't show technical facts when asked about preferences.

---

## Module 03: Episodic Memory

### What It Teaches
How to store and retrieve conversation history across sessions.

### Key Concepts

**1. Sessions & Turns**
Conversations are organized by:
- **Session**: A complete conversation (identified by session_id)
- **Turn**: A single message in the conversation (user or assistant)

```python
def store_turn(self, session_id, role, content, turn_number):
    turn_text = f"{role}: {content}"  # Combined for embedding
    payload = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "turn_number": turn_number,
        "timestamp": datetime.now().isoformat(),
    }
    # Store with embedding of the turn
```

**2. Cross-Session Search**
Find relevant past conversations regardless of which session they're from:

```python
def recall_relevant(self, query, session_id=None, limit=5):
    # If session_id provided, only search that session
    # Otherwise, search all sessions
    results = client.query_points(
        collection_name=EPISODES_COLLECTION,
        query=embed(query),
        query_filter=search_filter,
        limit=limit,
    )
```

**3. Session Reconstruction**
Get complete conversation history for a session:

```python
def get_session_history(self, session_id):
    results, _ = client.scroll(
        collection_name=EPISODES_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
        ),
        limit=1000,
    )
    return sorted(turns, key=lambda x: x["turn_number"])
```

**4. Session Summarization (LLM)**
Compress conversations into summaries:

```python
def summarize_session(self, session_id):
    history = self.get_session_history(session_id)
    conversation_text = "\n".join(f"{t['role']}: {t['content']}" for t in history)
    
    response = self.llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Summarize in 2-3 sentences..."},
            {"role": "user", "content": conversation_text},
        ],
    )
```

### Demo Flow
1. Stores 25 different conversations covering Python, Qdrant, deployment, FastAPI, etc.
2. Searches across all conversations for relevant turns
3. Reconstructs a specific session's history
4. Shows session summarization (if LLM configured)

### Key Takeaway
Episodic memory lets the agent say "I remember we talked about Qdrant filters last week" - it provides continuity across sessions.

---

## Module 04: Memory Agent

### What It Teaches
How to combine memory retrieval with LLM response generation.

### Key Concepts

**1. Memory-Augmented Context**
Before responding, recall relevant memories:

```python
def _build_context(self, user_message):
    memories = self.recall(user_message, limit=5)
    
    context_parts = ["Relevant memories:"]
    for m in memories:
        if m["score"] > 0.3:  # Only include relevant memories
            context_parts.append(f"- [{m['type']}] {m['content']}")
    
    return "\n".join(context_parts)
```

**2. Context Injection**
Add memories to the system prompt:

```python
def chat(self, user_message):
    memory_context = self._build_context(user_message)
    
    system_prompt = "You are a helpful assistant with memory capabilities..."
    if memory_context:
        system_prompt += f"\n\n{memory_context}"
    
    # Then call LLM with this enhanced prompt
```

**3. Conversation History**
Track current conversation:

```python
self.conversation_history.append({"role": "user", "content": user_message})
# ... generate response ...
self.conversation_history.append({"role": "assistant", "content": assistant_message})

# Only use recent history to avoid token limits
messages.extend(self.conversation_history[-10:])
```

### Demo Flow
1. Creates agent with seeded memories (preferences, context, facts, goals, skills)
2. Tests memory recall with sample queries
3. Shows chat with memory (if LLM configured)

### Key Takeaway
This is the pattern for memory-augmented generation: recall → inject → generate.

---

## Module 05: Advanced Memory System

### What It Teaches
Production-grade memory with multiple tiers, decay, consolidation, and reflection.

### Key Concepts

**1. Four Memory Tiers**

| Tier | Collection | Purpose | Lifetime |
|------|------------|---------|----------|
| Working | `adv_working_memory` | Current session buffer | Session |
| Episodic | `adv_episodic_memory` | Past conversations | Days-weeks |
| Semantic | `adv_semantic_memory` | Consolidated facts | Persistent |
| Meta | `adv_meta_memory` | Insights about user | Persistent |

**2. Memory Decay**
Memories fade over time unless reinforced:

```python
def _calculate_effective_importance(self, payload):
    base_importance = payload.get("importance", 0.5)
    access_count = payload.get("access_count", 0)
    days_since_access = (datetime.now() - last_accessed).total_seconds() / 86400
    
    # Decay with 7-day half-life
    half_life = 7.0
    recency_factor = math.pow(0.5, days_since_access / half_life)
    
    # Boost for frequently accessed memories
    access_factor = 1.0 + 0.1 * math.log(1 + access_count)
    
    return min(1.0, base_importance * recency_factor * access_factor)
```

**3. Session Lifecycle**

```
Working Memory → commit → Episodic Memory → consolidate → Semantic Memory
                                                    ↓
                                              Meta Memory ← reflect
```

```python
# During conversation
mem.add_to_working("user", message)
mem.add_to_working("assistant", response)

# At session end
mem.commit_session_to_episodic()  # Summarizes and stores

# Later (daily job)
mem.consolidate_episodes()  # Extracts facts, preferences from episodes

# Periodically (weekly)
mem.reflect()  # Generates high-level insights
mem.forget_low_value_memories()  # Cleans up decayed memories
```

**4. Consolidation**
Extract durable knowledge from conversations:

```python
def consolidate_episodes(self):
    # Get old, unconsolidated episodes
    episodes = get_episodes(older_than=1_day, consolidated=False)
    
    # LLM extracts structured memories
    response = llm.chat.completions.create(
        messages=[{
            "role": "system",
            "content": """Extract durable knowledge as JSON:
            [{"content": "...", "type": "fact|preference|skill", "importance": 0.8}]"""
        }, {
            "role": "user", 
            "content": episode_summaries
        }]
    )
    
    # Store as semantic memories
    for mem in json.loads(response):
        self.store_semantic(mem["content"], mem["type"], mem["importance"])
    
    # Mark episodes as processed
    mark_consolidated(episodes)
```

**5. Reflection**
Generate meta-insights about the user:

```python
def reflect(self):
    # Gather semantic memories by type
    by_type = {"fact": [...], "preference": [...], ...}
    
    # LLM synthesizes patterns
    response = llm.chat.completions.create(
        messages=[{
            "role": "system",
            "content": """Generate 3-5 high-level insights:
            [{"insight": "...", "confidence": 0.8, "actionable": "..."}]"""
        }, {
            "role": "user",
            "content": formatted_memories
        }]
    )
    
    # Store in meta memory
    for insight in json.loads(response):
        store_meta(insight)
```

**6. Unified Recall**
Query all tiers at once:

```python
def recall(self, query):
    return {
        "working": self.get_working_context(limit=5),
        "episodic": self.recall_episodes(query, limit=3),
        "semantic": self.recall_semantic(query, limit=5),
        "meta": self.get_meta_insights(limit=3),
    }
```

**7. Forgetting**
Remove memories that have decayed below threshold:

```python
def forget_low_value_memories(self, threshold=0.1):
    for collection in [EPISODIC_MEMORY, SEMANTIC_MEMORY]:
        for memory in scroll(collection):
            effective = _calculate_effective_importance(memory.payload)
            if effective < threshold:
                delete(memory.id)
```

### Demo Flow
1. Seeds 28 semantic memories (facts, preferences, goals, skills, relationships)
2. Tests semantic recall
3. Adds conversation to working memory
4. Commits to episodic memory
5. Shows memory statistics
6. Runs reflection (if LLM)
7. Shows unified recall

### Interactive Commands
```
/stats        - Show count per tier
/recall <q>   - Query all tiers
/reflect      - Generate insights
/consolidate  - Episodes → Semantic
/forget       - Clean up decayed memories
/clear        - Reset everything
```

### Key Takeaway
This is a production-ready architecture. The key innovations are:
- **Decay**: Prevents unbounded memory growth
- **Consolidation**: Extracts durable knowledge from time-bound episodes
- **Reflection**: Synthesizes meta-level understanding
- **Unified recall**: Single query across all memory types

---

## Workshop Presentation Script

### Timing: ~90 minutes total

### Introduction (10 min)

**Talking Points:**
- "Traditional chatbots have no memory - every conversation starts fresh"
- "Memory-augmented agents remember context, preferences, and past interactions"
- "Today we'll build increasingly sophisticated memory systems"
- "Core memory features are FREE - embeddings run locally"

**Show architecture diagram from README**

### Module 01: Basics (15 min)

**Run:**
```bash
python -m workshop.01_basics
```

**Key Points to Highlight:**
1. FastEmbed downloads model on first run (~50MB, one-time)
2. Watch how semantic search finds relevant memories without exact keywords
3. Score values show similarity (1.0 = identical, 0.0 = unrelated)

**Discussion Questions:**
- "Notice how 'What are the user's preferences?' finds preference memories?"
- "What happens if we store contradictory information?"

### Module 02: Semantic Memory (20 min)

**Run:**
```bash
python -m workshop.02_semantic_memory
```

**Key Points to Highlight:**
1. Memory types enable focused retrieval
2. Importance scoring for prioritization
3. Payload indexes make filtering fast
4. Access tracking shows which memories are actually used

**Discussion Questions:**
- "When would you filter by type vs. let semantic search decide?"
- "How would you set importance scores in a real system?"

### Module 03: Episodic Memory (15 min)

**Run:**
```bash
python -m workshop.03_episodic_memory
```

**Key Points to Highlight:**
1. 25 conversations stored - notice how search finds across all of them
2. Session reconstruction maintains turn order
3. Summarization requires LLM but core storage is free

**Discussion Questions:**
- "How long should you keep episodic memories?"
- "What's the trade-off between storing full conversations vs. summaries?"

### Module 04: Memory Agent (15 min)

**Run:**
```bash
python -m workshop.04_memory_agent --mode demo
```

**Key Points to Highlight:**
1. This combines everything: store, recall, inject into prompt
2. Memory context is added to system prompt
3. Works without LLM for memory features

**Live Demo (if time permits):**
```bash
python -m workshop.04_memory_agent --mode interactive
```
- Store a memory manually: `/remember User's favorite color is blue`
- Chat about it: "What's my favorite color?"
- Show memories: `/memories`

### Module 05: Advanced System (15 min)

**Run:**
```bash
python -m workshop.05_advanced_memory_system
```
(Select "demo" mode)

**Key Points to Highlight:**
1. Four-tier architecture mirrors human memory
2. Decay formula - explain the math briefly
3. Consolidation extracts knowledge from conversations
4. Reflection generates meta-insights

**Show the memory flow diagram:**
```
Working → Episodic → Semantic
              ↓
            Meta
```

**Discussion Questions:**
- "How would you schedule consolidation in production?"
- "What threshold would you use for forgetting?"

### Q&A and Next Steps (10 min)

**Suggested Extensions:**
1. Add memory linking (graph relationships between memories)
2. Implement memory search with re-ranking
3. Build a memory dashboard/UI
4. Add multi-user support with tenant isolation
5. Implement memory export/import for backup

**Resources:**
- Qdrant Documentation: https://qdrant.tech/documentation/
- FastEmbed: https://github.com/qdrant/fastembed
- Workshop repo: (your repo URL)

---

## Common Questions

**Q: How many memories can I store?**
A: Qdrant free tier allows 1GB storage. With 384-dim vectors, that's roughly 500K memories.

**Q: Does this work offline?**
A: Embeddings work offline (FastEmbed). Chat/reflection need OpenAI API.

**Q: How do I use different embedding models?**
A: Change `EMBEDDING_MODEL` in config.py. Update `EMBEDDING_DIM` to match.

**Q: Can I use local LLMs instead of OpenAI?**
A: Yes! Use Ollama or vLLM and point to their OpenAI-compatible endpoint.

**Q: How do I handle multiple users?**
A: Use `agent_id` (already implemented) as a user identifier. Each user's memories are isolated.

**Q: What about privacy/GDPR?**
A: Implement a `delete_all_user_data(agent_id)` function. The `clear_all()` method shows the pattern.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Connection refused" | Check QDRANT_URL in .env matches your cluster |
| "API key invalid" | Copy API key from Qdrant Cloud dashboard |
| "Model not found" | FastEmbed downloads on first run - wait for it |
| "OpenAI rate limit" | Add retry logic or reduce request frequency |
| "Out of memory" | Use smaller embedding model or batch operations |
