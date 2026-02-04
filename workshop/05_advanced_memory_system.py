"""Workshop 05: Advanced Memory System"""

import json
import math
import uuid
from datetime import datetime, timedelta
from enum import Enum

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)
from rich import print as rprint
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from workshop.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    LLM_MODEL,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_URL,
)

WORKING_MEMORY = "adv_working_memory"
EPISODIC_MEMORY = "adv_episodic_memory"
SEMANTIC_MEMORY = "adv_semantic_memory"
META_MEMORY = "adv_meta_memory"


class MemoryTier(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    META = "meta"


class AdvancedMemorySystem:
    def __init__(self, agent_id: str = "default"):
        self.agent_id = agent_id
        self.session_id = str(uuid.uuid4())[:8]
        self.session_start = datetime.now()

        if QDRANT_API_KEY:
            self.qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            self.qdrant = QdrantClient(url=QDRANT_URL)

        self.embedder = TextEmbedding(model_name=EMBEDDING_MODEL)

        self.llm = None
        if OPENAI_API_KEY:
            from openai import OpenAI
            self.llm = OpenAI(api_key=OPENAI_API_KEY)

        self._setup_collections()
        self.working_buffer = []

    def _setup_collections(self):
        collections_config = [
            (WORKING_MEMORY, ["agent_id", "session_id"]),
            (EPISODIC_MEMORY, ["agent_id", "session_id", "importance", "consolidated"]),
            (SEMANTIC_MEMORY, ["agent_id", "type", "importance"]),
            (META_MEMORY, ["agent_id", "reflection_type"]),
        ]

        existing = {c.name for c in self.qdrant.get_collections().collections}

        for collection_name, index_fields in collections_config:
            if collection_name not in existing:
                self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                )
                for field in index_fields:
                    if field == "importance":
                        schema = PayloadSchemaType.FLOAT
                    elif field == "consolidated":
                        schema = PayloadSchemaType.BOOL
                    else:
                        schema = PayloadSchemaType.KEYWORD
                    self.qdrant.create_payload_index(
                        collection_name=collection_name,
                        field_name=field,
                        field_schema=schema,
                    )

    def _embed(self, text: str) -> list[float]:
        return list(self.embedder.embed([text]))[0].tolist()

    def _calculate_effective_importance(self, payload: dict) -> float:
        base_importance = payload.get("importance", 0.5)
        access_count = payload.get("access_count", 0)
        last_accessed_str = payload.get("last_accessed", datetime.now().isoformat())

        try:
            last_accessed = datetime.fromisoformat(last_accessed_str)
        except (ValueError, TypeError):
            last_accessed = datetime.now()

        days_since_access = (datetime.now() - last_accessed).total_seconds() / 86400
        half_life = 7.0
        recency_factor = math.pow(0.5, days_since_access / half_life)
        access_factor = 1.0 + 0.1 * math.log(1 + access_count)

        return min(1.0, base_importance * recency_factor * access_factor)

    def add_to_working(self, role: str, content: str) -> str:
        turn_id = str(uuid.uuid4())
        turn = {
            "id": turn_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        self.working_buffer.append(turn)

        self.qdrant.upsert(
            collection_name=WORKING_MEMORY,
            points=[PointStruct(
                id=turn_id,
                vector=self._embed(f"{role}: {content}"),
                payload={
                    "agent_id": self.agent_id,
                    "session_id": self.session_id,
                    **turn,
                },
            )],
        )
        return turn_id

    def get_working_context(self, limit: int = 10) -> list[dict]:
        return self.working_buffer[-limit:]

    def commit_session_to_episodic(self) -> str:
        if not self.working_buffer:
            return None

        conversation = "\n".join(
            f"{t['role']}: {t['content']}" for t in self.working_buffer
        )

        summary = f"Session with {len(self.working_buffer)} turns"
        importance = 0.5

        if self.llm:
            summary = self._generate_episode_summary(conversation)
            importance = self._assess_importance(conversation)

        episode_id = str(uuid.uuid4())

        self.qdrant.upsert(
            collection_name=EPISODIC_MEMORY,
            points=[PointStruct(
                id=episode_id,
                vector=self._embed(summary),
                payload={
                    "agent_id": self.agent_id,
                    "session_id": self.session_id,
                    "summary": summary,
                    "turn_count": len(self.working_buffer),
                    "full_conversation": conversation,
                    "importance": importance,
                    "access_count": 0,
                    "last_accessed": datetime.now().isoformat(),
                    "created_at": self.session_start.isoformat(),
                    "consolidated": False,
                },
            )],
        )

        self._clear_working_memory()
        self.working_buffer = []

        return episode_id

    def _generate_episode_summary(self, conversation: str) -> str:
        if not self.llm:
            return f"Session with {len(conversation.split(chr(10)))} turns"

        try:
            response = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize this conversation in 2-3 sentences. "
                        "Focus on: main topic, key decisions/conclusions, any user preferences revealed.",
                    },
                    {"role": "user", "content": conversation},
                ],
                max_tokens=150,
            )
            return response.choices[0].message.content
        except Exception:
            return f"Session with {len(conversation.split(chr(10)))} turns"

    def _assess_importance(self, content: str) -> float:
        if not self.llm:
            return 0.5

        try:
            response = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Rate the importance of remembering this conversation for future interactions. "
                        "Consider: user preferences, decisions made, technical details shared. "
                        "Return ONLY a number between 0.0 and 1.0.",
                    },
                    {"role": "user", "content": content[:2000]},
                ],
                max_tokens=10,
            )
            return float(response.choices[0].message.content.strip())
        except Exception:
            return 0.5

    def recall_episodes(self, query: str, limit: int = 5) -> list[dict]:
        results = self.qdrant.query_points(
            collection_name=EPISODIC_MEMORY,
            query=self._embed(query),
            query_filter=Filter(
                must=[FieldCondition(key="agent_id", match=MatchValue(value=self.agent_id))]
            ),
            limit=limit,
        )

        episodes = []
        for r in results.points:
            self.qdrant.set_payload(
                collection_name=EPISODIC_MEMORY,
                payload={
                    "access_count": r.payload.get("access_count", 0) + 1,
                    "last_accessed": datetime.now().isoformat(),
                },
                points=[r.id],
            )

            effective_importance = self._calculate_effective_importance(r.payload)
            episodes.append({
                "id": r.id,
                "summary": r.payload.get("summary"),
                "importance": r.payload.get("importance"),
                "effective_importance": effective_importance,
                "score": r.score,
                "created_at": r.payload.get("created_at"),
            })

        return episodes

    def store_semantic(
        self,
        content: str,
        memory_type: str,
        importance: float = 0.5,
        source_episode_id: str = None,
    ) -> str:
        memory_id = str(uuid.uuid4())

        self.qdrant.upsert(
            collection_name=SEMANTIC_MEMORY,
            points=[PointStruct(
                id=memory_id,
                vector=self._embed(content),
                payload={
                    "agent_id": self.agent_id,
                    "content": content,
                    "type": memory_type,
                    "importance": importance,
                    "access_count": 0,
                    "last_accessed": datetime.now().isoformat(),
                    "created_at": datetime.now().isoformat(),
                    "source_episode_id": source_episode_id,
                },
            )],
        )
        return memory_id

    def recall_semantic(
        self,
        query: str,
        memory_type: str = None,
        min_importance: float = 0.0,
        limit: int = 10,
    ) -> list[dict]:
        conditions = [
            FieldCondition(key="agent_id", match=MatchValue(value=self.agent_id))
        ]
        if memory_type:
            conditions.append(
                FieldCondition(key="type", match=MatchValue(value=memory_type))
            )

        results = self.qdrant.query_points(
            collection_name=SEMANTIC_MEMORY,
            query=self._embed(query),
            query_filter=Filter(must=conditions),
            limit=limit * 2,
        )

        memories = []
        for r in results.points:
            effective_importance = self._calculate_effective_importance(r.payload)
            if effective_importance >= min_importance:
                self.qdrant.set_payload(
                    collection_name=SEMANTIC_MEMORY,
                    payload={
                        "access_count": r.payload.get("access_count", 0) + 1,
                        "last_accessed": datetime.now().isoformat(),
                    },
                    points=[r.id],
                )

                memories.append({
                    "id": r.id,
                    "content": r.payload.get("content"),
                    "type": r.payload.get("type"),
                    "importance": r.payload.get("importance"),
                    "effective_importance": effective_importance,
                    "score": r.score,
                    "combined_score": r.score * effective_importance,
                })

        memories.sort(key=lambda x: x["combined_score"], reverse=True)
        return memories[:limit]

    def consolidate_episodes(self, max_episodes: int = 10) -> list[str]:
        if not self.llm:
            rprint("[yellow]Consolidation requires OPENAI_API_KEY[/yellow]")
            return []

        cutoff = (datetime.now() - timedelta(days=1)).isoformat()

        results, _ = self.qdrant.scroll(
            collection_name=EPISODIC_MEMORY,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="agent_id", match=MatchValue(value=self.agent_id)),
                    FieldCondition(key="consolidated", match=MatchValue(value=False)),
                ]
            ),
            limit=max_episodes,
        )

        if not results:
            return []

        episode_texts = []
        episode_ids = []
        for r in results:
            created = r.payload.get("created_at", datetime.now().isoformat())
            if created < cutoff:
                episode_texts.append(r.payload.get("summary", ""))
                episode_ids.append(r.id)

        if not episode_texts:
            return []

        new_memories = self._extract_semantic_from_episodes(episode_texts, episode_ids)

        for eid in episode_ids:
            self.qdrant.set_payload(
                collection_name=EPISODIC_MEMORY,
                payload={"consolidated": True},
                points=[eid],
            )

        return new_memories

    def _extract_semantic_from_episodes(
        self, episode_summaries: list[str], episode_ids: list[str]
    ) -> list[str]:
        if not self.llm:
            return []

        combined = "\n---\n".join(episode_summaries)

        try:
            response = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """Analyze these conversation summaries and extract durable knowledge.
Return a JSON array of memories. Each memory should have:
- "content": The fact, preference, or knowledge to remember
- "type": One of "fact", "preference", "skill", "relationship", "goal"
- "importance": 0.0-1.0 based on long-term value

Only extract useful information. Return ONLY the JSON array.""",
                    },
                    {"role": "user", "content": combined},
                ],
                max_tokens=1000,
            )

            memories = json.loads(response.choices[0].message.content)
            stored_ids = []

            for mem in memories:
                if isinstance(mem, dict) and "content" in mem:
                    mid = self.store_semantic(
                        content=mem["content"],
                        memory_type=mem.get("type", "fact"),
                        importance=mem.get("importance", 0.5),
                        source_episode_id=episode_ids[0] if episode_ids else None,
                    )
                    stored_ids.append(mid)

            return stored_ids
        except Exception:
            return []

    def reflect(self) -> dict:
        if not self.llm:
            return {"status": "llm_required", "insights": []}

        semantic_results, _ = self.qdrant.scroll(
            collection_name=SEMANTIC_MEMORY,
            scroll_filter=Filter(
                must=[FieldCondition(key="agent_id", match=MatchValue(value=self.agent_id))]
            ),
            limit=50,
        )

        if len(semantic_results) < 5:
            return {"status": "insufficient_data", "insights": []}

        by_type = {}
        for r in semantic_results:
            t = r.payload.get("type", "unknown")
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(r.payload.get("content", ""))

        reflection_prompt = "Based on these stored memories about the user, generate insights:\n\n"
        for t, contents in by_type.items():
            reflection_prompt += f"## {t.upper()}\n"
            for c in contents[:10]:
                reflection_prompt += f"- {c}\n"
            reflection_prompt += "\n"

        try:
            response = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are reflecting on what you know about a user.
Generate 3-5 high-level insights that synthesize patterns across memories.
Return ONLY a JSON array, no other text. Each object must have:
- "insight": The synthesized understanding
- "confidence": number 0.0-1.0
- "actionable": How this should influence future interactions

Example: [{"insight": "...", "confidence": 0.8, "actionable": "..."}]""",
                    },
                    {"role": "user", "content": reflection_prompt},
                ],
                max_tokens=500,
            )

            raw = (response.choices[0].message.content or "").strip()
            # Strip markdown code block if present (```json or ```)
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1] if "\n" in raw else raw[3:]
                if raw.startswith("json"):
                    raw = raw[4:].lstrip()
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0].rstrip()
            # Extract JSON array in case there's extra text
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]
            insights = json.loads(raw)

            for insight in insights:
                if isinstance(insight, dict) and "insight" in insight:
                    self.qdrant.upsert(
                        collection_name=META_MEMORY,
                        points=[PointStruct(
                            id=str(uuid.uuid4()),
                            vector=self._embed(insight["insight"]),
                            payload={
                                "agent_id": self.agent_id,
                                "reflection_type": "user_insight",
                                "insight": insight["insight"],
                                "confidence": insight.get("confidence", 0.5),
                                "actionable": insight.get("actionable", ""),
                                "created_at": datetime.now().isoformat(),
                            },
                        )],
                    )

            return {"status": "success", "insights": insights}
        except Exception as e:
            return {"status": "error", "error": str(e), "insights": []}

    def get_meta_insights(self, limit: int = 5) -> list[dict]:
        results, _ = self.qdrant.scroll(
            collection_name=META_MEMORY,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="agent_id", match=MatchValue(value=self.agent_id)),
                    FieldCondition(key="reflection_type", match=MatchValue(value="user_insight")),
                ]
            ),
            limit=limit,
        )

        return [
            {
                "insight": r.payload.get("insight"),
                "confidence": r.payload.get("confidence"),
                "actionable": r.payload.get("actionable"),
            }
            for r in results
        ]

    def forget_low_value_memories(self, threshold: float = 0.1) -> int:
        forgotten_count = 0

        for collection in [EPISODIC_MEMORY, SEMANTIC_MEMORY]:
            results, _ = self.qdrant.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="agent_id", match=MatchValue(value=self.agent_id))]
                ),
                limit=1000,
            )

            to_delete = []
            for r in results:
                effective = self._calculate_effective_importance(r.payload)
                if effective < threshold:
                    to_delete.append(r.id)

            if to_delete:
                self.qdrant.delete(
                    collection_name=collection,
                    points_selector=to_delete,
                )
                forgotten_count += len(to_delete)

        return forgotten_count

    def recall(self, query: str, include_meta: bool = True) -> dict:
        return {
            "working": self.get_working_context(limit=5),
            "episodic": self.recall_episodes(query, limit=3),
            "semantic": self.recall_semantic(query, limit=5),
            "meta": self.get_meta_insights(limit=3) if include_meta else [],
        }

    def format_context_for_prompt(self, context: dict) -> str:
        parts = []

        if context.get("meta"):
            parts.append("## Understanding of User")
            for m in context["meta"]:
                parts.append(f"- {m['insight']} (confidence: {m['confidence']:.1f})")

        if context.get("semantic"):
            parts.append("\n## Relevant Knowledge")
            for m in context["semantic"]:
                parts.append(f"- [{m['type']}] {m['content']}")

        if context.get("episodic"):
            parts.append("\n## Relevant Past Conversations")
            for e in context["episodic"]:
                parts.append(f"- {e['summary']}")

        return "\n".join(parts) if parts else ""

    def chat(self, user_message: str) -> str:
        self.add_to_working("user", user_message)
        context = self.recall(user_message)
        context_str = self.format_context_for_prompt(context)

        if not self.llm:
            return f"[LLM not configured]\n\nRelevant context:\n{context_str}" if context_str else "[LLM not configured - add OPENAI_API_KEY for chat]"

        system_prompt = """You are a helpful assistant with sophisticated memory capabilities.
You remember facts, preferences, and past conversations with users.
Use your memories to provide personalized, contextual responses.
Be concise but thorough."""

        if context_str:
            system_prompt += f"\n\n{context_str}"

        messages = [{"role": "system", "content": system_prompt}]
        for turn in self.get_working_context(limit=10):
            messages.append({"role": turn["role"], "content": turn["content"]})

        try:
            response = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=500,
            )
            assistant_message = response.choices[0].message.content
        except Exception as e:
            assistant_message = f"Error: {e}"

        self.add_to_working("assistant", assistant_message)
        return assistant_message

    def _clear_working_memory(self):
        results, _ = self.qdrant.scroll(
            collection_name=WORKING_MEMORY,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="agent_id", match=MatchValue(value=self.agent_id)),
                    FieldCondition(key="session_id", match=MatchValue(value=self.session_id)),
                ]
            ),
            limit=1000,
        )
        if results:
            self.qdrant.delete(
                collection_name=WORKING_MEMORY,
                points_selector=[r.id for r in results],
            )

    def clear_all(self):
        for collection in [WORKING_MEMORY, EPISODIC_MEMORY, SEMANTIC_MEMORY, META_MEMORY]:
            results, _ = self.qdrant.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="agent_id", match=MatchValue(value=self.agent_id))]
                ),
                limit=10000,
            )
            if results:
                self.qdrant.delete(
                    collection_name=collection,
                    points_selector=[r.id for r in results],
                )


def show_memory_stats(mem: AdvancedMemorySystem):
    table = Table(title="Memory Statistics")
    table.add_column("Tier")
    table.add_column("Count")
    table.add_column("Description")

    for collection, desc in [
        (WORKING_MEMORY, "Current session turns"),
        (EPISODIC_MEMORY, "Past conversation sessions"),
        (SEMANTIC_MEMORY, "Consolidated knowledge"),
        (META_MEMORY, "Reflections and insights"),
    ]:
        count_results, _ = mem.qdrant.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="agent_id", match=MatchValue(value=mem.agent_id))]
            ),
            limit=10000,
        )
        table.add_row(collection.split("_")[-2], str(len(count_results)), desc)

    rprint(table)


# Shared seed data for demo and interactive mode
DEMO_SEED_DATA = [
    # Facts about user
    ("User is a Python developer working on ML projects", "fact", 0.8),
    ("User's project uses FastAPI and Qdrant", "fact", 0.75),
    ("User works at a fintech startup in NYC", "fact", 0.6),
    ("User has 5 years of experience in backend development", "fact", 0.7),
    ("User's team follows trunk-based development", "fact", 0.55),
    ("The codebase has 80% test coverage", "fact", 0.5),
    ("Production runs on AWS with EKS", "fact", 0.65),
    ("The API serves 50k daily active users", "fact", 0.7),
    # Preferences
    ("User prefers functional programming patterns", "preference", 0.9),
    ("User dislikes overly verbose explanations", "preference", 0.7),
    ("User wants type hints on all functions", "preference", 0.85),
    ("User prefers composition over inheritance", "preference", 0.8),
    ("User likes early returns over nested conditionals", "preference", 0.75),
    ("User prefers pytest over unittest", "preference", 0.7),
    ("User wants code examples more than theory", "preference", 0.85),
    ("User dislikes unnecessary abstractions", "preference", 0.8),
    # Goals
    ("User is building a recommendation system for e-commerce", "goal", 0.85),
    ("User wants to reduce API latency to under 50ms", "goal", 0.8),
    ("User plans to implement real-time personalization", "goal", 0.75),
    ("User needs to scale to 100k concurrent users", "goal", 0.7),
    ("User wants to add A/B testing for recommendations", "goal", 0.65),
    # Skills
    ("User knows PyTorch and has trained custom models", "skill", 0.8),
    ("User is proficient with Docker and Kubernetes", "skill", 0.75),
    ("User can write complex SQL queries", "skill", 0.7),
    ("User has experience with Kafka for event streaming", "skill", 0.65),
    ("User knows how to use Qdrant for vector search", "skill", 0.85),
    # Relationships
    ("User reports to Sarah, the engineering director", "relationship", 0.5),
    ("User mentors two junior developers", "relationship", 0.55),
    ("User collaborates closely with the ML team", "relationship", 0.6),
]

# Past conversation episodes (summary, full_conversation, importance)
DEMO_EPISODES = [
    (
        "User asked about structuring a recommendation API and handling cold-start users. Discussed RESTful design and content-based vs collaborative filtering.",
        "user: How should I structure my recommendation API?\nassistant: Use a RESTful design with endpoints for recommendations, user profiles, and item catalogs.\nuser: What about handling cold-start users?\nassistant: Use content-based filtering initially, then transition to collaborative filtering as you gather data.",
        0.7,
    ),
    (
        "User discussed reducing API latency and deployment on EKS. Talked about connection pooling and caching strategies.",
        "user: How can I get my API under 50ms?\nassistant: Focus on connection pooling, query optimization, and caching hot paths.\nuser: We're on EKS.\nassistant: Use node-local caching and consider read replicas for the database.",
        0.6,
    ),
]

# Meta insights (insight, confidence, actionable)
DEMO_META_INSIGHTS = [
    (
        "User is a practical engineer who values code examples and concrete advice over long explanations.",
        0.85,
        "Keep responses concise with runnable examples when possible.",
    ),
    (
        "User is building production ML systems and cares about latency, scale, and deployment.",
        0.8,
        "Prioritize production-ready patterns and mention tradeoffs.",
    ),
]


def _seed_semantic_memories(mem: AdvancedMemorySystem) -> int:
    """Seed semantic memory with demo data. Returns count stored."""
    for content, mtype, importance in DEMO_SEED_DATA:
        mem.store_semantic(content, mtype, importance)
    return len(DEMO_SEED_DATA)


def _seed_episodic(mem: AdvancedMemorySystem) -> int:
    """Seed episodic memory with past session summaries. Returns count stored."""
    now = datetime.now()
    for i, (summary, full_conversation, importance) in enumerate(DEMO_EPISODES):
        session_id = f"seed-session-{i + 1}"
        created = (now - timedelta(days=2 + i)).isoformat()
        mem.qdrant.upsert(
            collection_name=EPISODIC_MEMORY,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=mem._embed(summary),
                payload={
                    "agent_id": mem.agent_id,
                    "session_id": session_id,
                    "summary": summary,
                    "turn_count": full_conversation.count("\n") + 1,
                    "full_conversation": full_conversation,
                    "importance": importance,
                    "access_count": 0,
                    "last_accessed": now.isoformat(),
                    "created_at": created,
                    "consolidated": False,
                },
            )],
        )
    return len(DEMO_EPISODES)


def _seed_meta(mem: AdvancedMemorySystem) -> int:
    """Seed meta memory with demo insights. Returns count stored."""
    for insight, confidence, actionable in DEMO_META_INSIGHTS:
        mem.qdrant.upsert(
            collection_name=META_MEMORY,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=mem._embed(insight),
                payload={
                    "agent_id": mem.agent_id,
                    "reflection_type": "user_insight",
                    "insight": insight,
                    "confidence": confidence,
                    "actionable": actionable,
                    "created_at": datetime.now().isoformat(),
                },
            )],
        )
    return len(DEMO_META_INSIGHTS)


def demo_memory_lifecycle():
    rprint(Panel("Workshop 05: Advanced Memory System Demo — Tiers, decay, consolidation & reflection", style="bold magenta"))

    mem = AdvancedMemorySystem(agent_id="demo-advanced")
    mem.clear_all()

    rprint("\n[bold]1. Seeding initial knowledge...[/bold]")
    n = _seed_semantic_memories(mem)
    rprint(f"  Stored {n} semantic memories across {len(set(m[1] for m in DEMO_SEED_DATA))} types")

    rprint("\n[bold]2. Testing memory recall...[/bold]")
    queries = [
        "What programming language does the user know?",
        "What is the user building?",
        "How should I format responses?",
    ]
    for query in queries:
        rprint(f"\n[cyan]Query:[/cyan] {query}")
        results = mem.recall_semantic(query, limit=2)
        for m in results:
            rprint(f"  [score={m['score']:.3f}] [{m['type']}] {m['content']}")

    rprint("\n[bold]3. Adding conversation to working memory...[/bold]")
    mem.add_to_working("user", "How should I structure my recommendation API?")
    mem.add_to_working("assistant", "Use a RESTful design with endpoints for recommendations, user profiles, and item catalogs.")
    mem.add_to_working("user", "What about handling cold-start users?")
    mem.add_to_working("assistant", "Use content-based filtering initially, then transition to collaborative filtering as you gather data.")
    rprint(f"  Added {len(mem.working_buffer)} turns to working memory")

    rprint("\n[bold]4. Committing session to episodic memory...[/bold]")
    episode_id = mem.commit_session_to_episodic()
    rprint(f"  Created episode: {episode_id}")

    rprint("\n[bold]5. Memory statistics:[/bold]")
    show_memory_stats(mem)

    if mem.llm:
        rprint("\n[bold]6. Running reflection cycle...[/bold]")
        reflection = mem.reflect()
        if reflection["status"] == "success":
            for insight in reflection["insights"]:
                if isinstance(insight, dict):
                    rprint(f"  [yellow]Insight:[/yellow] {insight.get('insight', 'N/A')}")
    else:
        rprint("\n[yellow]Skipping reflection - add OPENAI_API_KEY for LLM features[/yellow]")

    rprint("\n[bold]7. Unified recall demo:[/bold]")
    query = "How do I improve my recommendation system?"
    context = mem.recall(query)
    rprint(f"  Query: '{query}'")
    rprint(f"  Semantic memories found: {len(context['semantic'])}")
    rprint(f"  Episodes found: {len(context['episodic'])}")
    rprint(f"  Meta insights: {len(context['meta'])}")


def interactive_mode():
    rprint(Panel("Workshop 05: Advanced Memory System — Interactive chat & commands", style="bold magenta"))

    agent_id = Prompt.ask("Agent ID", default="my-agent")
    mem = AdvancedMemorySystem(agent_id=agent_id)

    seed_choice = Prompt.ask(
        "Seed demo memories so all tiers have data?",
        choices=["y", "n"],
        default="y",
    )
    if seed_choice == "y":
        n_sem = _seed_semantic_memories(mem)
        n_ep = _seed_episodic(mem)
        n_meta = _seed_meta(mem)
        rprint(f"[green]Seeded: {n_sem} semantic, {n_ep} episodic, {n_meta} meta. Use /stats to see counts.[/green]")

    rprint(f"\n[green]Started agent '{agent_id}' (session: {mem.session_id})[/green]")
    rprint("Commands: /stats, /recall <query>, /reflect, /consolidate, /forget, /clear, /quit")

    if not mem.llm:
        rprint("[yellow]Note: Chat/reflection/consolidation require OPENAI_API_KEY[/yellow]")

    rprint("")

    while True:
        try:
            user_input = Prompt.ask("[cyan]You[/cyan]")
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input.strip():
            continue

        if user_input.lower() == "/quit":
            mem.commit_session_to_episodic()
            break

        elif user_input.lower() == "/stats":
            show_memory_stats(mem)
            continue

        elif user_input.lower().startswith("/recall "):
            query = user_input[8:]
            context = mem.recall(query)
            rprint(f"\n[bold]Recall for '{query}':[/bold]")
            for tier, items in context.items():
                if items:
                    rprint(f"  [yellow]{tier}:[/yellow] {len(items)} items")
                    for item in items[:2]:
                        content = item.get("content") or item.get("summary") or item.get("insight") or str(item)
                        rprint(f"    - {str(content)[:80]}...")
            continue

        elif user_input.lower() == "/reflect":
            rprint("\n[bold]Running reflection...[/bold]")
            result = mem.reflect()
            if result["status"] == "success":
                for insight in result["insights"]:
                    if isinstance(insight, dict):
                        rprint(f"  {insight.get('insight', 'N/A')}")
            else:
                msg = result.get("error") or result.get("status", "Unknown")
                rprint(f"  [red]{msg}[/red]")
            continue

        elif user_input.lower() == "/consolidate":
            rprint("\n[bold]Running consolidation...[/bold]")
            new_ids = mem.consolidate_episodes()
            rprint(f"  Created {len(new_ids)} new semantic memories")
            continue

        elif user_input.lower() == "/forget":
            rprint("\n[bold]Forgetting low-value memories...[/bold]")
            count = mem.forget_low_value_memories(threshold=0.1)
            rprint(f"  Forgotten {count} memories")
            continue

        elif user_input.lower() == "/clear":
            mem.clear_all()
            rprint("[yellow]All memories cleared[/yellow]")
            continue

        response = mem.chat(user_input)
        rprint(f"[green]Agent:[/green] {response}\n")

    rprint("\n[yellow]Session ended[/yellow]")


def main():
    mode = Prompt.ask(
        "Select mode",
        choices=["demo", "interactive"],
        default="demo",
    )

    if mode == "demo":
        demo_memory_lifecycle()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
