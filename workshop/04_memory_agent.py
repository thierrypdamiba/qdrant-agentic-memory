"""Workshop 04: Complete Memory-Augmented Agent"""

import argparse
import uuid
from datetime import datetime

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from rich import print as rprint
from rich.panel import Panel
from rich.prompt import Prompt

from workshop.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    LLM_MODEL,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_URL,
)

AGENT_MEMORIES = "agent_unified_memories"


class MemoryAgent:
    def __init__(self, agent_id: str = "default"):
        self.agent_id = agent_id
        self.session_id = str(uuid.uuid4())[:8]

        if QDRANT_API_KEY:
            self.qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            self.qdrant = QdrantClient(url=QDRANT_URL)
        self.embedder = TextEmbedding(model_name=EMBEDDING_MODEL)

        self.llm = None
        if OPENAI_API_KEY:
            from openai import OpenAI
            self.llm = OpenAI(api_key=OPENAI_API_KEY)

        self._setup_collection()
        self.conversation_history = []

    def _setup_collection(self):
        collections = self.qdrant.get_collections().collections
        exists = any(c.name == AGENT_MEMORIES for c in collections)

        if not exists:
            self.qdrant.create_collection(
                collection_name=AGENT_MEMORIES,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            self.qdrant.create_payload_index(
                collection_name=AGENT_MEMORIES,
                field_name="agent_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )

    def _embed(self, text: str) -> list[float]:
        return list(self.embedder.embed([text]))[0].tolist()

    def remember(
        self,
        content: str,
        memory_type: str = "general",
        importance: float = 0.5,
    ) -> str:
        memory_id = str(uuid.uuid4())

        payload = {
            "agent_id": self.agent_id,
            "content": content,
            "type": memory_type,
            "importance": importance,
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
        }

        self.qdrant.upsert(
            collection_name=AGENT_MEMORIES,
            points=[PointStruct(id=memory_id, vector=self._embed(content), payload=payload)],
        )

        return memory_id

    def recall(self, query: str, limit: int = 5) -> list[dict]:
        results = self.qdrant.query_points(
            collection_name=AGENT_MEMORIES,
            query=self._embed(query),
            query_filter=Filter(
                must=[FieldCondition(key="agent_id", match=MatchValue(value=self.agent_id))]
            ),
            limit=limit,
        )

        return [
            {
                "content": r.payload["content"],
                "type": r.payload["type"],
                "importance": r.payload["importance"],
                "score": r.score,
            }
            for r in results.points
        ]

    def _build_context(self, user_message: str) -> str:
        memories = self.recall(user_message, limit=5)

        if not memories:
            return ""

        context_parts = ["Relevant memories:"]
        for m in memories:
            if m["score"] > 0.3:
                context_parts.append(f"- [{m['type']}] {m['content']}")

        return "\n".join(context_parts)

    def chat(self, user_message: str) -> str:
        if not self.llm:
            self.conversation_history.append({"role": "user", "content": user_message})
            context = self._build_context(user_message)
            return f"[LLM not configured - add OPENAI_API_KEY for chat]\n\nRelevant memories found:\n{context}" if context else "[LLM not configured - add OPENAI_API_KEY for chat]"

        self.conversation_history.append({"role": "user", "content": user_message})
        memory_context = self._build_context(user_message)

        system_prompt = """You are a helpful assistant with memory capabilities.
You remember information from past conversations and use it to provide better responses.
Be concise and helpful."""

        if memory_context:
            system_prompt += f"\n\n{memory_context}"

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history[-10:])

        try:
            response = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=500,
            )
            assistant_message = response.choices[0].message.content
        except Exception as e:
            assistant_message = f"Error generating response: {e}"

        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    def show_memories(self):
        results, _ = self.qdrant.scroll(
            collection_name=AGENT_MEMORIES,
            scroll_filter=Filter(
                must=[FieldCondition(key="agent_id", match=MatchValue(value=self.agent_id))]
            ),
            limit=100,
        )

        rprint(f"\n[bold]Memories for agent '{self.agent_id}':[/bold]")
        if not results:
            rprint("  No memories stored yet.")
        for r in results:
            rprint(
                f"  [{r.payload['type']}] (importance={r.payload['importance']:.1f}) "
                f"{r.payload['content'][:60]}..."
            )

    def clear_memories(self):
        results, _ = self.qdrant.scroll(
            collection_name=AGENT_MEMORIES,
            scroll_filter=Filter(
                must=[FieldCondition(key="agent_id", match=MatchValue(value=self.agent_id))]
            ),
            limit=10000,
        )

        if results:
            point_ids = [r.id for r in results]
            self.qdrant.delete(
                collection_name=AGENT_MEMORIES,
                points_selector=point_ids,
            )


def demo_mode():
    rprint(Panel("Workshop 04: Memory Agent Demo — Seeded memories, recall & chat", style="bold blue"))

    agent = MemoryAgent(agent_id="demo-agent")
    agent.clear_memories()

    rprint("\n[bold]Seeding initial memories...[/bold]")
    seed_memories = [
        # Preferences
        ("User prefers Python over JavaScript", "preference", 0.8),
        ("User likes concise code examples", "preference", 0.6),
        ("User prefers functional programming patterns", "preference", 0.75),
        ("User wants type hints in all code", "preference", 0.8),
        ("User dislikes verbose explanations", "preference", 0.7),
        ("User prefers async/await over threads", "preference", 0.65),
        # Context
        ("User is working on a recommendation system project", "context", 0.7),
        ("User's company is an e-commerce startup", "context", 0.6),
        ("The project uses FastAPI and Qdrant", "context", 0.75),
        ("User's team has 5 backend engineers", "context", 0.5),
        ("Production environment runs on AWS EKS", "context", 0.6),
        # Technical facts
        ("The API handles 10k requests per second", "fact", 0.7),
        ("User uses PostgreSQL for relational data", "fact", 0.65),
        ("Redis is used for caching and sessions", "fact", 0.6),
        ("The ML model uses collaborative filtering", "fact", 0.75),
        # Goals
        ("User wants to improve recommendation accuracy", "goal", 0.85),
        ("User needs to reduce API latency below 100ms", "goal", 0.8),
        ("User plans to add real-time personalization", "goal", 0.7),
        # Skills
        ("User knows PyTorch and TensorFlow", "skill", 0.7),
        ("User is experienced with Docker and Kubernetes", "skill", 0.65),
    ]
    for content, mem_type, importance in seed_memories:
        agent.remember(content, mem_type, importance)
    rprint(f"  Stored {len(seed_memories)} memories")

    rprint("\n[bold]Testing memory recall...[/bold]")
    queries = [
        "What programming language should I use?",
        "What is the user working on?",
        "How should I format my responses?",
    ]
    for query in queries:
        rprint(f"\n[cyan]Query:[/cyan] {query}")
        memories = agent.recall(query, limit=2)
        for m in memories:
            rprint(f"  [score={m['score']:.3f}] {m['content']}")

    agent.show_memories()

    if agent.llm:
        rprint("\n[bold]Testing chat with memory...[/bold]")
        demo_messages = [
            "What's the best way to build a recommendation engine?",
            "Can you show me a quick example?",
        ]
        for msg in demo_messages:
            rprint(f"\n[cyan]User:[/cyan] {msg}")
            response = agent.chat(msg)
            rprint(f"[green]Assistant:[/green] {response}")
    else:
        rprint("\n[yellow]Skipping chat demo - add OPENAI_API_KEY for LLM features[/yellow]")


def interactive_mode():
    rprint(Panel("Workshop 04: Interactive Memory Agent", style="bold blue"))

    agent_id = Prompt.ask("Enter agent ID", default="my-agent")
    agent = MemoryAgent(agent_id=agent_id)

    rprint(f"\n[green]Started agent '{agent_id}' (session: {agent.session_id})[/green]")
    rprint("Commands: /memories, /remember <text>, /clear, /quit\n")

    if not agent.llm:
        rprint("[yellow]Note: Chat requires OPENAI_API_KEY. Memory features work without it.[/yellow]\n")

    while True:
        try:
            user_input = Prompt.ask("[cyan]You[/cyan]")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() == "/quit":
            break
        elif user_input.lower() == "/memories":
            agent.show_memories()
            continue
        elif user_input.lower().startswith("/remember "):
            content = user_input[10:]
            agent.remember(content, "manual", 0.8)
            rprint(f"[green]Remembered: {content}[/green]")
            continue
        elif user_input.lower() == "/clear":
            agent.clear_memories()
            rprint("[yellow]Memories cleared[/yellow]")
            continue
        elif not user_input.strip():
            continue

        response = agent.chat(user_input)
        rprint(f"[green]Agent:[/green] {response}\n")

    rprint("\n[yellow]Session ended[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Memory Agent Workshop")
    parser.add_argument(
        "--mode",
        choices=["demo", "interactive"],
        default=None,
        help="Run mode: demo or interactive",
    )
    args = parser.parse_args()

    if args.mode:
        mode = args.mode
    else:
        mode = Prompt.ask(
            "Select mode",
            choices=["demo", "interactive"],
            default="demo",
        )

    if mode == "demo":
        demo_mode()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
