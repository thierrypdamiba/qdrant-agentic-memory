"""Workshop 02: Semantic Memory System"""

import uuid
from datetime import datetime
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
from rich.table import Table

from workshop.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_URL,
)

SEMANTIC_COLLECTION = "semantic_memories"


class MemoryType(str, Enum):
    FACT = "fact"
    SKILL = "skill"
    PREFERENCE = "preference"
    CONTEXT = "context"


class SemanticMemory:
    def __init__(self):
        if QDRANT_API_KEY:
            self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            self.client = QdrantClient(url=QDRANT_URL)
        self.embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
        self._setup_collection()

    def _setup_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == SEMANTIC_COLLECTION for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=SEMANTIC_COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            self.client.create_payload_index(
                collection_name=SEMANTIC_COLLECTION,
                field_name="type",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=SEMANTIC_COLLECTION,
                field_name="importance",
                field_schema=PayloadSchemaType.FLOAT,
            )

    def _embed(self, text: str) -> list[float]:
        return list(self.embedder.embed([text]))[0].tolist()

    def store(
        self,
        content: str,
        memory_type: MemoryType,
        importance: float = 0.5,
        source: str = "user",
    ) -> str:
        memory_id = str(uuid.uuid4())

        payload = {
            "content": content,
            "type": memory_type.value,
            "importance": importance,
            "source": source,
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
            "last_accessed": datetime.now().isoformat(),
        }

        self.client.upsert(
            collection_name=SEMANTIC_COLLECTION,
            points=[PointStruct(id=memory_id, vector=self._embed(content), payload=payload)],
        )

        return memory_id

    def recall(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        min_importance: float = 0.0,
        limit: int = 5,
    ) -> list[dict]:
        conditions = []
        if memory_type:
            conditions.append(
                FieldCondition(key="type", match=MatchValue(value=memory_type.value))
            )
        if min_importance > 0:
            conditions.append(
                FieldCondition(key="importance", range=Range(gte=min_importance))
            )

        search_filter = Filter(must=conditions) if conditions else None

        results = self.client.query_points(
            collection_name=SEMANTIC_COLLECTION,
            query=self._embed(query),
            query_filter=search_filter,
            limit=limit,
        )

        for r in results.points:
            self._update_access(r.id, r.payload)

        return [
            {
                "id": r.id,
                "content": r.payload["content"],
                "type": r.payload["type"],
                "importance": r.payload["importance"],
                "score": r.score,
            }
            for r in results.points
        ]

    def _update_access(self, memory_id: str, payload: dict):
        self.client.set_payload(
            collection_name=SEMANTIC_COLLECTION,
            payload={
                "access_count": payload.get("access_count", 0) + 1,
                "last_accessed": datetime.now().isoformat(),
            },
            points=[memory_id],
        )

    def get_important_memories(self, min_importance: float = 0.7) -> list[dict]:
        results, _ = self.client.scroll(
            collection_name=SEMANTIC_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="importance", range=Range(gte=min_importance))]
            ),
            limit=100,
        )

        return [
            {
                "id": r.id,
                "content": r.payload["content"],
                "type": r.payload["type"],
                "importance": r.payload["importance"],
            }
            for r in results
        ]

    def clear(self):
        self.client.delete_collection(SEMANTIC_COLLECTION)
        self._setup_collection()


def main():
    rprint(Panel("Workshop 02: Semantic Memory System (FREE)", style="bold blue"))

    memory = SemanticMemory()
    memory.clear()

    rprint("\n[bold]Storing semantic memories...[/bold]")

    memories_to_store = [
        # Facts (40)
        ("Python uses indentation for code blocks.", MemoryType.FACT, 0.8),
        ("Qdrant supports both sparse and dense vectors.", MemoryType.FACT, 0.85),
        ("FastAPI automatically generates OpenAPI documentation.", MemoryType.FACT, 0.7),
        ("PostgreSQL supports JSONB for semi-structured data.", MemoryType.FACT, 0.75),
        ("Redis can be used as a message broker with pub/sub.", MemoryType.FACT, 0.65),
        ("Docker images should be kept small for faster deployments.", MemoryType.FACT, 0.7),
        ("JWT tokens should have short expiration times.", MemoryType.FACT, 0.8),
        ("Cosine similarity is preferred for normalized embeddings.", MemoryType.FACT, 0.85),
        ("Python 3.10 introduced structural pattern matching.", MemoryType.FACT, 0.6),
        ("Pydantic v2 is significantly faster than v1.", MemoryType.FACT, 0.75),
        ("Async functions in Python return coroutines.", MemoryType.FACT, 0.7),
        ("SQLAlchemy 2.0 uses a new query syntax.", MemoryType.FACT, 0.65),
        ("HNSW is the default index type in Qdrant.", MemoryType.FACT, 0.8),
        ("BGE embeddings have 384 dimensions.", MemoryType.FACT, 0.75),
        ("Kubernetes pods can have multiple containers.", MemoryType.FACT, 0.6),
        ("GraphQL uses a single endpoint for all queries.", MemoryType.FACT, 0.7),
        ("WebSocket connections are full-duplex.", MemoryType.FACT, 0.65),
        ("OAuth 2.0 has four grant types.", MemoryType.FACT, 0.7),
        ("bcrypt is a one-way hashing algorithm.", MemoryType.FACT, 0.8),
        ("PostgreSQL VACUUM reclaims storage.", MemoryType.FACT, 0.55),
        ("Redis supports Lua scripting.", MemoryType.FACT, 0.6),
        ("Prometheus uses a pull-based model.", MemoryType.FACT, 0.65),
        ("Terraform state should be stored remotely.", MemoryType.FACT, 0.75),
        ("GitHub Actions workflows use YAML syntax.", MemoryType.FACT, 0.7),
        ("Docker layers are cached for faster builds.", MemoryType.FACT, 0.75),
        ("Python GIL prevents true multithreading.", MemoryType.FACT, 0.8),
        ("Celery requires a message broker.", MemoryType.FACT, 0.7),
        ("Alembic tracks migrations with revision IDs.", MemoryType.FACT, 0.65),
        ("pytest fixtures can have different scopes.", MemoryType.FACT, 0.7),
        ("Pydantic models validate data at runtime.", MemoryType.FACT, 0.8),
        ("FastAPI depends on Starlette and Pydantic.", MemoryType.FACT, 0.75),
        ("SQLAlchemy supports both ORM and Core patterns.", MemoryType.FACT, 0.7),
        ("Elasticsearch uses inverted indexes.", MemoryType.FACT, 0.65),
        ("Kafka partitions enable parallel processing.", MemoryType.FACT, 0.7),
        ("S3 eventual consistency was removed in 2020.", MemoryType.FACT, 0.6),
        ("Lambda cold starts add latency.", MemoryType.FACT, 0.75),
        ("CloudFront caches at edge locations.", MemoryType.FACT, 0.7),
        ("nginx can do load balancing.", MemoryType.FACT, 0.65),
        ("gRPC uses Protocol Buffers.", MemoryType.FACT, 0.7),
        ("OpenTelemetry combines metrics, logs, and traces.", MemoryType.FACT, 0.75),
        # Preferences (35)
        ("The user prefers functional programming patterns.", MemoryType.PREFERENCE, 0.9),
        ("User dislikes verbose error messages.", MemoryType.PREFERENCE, 0.75),
        ("User prefers explicit imports over wildcards.", MemoryType.PREFERENCE, 0.8),
        ("User likes code examples more than lengthy explanations.", MemoryType.PREFERENCE, 0.85),
        ("User prefers composition over inheritance.", MemoryType.PREFERENCE, 0.8),
        ("User wants type hints on all function signatures.", MemoryType.PREFERENCE, 0.9),
        ("User dislikes deeply nested code.", MemoryType.PREFERENCE, 0.7),
        ("User prefers early returns over nested conditionals.", MemoryType.PREFERENCE, 0.75),
        ("User likes descriptive variable names.", MemoryType.PREFERENCE, 0.8),
        ("User prefers pytest over unittest.", MemoryType.PREFERENCE, 0.7),
        ("User wants functions to be under 20 lines.", MemoryType.PREFERENCE, 0.65),
        ("User prefers dataclasses for simple data structures.", MemoryType.PREFERENCE, 0.8),
        ("User likes dependency injection patterns.", MemoryType.PREFERENCE, 0.7),
        ("User prefers immutable data structures.", MemoryType.PREFERENCE, 0.75),
        ("User dislikes global state.", MemoryType.PREFERENCE, 0.85),
        ("User prefers explicit over implicit behavior.", MemoryType.PREFERENCE, 0.8),
        ("User likes small pull requests.", MemoryType.PREFERENCE, 0.75),
        ("User prefers trunk-based development.", MemoryType.PREFERENCE, 0.7),
        ("User dislikes long-lived feature branches.", MemoryType.PREFERENCE, 0.65),
        ("User prefers automated testing over manual.", MemoryType.PREFERENCE, 0.85),
        ("User likes integration tests for APIs.", MemoryType.PREFERENCE, 0.7),
        ("User prefers property-based testing for edge cases.", MemoryType.PREFERENCE, 0.6),
        ("User dislikes mocking too much.", MemoryType.PREFERENCE, 0.65),
        ("User prefers real databases in tests.", MemoryType.PREFERENCE, 0.7),
        ("User likes clear error messages with context.", MemoryType.PREFERENCE, 0.8),
        ("User prefers structured logging.", MemoryType.PREFERENCE, 0.75),
        ("User dislikes print debugging in production.", MemoryType.PREFERENCE, 0.85),
        ("User prefers feature flags for deployments.", MemoryType.PREFERENCE, 0.7),
        ("User likes canary releases.", MemoryType.PREFERENCE, 0.65),
        ("User prefers blue-green deployments.", MemoryType.PREFERENCE, 0.6),
        ("User dislikes big bang releases.", MemoryType.PREFERENCE, 0.8),
        ("User prefers incremental migrations.", MemoryType.PREFERENCE, 0.75),
        ("User likes documentation close to code.", MemoryType.PREFERENCE, 0.7),
        ("User prefers docstrings over external docs.", MemoryType.PREFERENCE, 0.65),
        ("User dislikes outdated documentation.", MemoryType.PREFERENCE, 0.85),
        # Skills (35)
        ("To deploy, run 'make deploy' in the project root.", MemoryType.SKILL, 0.7),
        ("For testing, use pytest with the -v flag.", MemoryType.SKILL, 0.65),
        ("Use 'git rebase -i' to squash commits before merging.", MemoryType.SKILL, 0.6),
        ("Run 'docker compose up -d' to start services in background.", MemoryType.SKILL, 0.7),
        ("Use 'uv pip install -e .' for editable installs.", MemoryType.SKILL, 0.75),
        ("Profile Python code with 'python -m cProfile script.py'.", MemoryType.SKILL, 0.55),
        ("Use 'ruff check --fix' to auto-fix linting issues.", MemoryType.SKILL, 0.7),
        ("Create virtual environments with 'uv venv'.", MemoryType.SKILL, 0.8),
        ("Use 'gh pr create' to create pull requests from CLI.", MemoryType.SKILL, 0.65),
        ("Run 'alembic upgrade head' to apply database migrations.", MemoryType.SKILL, 0.7),
        ("Use 'docker system prune' to clean up.", MemoryType.SKILL, 0.6),
        ("Run 'kubectl get pods -w' to watch pod status.", MemoryType.SKILL, 0.7),
        ("Use 'terraform plan' before apply.", MemoryType.SKILL, 0.8),
        ("Debug with 'python -m pdb script.py'.", MemoryType.SKILL, 0.65),
        ("Use 'psql \\d+ table' to see table schema.", MemoryType.SKILL, 0.7),
        ("Run 'redis-cli MONITOR' to debug.", MemoryType.SKILL, 0.6),
        ("Use 'curl -v' for verbose HTTP debugging.", MemoryType.SKILL, 0.7),
        ("Profile memory with 'memory_profiler'.", MemoryType.SKILL, 0.65),
        ("Use 'py-spy' for production profiling.", MemoryType.SKILL, 0.75),
        ("Run 'pytest --cov' for coverage reports.", MemoryType.SKILL, 0.7),
        ("Use 'httpie' for nicer HTTP requests.", MemoryType.SKILL, 0.6),
        ("Debug async with 'asyncio.get_event_loop().set_debug(True)'.", MemoryType.SKILL, 0.65),
        ("Use 'git bisect' to find bugs.", MemoryType.SKILL, 0.7),
        ("Run 'git reflog' to recover lost commits.", MemoryType.SKILL, 0.75),
        ("Use 'docker logs -f' to stream logs.", MemoryType.SKILL, 0.65),
        ("Run 'kubectl describe pod' for debugging.", MemoryType.SKILL, 0.7),
        ("Use 'aws s3 sync' for uploads.", MemoryType.SKILL, 0.6),
        ("Run 'pg_dump' for backups.", MemoryType.SKILL, 0.75),
        ("Use 'EXPLAIN ANALYZE' for query optimization.", MemoryType.SKILL, 0.8),
        ("Run 'redis-cli INFO' for stats.", MemoryType.SKILL, 0.65),
        ("Use 'jq' for JSON processing.", MemoryType.SKILL, 0.7),
        ("Run 'htop' for system monitoring.", MemoryType.SKILL, 0.6),
        ("Use 'tcpdump' for network debugging.", MemoryType.SKILL, 0.65),
        ("Run 'strace' for system call tracing.", MemoryType.SKILL, 0.6),
        ("Use 'lsof -i' to see open ports.", MemoryType.SKILL, 0.65),
        # Context (30)
        ("User is building a recommendation engine.", MemoryType.CONTEXT, 0.6),
        ("The codebase uses async/await throughout.", MemoryType.CONTEXT, 0.5),
        ("The team has 5 backend engineers.", MemoryType.CONTEXT, 0.4),
        ("The project started 6 months ago.", MemoryType.CONTEXT, 0.35),
        ("User's company is in the e-commerce space.", MemoryType.CONTEXT, 0.5),
        ("The API serves mobile and web clients.", MemoryType.CONTEXT, 0.55),
        ("Production runs on AWS with EKS.", MemoryType.CONTEXT, 0.6),
        ("The team does two-week sprints.", MemoryType.CONTEXT, 0.4),
        ("Code reviews require at least one approval.", MemoryType.CONTEXT, 0.5),
        ("The staging environment mirrors production.", MemoryType.CONTEXT, 0.45),
        ("User works in the platform team.", MemoryType.CONTEXT, 0.5),
        ("The company has 100 employees.", MemoryType.CONTEXT, 0.35),
        ("User reports to the CTO.", MemoryType.CONTEXT, 0.4),
        ("The product handles payments.", MemoryType.CONTEXT, 0.6),
        ("PCI compliance is required.", MemoryType.CONTEXT, 0.7),
        ("User is on-call this week.", MemoryType.CONTEXT, 0.5),
        ("The main database is 500GB.", MemoryType.CONTEXT, 0.45),
        ("Peak traffic is at 6pm EST.", MemoryType.CONTEXT, 0.5),
        ("The team uses Notion for docs.", MemoryType.CONTEXT, 0.35),
        ("Figma is used for designs.", MemoryType.CONTEXT, 0.3),
        ("User previously worked at Stripe.", MemoryType.CONTEXT, 0.4),
        ("User has a CS degree from MIT.", MemoryType.CONTEXT, 0.35),
        ("User has 8 years of experience.", MemoryType.CONTEXT, 0.5),
        ("User is based in NYC.", MemoryType.CONTEXT, 0.4),
        ("The office is in Manhattan.", MemoryType.CONTEXT, 0.35),
        ("User works hybrid 3 days in office.", MemoryType.CONTEXT, 0.4),
        ("The company was founded in 2020.", MemoryType.CONTEXT, 0.35),
        ("Series B funding was $50M.", MemoryType.CONTEXT, 0.4),
        ("The CEO is ex-Google.", MemoryType.CONTEXT, 0.3),
        ("User is working on fraud detection.", MemoryType.CONTEXT, 0.55),
    ]

    for content, mem_type, importance in memories_to_store:
        memory.store(content, mem_type, importance)
        rprint(f"  [{mem_type.value}] {content[:50]}...")

    rprint("\n[bold]Recalling memories with filters...[/bold]")

    rprint("\n[cyan]Query:[/cyan] 'How do I write code in this project?'")
    results = memory.recall("How do I write code in this project?", limit=3)
    for r in results:
        rprint(f"  [{r['type']}] Score {r['score']:.3f}: {r['content']}")

    rprint("\n[cyan]Query:[/cyan] 'What does the user like?' (preferences only)")
    results = memory.recall("What does the user like?", memory_type=MemoryType.PREFERENCE)
    for r in results:
        rprint(f"  [importance={r['importance']}] {r['content']}")

    rprint("\n[cyan]Query:[/cyan] 'Important things to remember' (importance >= 0.7)")
    results = memory.recall("Important things to remember", min_importance=0.7)
    for r in results:
        rprint(f"  [importance={r['importance']}] {r['content']}")

    rprint("\n[bold]All highly important memories (>= 0.7):[/bold]")
    important = memory.get_important_memories(0.7)

    table = Table(show_header=True)
    table.add_column("Type")
    table.add_column("Importance")
    table.add_column("Content")

    for m in important:
        table.add_row(m["type"], f"{m['importance']:.2f}", m["content"][:50] + "...")

    rprint(table)


if __name__ == "__main__":
    main()
