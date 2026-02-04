"""Workshop 03: Episodic Memory for Conversations"""

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

from workshop.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    LLM_MODEL,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_URL,
)

EPISODES_COLLECTION = "conversation_episodes"


class EpisodicMemory:
    def __init__(self):
        if QDRANT_API_KEY:
            self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            self.client = QdrantClient(url=QDRANT_URL)
        self.embedder = TextEmbedding(model_name=EMBEDDING_MODEL)

        self.llm = None
        if OPENAI_API_KEY:
            from openai import OpenAI
            self.llm = OpenAI(api_key=OPENAI_API_KEY)

        self._setup_collection()

    def _setup_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == EPISODES_COLLECTION for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=EPISODES_COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            self.client.create_payload_index(
                collection_name=EPISODES_COLLECTION,
                field_name="session_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )

    def _embed(self, text: str) -> list[float]:
        return list(self.embedder.embed([text]))[0].tolist()

    def store_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        turn_number: int,
    ) -> str:
        turn_id = str(uuid.uuid4())
        turn_text = f"{role}: {content}"

        payload = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "turn_number": turn_number,
            "timestamp": datetime.now().isoformat(),
        }

        self.client.upsert(
            collection_name=EPISODES_COLLECTION,
            points=[PointStruct(id=turn_id, vector=self._embed(turn_text), payload=payload)],
        )

        return turn_id

    def store_conversation(self, session_id: str, messages: list[dict]) -> list[str]:
        turn_ids = []
        for i, msg in enumerate(messages):
            turn_id = self.store_turn(
                session_id=session_id,
                role=msg["role"],
                content=msg["content"],
                turn_number=i,
            )
            turn_ids.append(turn_id)
        return turn_ids

    def recall_relevant(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        search_filter = None
        if session_id:
            search_filter = Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            )

        results = self.client.query_points(
            collection_name=EPISODES_COLLECTION,
            query=self._embed(query),
            query_filter=search_filter,
            limit=limit,
        )

        return [
            {
                "session_id": r.payload["session_id"],
                "role": r.payload["role"],
                "content": r.payload["content"],
                "turn_number": r.payload["turn_number"],
                "score": r.score,
            }
            for r in results.points
        ]

    def get_session_history(self, session_id: str) -> list[dict]:
        results, _ = self.client.scroll(
            collection_name=EPISODES_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            ),
            limit=1000,
        )

        turns = [
            {
                "role": r.payload["role"],
                "content": r.payload["content"],
                "turn_number": r.payload["turn_number"],
            }
            for r in results
        ]

        return sorted(turns, key=lambda x: x["turn_number"])

    def summarize_session(self, session_id: str) -> str:
        if not self.llm:
            return "[LLM not configured - add OPENAI_API_KEY for summaries]"

        history = self.get_session_history(session_id)
        if not history:
            return "No conversation found for this session."

        conversation_text = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in history
        )

        try:
            response = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize the following conversation in 2-3 sentences. "
                        "Focus on the main topics discussed and any conclusions reached.",
                    },
                    {"role": "user", "content": conversation_text},
                ],
                max_tokens=200,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Could not generate summary: {e}"

    def clear(self):
        self.client.delete_collection(EPISODES_COLLECTION)
        self._setup_collection()


def main():
    rprint(Panel("Workshop 03: Episodic Memory — Conversation history & session recall", style="bold blue"))

    memory = EpisodicMemory()
    memory.clear()

    rprint("\n[bold]Storing example conversations...[/bold]")

    conversations = [
        ("session-python-001", "Python file handling", [
            {"role": "user", "content": "How do I read a file in Python?"},
            {"role": "assistant", "content": "Use open() with a context manager: with open('file.txt', 'r') as f: content = f.read()"},
            {"role": "user", "content": "What about reading line by line?"},
            {"role": "assistant", "content": "Use f.readlines() for a list, or iterate: for line in f: process(line)"},
            {"role": "user", "content": "How do I handle encoding issues?"},
            {"role": "assistant", "content": "Specify encoding explicitly: open('file.txt', 'r', encoding='utf-8')"},
        ]),
        ("session-qdrant-002", "Qdrant basics", [
            {"role": "user", "content": "What is Qdrant?"},
            {"role": "assistant", "content": "Qdrant is a vector similarity search engine for AI applications."},
            {"role": "user", "content": "How do I filter results?"},
            {"role": "assistant", "content": "Use the Filter class with FieldCondition for exact match, range, or geo filters."},
            {"role": "user", "content": "Can I combine filters?"},
            {"role": "assistant", "content": "Yes, use must (AND), should (OR), and must_not in Filter."},
        ]),
        ("session-deploy-003", "deployment", [
            {"role": "user", "content": "How do I deploy my app to production?"},
            {"role": "assistant", "content": "Run 'make deploy' from the root directory. It builds and pushes the Docker image."},
            {"role": "user", "content": "What about environment variables?"},
            {"role": "assistant", "content": "Set them in .env.production. The deploy script reads from there."},
        ]),
        ("session-fastapi-004", "FastAPI patterns", [
            {"role": "user", "content": "How do I add authentication to FastAPI?"},
            {"role": "assistant", "content": "Use OAuth2PasswordBearer for token auth, or create a custom dependency."},
            {"role": "user", "content": "What about rate limiting?"},
            {"role": "assistant", "content": "Use slowapi library or implement with Redis counters in middleware."},
            {"role": "user", "content": "How do I handle background tasks?"},
            {"role": "assistant", "content": "Use BackgroundTasks for simple cases, Celery or ARQ for complex workflows."},
        ]),
        ("session-testing-005", "testing strategies", [
            {"role": "user", "content": "How should I structure my tests?"},
            {"role": "assistant", "content": "Use pytest with fixtures. Organize by feature: tests/test_users.py, tests/test_orders.py"},
            {"role": "user", "content": "What about mocking external APIs?"},
            {"role": "assistant", "content": "Use responses or httpx_mock for HTTP mocking, or create fixture factories."},
            {"role": "user", "content": "How do I test async code?"},
            {"role": "assistant", "content": "Use pytest-asyncio and mark tests with @pytest.mark.asyncio."},
            {"role": "user", "content": "What coverage should I aim for?"},
            {"role": "assistant", "content": "80% is a good target. Focus on critical paths, not hitting arbitrary numbers."},
        ]),
        ("session-database-006", "database optimization", [
            {"role": "user", "content": "My queries are slow, what should I check?"},
            {"role": "assistant", "content": "Run EXPLAIN ANALYZE, check for missing indexes, look for N+1 queries."},
            {"role": "user", "content": "How do I add an index?"},
            {"role": "assistant", "content": "CREATE INDEX idx_name ON table(column). For Alembic: op.create_index()."},
            {"role": "user", "content": "When should I use connection pooling?"},
            {"role": "assistant", "content": "Always in production. Use asyncpg pool or SQLAlchemy's pool settings."},
        ]),
        ("session-docker-007", "Docker best practices", [
            {"role": "user", "content": "How do I reduce my Docker image size?"},
            {"role": "assistant", "content": "Use multi-stage builds, slim base images, and combine RUN commands."},
            {"role": "user", "content": "What base image should I use for Python?"},
            {"role": "assistant", "content": "python:3.11-slim for production, python:3.11 for development with build tools."},
            {"role": "user", "content": "How do I handle secrets in Docker?"},
            {"role": "assistant", "content": "Use Docker secrets or mount them at runtime. Never bake secrets into images."},
        ]),
        ("session-git-008", "git workflow", [
            {"role": "user", "content": "How do I undo my last commit?"},
            {"role": "assistant", "content": "git reset --soft HEAD~1 keeps changes staged, --hard discards them."},
            {"role": "user", "content": "What's your preferred branching strategy?"},
            {"role": "assistant", "content": "Trunk-based development with short-lived feature branches. Merge within 1-2 days."},
            {"role": "user", "content": "How do I clean up merged branches?"},
            {"role": "assistant", "content": "git branch --merged | grep -v main | xargs git branch -d"},
        ]),
        ("session-ml-009", "ML pipeline questions", [
            {"role": "user", "content": "How do I version my ML models?"},
            {"role": "assistant", "content": "Use MLflow or DVC. Store model artifacts with metadata about training data and params."},
            {"role": "user", "content": "What about feature stores?"},
            {"role": "assistant", "content": "Feast is popular for offline/online feature serving. Keeps training and inference consistent."},
            {"role": "user", "content": "How do I monitor model drift?"},
            {"role": "assistant", "content": "Track prediction distributions over time. Alert on statistical divergence from baseline."},
            {"role": "user", "content": "Should I retrain automatically?"},
            {"role": "assistant", "content": "Start with scheduled retraining, then add drift-triggered retraining once you have monitoring."},
        ]),
        ("session-perf-010", "performance tuning", [
            {"role": "user", "content": "My API response times are inconsistent."},
            {"role": "assistant", "content": "Check for GC pauses, cold database connections, or external API latency."},
            {"role": "user", "content": "How do I profile Python code?"},
            {"role": "assistant", "content": "Use py-spy for production profiling, cProfile for dev. Look at flame graphs."},
            {"role": "user", "content": "When should I add caching?"},
            {"role": "assistant", "content": "When you have repeated expensive computations or DB queries. Measure first."},
        ]),
        ("session-kubernetes-011", "Kubernetes setup", [
            {"role": "user", "content": "How do I set up a Kubernetes cluster?"},
            {"role": "assistant", "content": "Use EKS, GKE, or AKS for managed clusters. For local dev, use minikube or kind."},
            {"role": "user", "content": "What's the difference between Deployment and StatefulSet?"},
            {"role": "assistant", "content": "StatefulSet gives stable network IDs and persistent storage. Use for databases."},
            {"role": "user", "content": "How do I handle secrets in Kubernetes?"},
            {"role": "assistant", "content": "Use Kubernetes Secrets, but consider external-secrets or Vault for production."},
            {"role": "user", "content": "What about ConfigMaps?"},
            {"role": "assistant", "content": "Use ConfigMaps for non-sensitive config. Mount as files or env vars."},
        ]),
        ("session-security-012", "security practices", [
            {"role": "user", "content": "How do I prevent SQL injection?"},
            {"role": "assistant", "content": "Use parameterized queries. Never concatenate user input into SQL strings."},
            {"role": "user", "content": "What about XSS?"},
            {"role": "assistant", "content": "Escape output, use CSP headers, and sanitize HTML input with bleach."},
            {"role": "user", "content": "How should I store passwords?"},
            {"role": "assistant", "content": "Use bcrypt or argon2 with sufficient work factor. Never MD5 or SHA1."},
            {"role": "user", "content": "What about API keys?"},
            {"role": "assistant", "content": "Hash them with SHA-256. Store only the hash, return the key once on creation."},
        ]),
        ("session-monitoring-013", "observability", [
            {"role": "user", "content": "What should I monitor?"},
            {"role": "assistant", "content": "The four golden signals: latency, traffic, errors, and saturation."},
            {"role": "user", "content": "How do I set up alerting?"},
            {"role": "assistant", "content": "Use Prometheus Alertmanager. Start with high-level alerts, avoid noise."},
            {"role": "user", "content": "What about distributed tracing?"},
            {"role": "assistant", "content": "Use OpenTelemetry with Jaeger or Tempo. Instrument at service boundaries."},
            {"role": "user", "content": "How do I correlate logs with traces?"},
            {"role": "assistant", "content": "Include trace_id in logs. OpenTelemetry handles this automatically."},
        ]),
        ("session-redis-014", "Redis patterns", [
            {"role": "user", "content": "When should I use Redis?"},
            {"role": "assistant", "content": "Caching, session storage, rate limiting, pub/sub, and simple queues."},
            {"role": "user", "content": "How do I handle cache invalidation?"},
            {"role": "assistant", "content": "Use TTL for time-based expiry, or explicit DELETE on writes."},
            {"role": "user", "content": "What about Redis Cluster?"},
            {"role": "assistant", "content": "Use it for horizontal scaling. Keys are sharded across nodes."},
            {"role": "user", "content": "How do I handle connection issues?"},
            {"role": "assistant", "content": "Use connection pooling and implement retry logic with exponential backoff."},
        ]),
        ("session-async-015", "async programming", [
            {"role": "user", "content": "When should I use async in Python?"},
            {"role": "assistant", "content": "For I/O-bound operations: HTTP requests, database queries, file I/O."},
            {"role": "user", "content": "What about CPU-bound tasks?"},
            {"role": "assistant", "content": "Use multiprocessing or run_in_executor. asyncio won't help with CPU work."},
            {"role": "user", "content": "How do I handle errors in async code?"},
            {"role": "assistant", "content": "Use try/except as usual. For multiple tasks, use asyncio.gather with return_exceptions=True."},
            {"role": "user", "content": "What's the best way to run background tasks?"},
            {"role": "assistant", "content": "asyncio.create_task() for fire-and-forget, or TaskGroup for managed concurrency."},
        ]),
        ("session-api-design-016", "API design", [
            {"role": "user", "content": "REST or GraphQL?"},
            {"role": "assistant", "content": "REST for simple CRUD, GraphQL when clients need flexible queries."},
            {"role": "user", "content": "How should I version my API?"},
            {"role": "assistant", "content": "URL path versioning (/v1/) is simplest. Header versioning for cleaner URLs."},
            {"role": "user", "content": "What about pagination?"},
            {"role": "assistant", "content": "Cursor-based for real-time data, offset-based for simple cases. Include total count."},
            {"role": "user", "content": "How do I handle errors?"},
            {"role": "assistant", "content": "Use standard HTTP codes. Include error code, message, and details in response body."},
        ]),
        ("session-migrations-017", "database migrations", [
            {"role": "user", "content": "How do I handle large table migrations?"},
            {"role": "assistant", "content": "Use pt-online-schema-change for MySQL, or pg_repack for PostgreSQL."},
            {"role": "user", "content": "What about zero-downtime migrations?"},
            {"role": "assistant", "content": "Use expand-contract pattern: add new column, migrate data, remove old column."},
            {"role": "user", "content": "How do I rollback migrations?"},
            {"role": "assistant", "content": "Write down migrations. Test rollbacks in staging before production."},
            {"role": "user", "content": "Should I use Django migrations or Alembic?"},
            {"role": "assistant", "content": "Django migrations for Django projects, Alembic for SQLAlchemy or standalone."},
        ]),
        ("session-logging-018", "logging practices", [
            {"role": "user", "content": "What logging format should I use?"},
            {"role": "assistant", "content": "JSON for production (machine-readable), human-friendly for development."},
            {"role": "user", "content": "What should I log?"},
            {"role": "assistant", "content": "Request ID, user ID, action, duration, and any error details. Avoid PII."},
            {"role": "user", "content": "How do I handle log rotation?"},
            {"role": "assistant", "content": "Use logrotate on Linux, or log to stdout and let the container runtime handle it."},
            {"role": "user", "content": "What about log levels?"},
            {"role": "assistant", "content": "ERROR for failures, WARNING for concerns, INFO for events, DEBUG for details."},
        ]),
        ("session-cicd-019", "CI/CD pipelines", [
            {"role": "user", "content": "What should run in CI?"},
            {"role": "assistant", "content": "Lint, type check, unit tests, integration tests, security scans, in that order."},
            {"role": "user", "content": "How do I speed up CI?"},
            {"role": "assistant", "content": "Parallelize tests, cache dependencies, use smaller Docker images."},
            {"role": "user", "content": "Should I deploy on every commit?"},
            {"role": "assistant", "content": "To staging, yes. To production, use feature flags or manual approval."},
            {"role": "user", "content": "How do I handle flaky tests?"},
            {"role": "assistant", "content": "Quarantine them, fix root causes, or add retries as last resort."},
        ]),
        ("session-architecture-020", "system architecture", [
            {"role": "user", "content": "When should I use microservices?"},
            {"role": "assistant", "content": "When you have independent teams and clear domain boundaries. Start with monolith."},
            {"role": "user", "content": "How do I handle service communication?"},
            {"role": "assistant", "content": "REST for sync, message queues for async. Use retries and circuit breakers."},
            {"role": "user", "content": "What about data consistency?"},
            {"role": "assistant", "content": "Accept eventual consistency for distributed systems. Use sagas for transactions."},
            {"role": "user", "content": "How do I handle partial failures?"},
            {"role": "assistant", "content": "Implement circuit breakers, fallbacks, and graceful degradation."},
        ]),
        ("session-embeddings-021", "vector embeddings", [
            {"role": "user", "content": "Which embedding model should I use?"},
            {"role": "assistant", "content": "BGE-small for speed, BGE-large for quality. OpenAI ada-002 for simplicity."},
            {"role": "user", "content": "How do I chunk documents?"},
            {"role": "assistant", "content": "Use semantic chunking or fixed size with overlap. 512 tokens is common."},
            {"role": "user", "content": "What similarity metric should I use?"},
            {"role": "assistant", "content": "Cosine for normalized embeddings, dot product for magnitude-aware similarity."},
            {"role": "user", "content": "How do I handle multilingual content?"},
            {"role": "assistant", "content": "Use multilingual models like mE5 or translate to English first."},
        ]),
        ("session-llm-022", "LLM integration", [
            {"role": "user", "content": "How do I handle LLM rate limits?"},
            {"role": "assistant", "content": "Implement retry with exponential backoff, use queues for batch processing."},
            {"role": "user", "content": "What about prompt injection?"},
            {"role": "assistant", "content": "Validate input, use system prompts for instructions, never trust LLM output blindly."},
            {"role": "user", "content": "How do I reduce costs?"},
            {"role": "assistant", "content": "Cache responses, use smaller models for simple tasks, batch requests."},
            {"role": "user", "content": "Should I fine-tune or use prompting?"},
            {"role": "assistant", "content": "Start with prompting and few-shot examples. Fine-tune only if needed."},
        ]),
        ("session-terraform-023", "Infrastructure as Code", [
            {"role": "user", "content": "How do I structure Terraform projects?"},
            {"role": "assistant", "content": "Use modules for reusable components, separate state per environment."},
            {"role": "user", "content": "What about state management?"},
            {"role": "assistant", "content": "Use remote state in S3 with DynamoDB locking. Never commit state files."},
            {"role": "user", "content": "How do I handle secrets in Terraform?"},
            {"role": "assistant", "content": "Use AWS Secrets Manager or Vault. Reference secrets, don't hardcode them."},
            {"role": "user", "content": "What about drift detection?"},
            {"role": "assistant", "content": "Run terraform plan regularly in CI. Alert on unexpected changes."},
        ]),
        ("session-websockets-024", "real-time features", [
            {"role": "user", "content": "WebSockets or Server-Sent Events?"},
            {"role": "assistant", "content": "SSE for server-to-client only, WebSockets for bidirectional communication."},
            {"role": "user", "content": "How do I scale WebSocket connections?"},
            {"role": "assistant", "content": "Use Redis pub/sub or a message broker to broadcast across instances."},
            {"role": "user", "content": "What about reconnection handling?"},
            {"role": "assistant", "content": "Implement client-side reconnection with exponential backoff and state sync."},
            {"role": "user", "content": "How do I authenticate WebSocket connections?"},
            {"role": "assistant", "content": "Pass token in initial HTTP upgrade request or first message."},
        ]),
        ("session-oauth-025", "OAuth implementation", [
            {"role": "user", "content": "Which OAuth flow should I use?"},
            {"role": "assistant", "content": "Authorization Code with PKCE for web/mobile, Client Credentials for server-to-server."},
            {"role": "user", "content": "How do I handle token refresh?"},
            {"role": "assistant", "content": "Use refresh tokens with longer expiry. Rotate them on use."},
            {"role": "user", "content": "What about social login?"},
            {"role": "assistant", "content": "Use libraries like authlib. Store provider user IDs for account linking."},
            {"role": "user", "content": "How do I revoke access?"},
            {"role": "assistant", "content": "Maintain a token blacklist or use short-lived tokens with no refresh."},
        ]),
    ]

    for session_id, topic, messages in conversations:
        memory.store_conversation(session_id, messages)
        rprint(f"  Stored: {topic} ({session_id}) - {len(messages)} turns")

    rprint("\n[bold]Recalling relevant conversations...[/bold]")

    queries = [
        ("How do I search with filters?", None),
        ("Tell me about reading files", None),
        ("deployment process", None),
    ]

    for query, session in queries:
        rprint(f"\n[cyan]Query:[/cyan] '{query}'")
        results = memory.recall_relevant(query, session_id=session, limit=2)
        for r in results:
            rprint(f"  [score={r['score']:.3f}] [{r['session_id']}] {r['role']}: {r['content'][:60]}...")

    rprint(f"\n[bold]Full history of session-qdrant-002:[/bold]")
    history = memory.get_session_history("session-qdrant-002")
    for turn in history:
        rprint(f"  [{turn['role']}] {turn['content'][:70]}...")

    rprint(f"\n[bold]Summary of session-ml-009:[/bold]")
    summary = memory.summarize_session("session-ml-009")
    rprint(f"  {summary}")


if __name__ == "__main__":
    main()
