"""Workshop 01: Basic Memory Storage with Qdrant"""

import uuid
from datetime import datetime

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rich import print as rprint
from rich.panel import Panel

from workshop.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    MEMORIES_COLLECTION,
    QDRANT_API_KEY,
    QDRANT_URL,
)


def create_client() -> QdrantClient:
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(url=QDRANT_URL)


def create_embedder() -> TextEmbedding:
    return TextEmbedding(model_name=EMBEDDING_MODEL)


def setup_collection(client: QdrantClient, collection_name: str) -> None:
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        rprint(f"[green]Created collection: {collection_name}[/green]")
    else:
        rprint(f"[yellow]Collection {collection_name} already exists[/yellow]")


def store_memory(
    client: QdrantClient,
    embedder: TextEmbedding,
    collection_name: str,
    content: str,
    metadata: dict | None = None,
) -> str:
    memory_id = str(uuid.uuid4())
    embedding = list(embedder.embed([content]))[0].tolist()

    payload = {
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "type": "memory",
        **(metadata or {}),
    }

    client.upsert(
        collection_name=collection_name,
        points=[PointStruct(id=memory_id, vector=embedding, payload=payload)],
    )
    return memory_id


def search_memories(
    client: QdrantClient,
    embedder: TextEmbedding,
    collection_name: str,
    query: str,
    limit: int = 5,
) -> list[dict]:
    query_embedding = list(embedder.embed([query]))[0].tolist()

    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=limit,
    )

    return [
        {
            "id": r.id,
            "content": r.payload.get("content"),
            "score": r.score,
            "metadata": {k: v for k, v in r.payload.items() if k != "content"},
        }
        for r in results.points
    ]


def main():
    rprint(Panel("Workshop 01: Basic Memory Storage — Store & search with vector embeddings", style="bold blue"))

    client = create_client()
    embedder = create_embedder()
    setup_collection(client, MEMORIES_COLLECTION)

    example_memories = [
        # Preferences (25)
        ("The user prefers dark mode in all applications.", {"category": "preference"}),
        ("User's favorite programming language is Python.", {"category": "preference"}),
        ("The user likes to receive concise responses.", {"category": "preference"}),
        ("User prefers tabs over spaces for indentation.", {"category": "preference"}),
        ("The user likes type hints in Python code.", {"category": "preference"}),
        ("User prefers async/await over callbacks.", {"category": "preference"}),
        ("The user dislikes overly verbose documentation.", {"category": "preference"}),
        ("User prefers PostgreSQL over MySQL.", {"category": "preference"}),
        ("User likes snake_case for variable names.", {"category": "preference"}),
        ("User prefers dataclasses over plain dicts.", {"category": "preference"}),
        ("User wants error messages to include stack traces.", {"category": "preference"}),
        ("User prefers pull requests over direct commits.", {"category": "preference"}),
        ("User likes small, focused commits.", {"category": "preference"}),
        ("User prefers conventional commit messages.", {"category": "preference"}),
        ("User likes to use rich for terminal output.", {"category": "preference"}),
        ("User prefers uv over pip for package management.", {"category": "preference"}),
        ("User likes pytest fixtures over setup/teardown.", {"category": "preference"}),
        ("User prefers httpx over requests library.", {"category": "preference"}),
        ("User likes Pydantic for data validation.", {"category": "preference"}),
        ("User prefers SQLAlchemy 2.0 style queries.", {"category": "preference"}),
        ("User likes to use pathlib over os.path.", {"category": "preference"}),
        ("User prefers f-strings over .format().", {"category": "preference"}),
        ("User likes walrus operator for assignments.", {"category": "preference"}),
        ("User prefers match/case over if/elif chains.", {"category": "preference"}),
        ("User likes list comprehensions over map/filter.", {"category": "preference"}),
        # Tasks and deadlines (20)
        ("The project deadline is March 15th, 2025.", {"category": "task"}),
        ("Code review for the API module is due Friday.", {"category": "task"}),
        ("Need to update the CI/CD pipeline by end of sprint.", {"category": "task"}),
        ("Documentation needs to be completed before release.", {"category": "task"}),
        ("Security audit scheduled for next week.", {"category": "task"}),
        ("Database migration planned for Saturday.", {"category": "task"}),
        ("Performance testing due before launch.", {"category": "task"}),
        ("User needs to refactor the auth module.", {"category": "task"}),
        ("Integration tests need to be added for payment flow.", {"category": "task"}),
        ("API versioning needs to be implemented.", {"category": "task"}),
        ("Monitoring dashboards need to be set up.", {"category": "task"}),
        ("Load testing scheduled for Thursday.", {"category": "task"}),
        ("User needs to write ADR for caching strategy.", {"category": "task"}),
        ("Sprint retrospective is on Friday afternoon.", {"category": "task"}),
        ("Demo to stakeholders next Tuesday.", {"category": "task"}),
        ("Need to upgrade to Python 3.12.", {"category": "task"}),
        ("Dependency audit due by end of month.", {"category": "task"}),
        ("User needs to document the deployment process.", {"category": "task"}),
        ("Code coverage needs to reach 85%.", {"category": "task"}),
        ("API rate limiting needs to be implemented.", {"category": "task"}),
        # Context (30)
        ("User mentioned they are working on a machine learning project.", {"category": "context"}),
        ("The codebase uses FastAPI for the backend.", {"category": "context"}),
        ("User's team follows trunk-based development.", {"category": "context"}),
        ("The project targets Python 3.11+.", {"category": "context"}),
        ("User works at a fintech startup.", {"category": "context"}),
        ("The application handles 10k requests per second.", {"category": "context"}),
        ("The team has 8 engineers.", {"category": "context"}),
        ("The company is based in San Francisco.", {"category": "context"}),
        ("User has been at the company for 2 years.", {"category": "context"}),
        ("The product launched 6 months ago.", {"category": "context"}),
        ("User reports to the VP of Engineering.", {"category": "context"}),
        ("The team uses Slack for communication.", {"category": "context"}),
        ("Jira is used for project management.", {"category": "context"}),
        ("The company raised Series B funding.", {"category": "context"}),
        ("User works remotely 3 days a week.", {"category": "context"}),
        ("The team does daily standups at 10am.", {"category": "context"}),
        ("Code reviews require 2 approvals.", {"category": "context"}),
        ("The staging environment uses smaller instances.", {"category": "context"}),
        ("Feature flags are used for gradual rollouts.", {"category": "context"}),
        ("The team practices blameless postmortems.", {"category": "context"}),
        ("On-call rotation is weekly.", {"category": "context"}),
        ("SLA target is 99.9% uptime.", {"category": "context"}),
        ("User mentors two junior developers.", {"category": "context"}),
        ("The team does bi-weekly releases.", {"category": "context"}),
        ("User is part of the platform team.", {"category": "context"}),
        ("The company has 50 total employees.", {"category": "context"}),
        ("User previously worked at Google.", {"category": "context"}),
        ("The main product is a payment API.", {"category": "context"}),
        ("User is working on fraud detection features.", {"category": "context"}),
        ("The team uses GitHub for version control.", {"category": "context"}),
        # Technical details (35)
        ("The recommendation model uses collaborative filtering.", {"category": "technical"}),
        ("Vector embeddings are stored in Qdrant Cloud.", {"category": "technical"}),
        ("The API uses JWT tokens for authentication.", {"category": "technical"}),
        ("Redis is used for caching user sessions.", {"category": "technical"}),
        ("The frontend is built with React and TypeScript.", {"category": "technical"}),
        ("Docker containers are orchestrated with Kubernetes.", {"category": "technical"}),
        ("The ML pipeline uses MLflow for experiment tracking.", {"category": "technical"}),
        ("Feature flags are managed through LaunchDarkly.", {"category": "technical"}),
        ("PostgreSQL 15 is the primary database.", {"category": "technical"}),
        ("Alembic handles database migrations.", {"category": "technical"}),
        ("Celery is used for background task processing.", {"category": "technical"}),
        ("RabbitMQ is the message broker.", {"category": "technical"}),
        ("Prometheus collects application metrics.", {"category": "technical"}),
        ("Grafana is used for dashboards.", {"category": "technical"}),
        ("Sentry handles error tracking.", {"category": "technical"}),
        ("AWS S3 stores uploaded files.", {"category": "technical"}),
        ("CloudFront is the CDN.", {"category": "technical"}),
        ("Terraform manages infrastructure.", {"category": "technical"}),
        ("GitHub Actions runs CI/CD pipelines.", {"category": "technical"}),
        ("The API uses OpenAPI 3.0 spec.", {"category": "technical"}),
        ("GraphQL is used for the mobile API.", {"category": "technical"}),
        ("WebSockets handle real-time updates.", {"category": "technical"}),
        ("The search uses Elasticsearch.", {"category": "technical"}),
        ("Embedding model is BAAI/bge-small-en-v1.5.", {"category": "technical"}),
        ("The API response time p99 is 120ms.", {"category": "technical"}),
        ("Database connection pool size is 20.", {"category": "technical"}),
        ("Rate limiting is 1000 requests per minute.", {"category": "technical"}),
        ("The JWT tokens expire after 1 hour.", {"category": "technical"}),
        ("Refresh tokens are valid for 30 days.", {"category": "technical"}),
        ("Password hashing uses bcrypt with cost 12.", {"category": "technical"}),
        ("API keys are hashed with SHA-256.", {"category": "technical"}),
        ("The database uses read replicas.", {"category": "technical"}),
        ("Connection pooling uses PgBouncer.", {"category": "technical"}),
        ("The cache TTL is 5 minutes.", {"category": "technical"}),
        ("Logs are shipped to DataDog.", {"category": "technical"}),
        # Skills (20)
        ("User learned how to use Qdrant filters yesterday.", {"category": "skill"}),
        ("User knows how to deploy with GitHub Actions.", {"category": "skill"}),
        ("User is familiar with PyTorch and TensorFlow.", {"category": "skill"}),
        ("User has experience with Apache Kafka.", {"category": "skill"}),
        ("User can write complex SQL window functions.", {"category": "skill"}),
        ("User knows Kubernetes helm charts.", {"category": "skill"}),
        ("User can debug memory leaks in Python.", {"category": "skill"}),
        ("User is proficient with git rebase.", {"category": "skill"}),
        ("User knows how to profile Python code.", {"category": "skill"}),
        ("User can set up Prometheus alerting rules.", {"category": "skill"}),
        ("User has experience with load testing.", {"category": "skill"}),
        ("User knows how to optimize PostgreSQL queries.", {"category": "skill"}),
        ("User can write custom pytest plugins.", {"category": "skill"}),
        ("User is familiar with OAuth 2.0 flows.", {"category": "skill"}),
        ("User knows how to set up SSL certificates.", {"category": "skill"}),
        ("User can configure nginx reverse proxy.", {"category": "skill"}),
        ("User has experience with Redis Cluster.", {"category": "skill"}),
        ("User knows how to use AWS Lambda.", {"category": "skill"}),
        ("User can write Terraform modules.", {"category": "skill"}),
        ("User is familiar with OpenTelemetry.", {"category": "skill"}),
    ]

    rprint("\n[bold]Storing memories...[/bold]")
    for content, metadata in example_memories:
        memory_id = store_memory(client, embedder, MEMORIES_COLLECTION, content, metadata)
        rprint(f"  Stored: {content[:50]}... (id: {memory_id[:8]})")

    rprint("\n[bold]Searching memories...[/bold]")
    queries = [
        "What are the user's preferences?",
        "When is the deadline?",
        "What is the user working on?",
    ]

    for query in queries:
        rprint(f"\n[cyan]Query:[/cyan] {query}")
        results = search_memories(client, embedder, MEMORIES_COLLECTION, query, limit=2)
        for r in results:
            rprint(f"  [green]Score {r['score']:.3f}:[/green] {r['content']}")


if __name__ == "__main__":
    main()
