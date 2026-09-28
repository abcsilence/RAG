"""Settings shared by ingest.py, rag.py, app.py and eval.py."""
from pathlib import Path

ROOT = Path(__file__).parent
PDF_PATH = ROOT / "Constitution of Nepal (2nd amd. English)_xf33zb3.pdf"
DATA_DIR = ROOT / "data"
INDEX_PATH = DATA_DIR / "constitution.faiss"
CHUNKS_PATH = DATA_DIR / "chunks.json"

# Embedding model. It scored best in eval.py of the models tried: all-MiniLM-L6-v2 (from
# the course) found 30 of the first 35 test answers, BAAI/bge-small-en-v1.5 33 and this one 34.
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
# bge models find passages better when the question starts with this instruction.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Chunk size in tokens. The model reads up to 512, but eval.py found answers better
# with small chunks (300 and 400 tokens both scored lower).
MAX_CHUNK_TOKENS = 190

# Groq LLM. The 120b model has the same free-tier limits as openai/gpt-oss-20b, and in
# tests it rewrote follow-up questions better and mixed up fewer Articles.
LLM_MODEL = "openai/gpt-oss-120b"
REASONING_EFFORT = "low"  # gpt-oss only: "low", "medium" or "high". Use None for other models.

# Retrieval
# Chunks sent to the LLM for a normal question. In eval.py 5 found 48/52 answers, 8 found 49
# and 10 found 50, but each chunk adds ~140 tokens to every request (free tier: 8K tokens a minute).
TOP_K = 8
MAX_REFERENCE_CHUNKS = 8  # chunks sent when the question names an Article, Part or Schedule
MAX_HISTORY_TURNS = 3     # earlier question/answer pairs sent to the LLM
