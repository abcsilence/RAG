"""Answer questions about the Constitution of Nepal: find the matching chunks, then ask the LLM.

Chat in the terminal:
    python rag.py

server.py (the API behind the web chat) and eval.py (tests) use the functions in this file.
"""
import json
import os
import re
from getpass import getpass

import faiss
import groq
import numpy as np
from dotenv import load_dotenv
from huggingface_hub.utils import logging as hub_logging
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as transformers_logging

import config

hub_logging.set_verbosity_error()             # hide the "unauthenticated requests to the HF Hub" warning
transformers_logging.disable_progress_bar()   # hide the "Loading weights" bar

SYSTEM_PROMPT = """You answer questions about the Constitution of Nepal (2015), including its First (2016) and Second (2020) Amendments.

Rules:
- Use ONLY the excerpts in the Context of the latest message. Do not use outside knowledge, even if you think you know the answer.
- Cite where each fact comes from, like (Article 17(2)(a)), (Preamble) or (Schedule 4). Use only this citation style, never marks like 【】.
- If the Context does not answer the question, say "I couldn't find that in the Constitution of Nepal." If something in the Context is related, mention it briefly. Never guess.
- When asked what an Article says, stay close to its exact wording.
- Keep answers clear and short. Use bullet points for lists.
- You are not a lawyer. For a personal legal problem, explain what the Constitution says and suggest talking to a lawyer.
- If the user only greets you or asks what you can do, answer briefly and invite a question about the Constitution."""

REWRITE_PROMPT = """Rewrite the user's last question as a complete question that can be understood without the conversation. It will be used to search the Constitution of Nepal.
Use the conversation to fill in what the question leaves out: replace words like "it", "that" or "this Article" with what they refer to, and add the topic being discussed when the question depends on it. Keep any Article, Part or Schedule numbers.
If the question is already complete on its own, return it unchanged. Reply with the question only."""

# "Article 17", "articles 16 and 17", "art. 5", "Part 3", "Schedule-4", and the Nepali धारा/भाग/अनुसूची
REFERENCE_RE = re.compile(
    r"(\b(?:articles?|art\.?|dhara|parts?|bhag|schedules?|anusuchi)|धारा|भाग|अनुसूची)"
    r"\s*-?\s*(\d{1,3}(?:\s*(?:,|and|&|or)\s*\d{1,3})*)",
    re.IGNORECASE,
)
REFERENCE_KINDS = {
    "article": "article", "art": "article", "dhara": "article", "धारा": "article",
    "part": "part", "bhag": "part", "भाग": "part",
    "schedule": "schedule", "anusuchi": "schedule", "अनुसूची": "schedule",
}
NEPALI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
CITATION_MARK_RE = re.compile(r"\s*【[^】]*】")  # gpt-oss models sometimes add marks like 【?】
STOPWORDS = set(
    "a an the of to in on for and or is are was be by with what who how which does do can shall "
    "this that it its as at from any there has have if will".split()
)


# ---------- Search ----------

def keywords(text):
    """Lowercase words without filler words, for keyword (BM25) search."""
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS]


def find_references(question):
    """Articles, Parts and Schedules named in a question, e.g. {("article", 17), ("part", 3)}."""
    references = set()
    for word, numbers in REFERENCE_RE.findall(question.translate(NEPALI_DIGITS)):
        kind = REFERENCE_KINDS[word.lower().rstrip("s.")]
        references.update((kind, int(n)) for n in re.findall(r"\d+", numbers))
    return references


class Retriever:
    """Finds the chunks of the Constitution that match a question."""

    def __init__(self):
        if not config.INDEX_PATH.exists():
            raise SystemExit("No index found. Run `python ingest.py` first.")
        self.chunks = json.loads(config.CHUNKS_PATH.read_text())
        self.index = faiss.read_index(str(config.INDEX_PATH))
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.bm25 = BM25Okapi([keywords(f"{c['label']}\n{c['text']}") for c in self.chunks])

    def rank(self, question):
        """Positions of all chunks, best match first. Combines two searches: by meaning
        (embeddings), which finds answers worded differently from the question, and by
        keywords (BM25), which finds exact terms like "Prime Minister" or "death penalty".
        The two rankings are merged with Reciprocal Rank Fusion: a chunk gets 1 / (60 + rank)
        points from each list. In eval.py this found 48 of the first 50 answers in the top 5,
        against 46 for embeddings alone."""
        vector = self.model.encode([config.QUERY_PREFIX + question], normalize_embeddings=True).astype("float32")
        by_meaning = self.index.search(vector, self.index.ntotal)[1][0]  # every chunk (there are ~600)
        keyword_scores = self.bm25.get_scores(keywords(question))
        by_keywords = [i for i in np.argsort(-keyword_scores) if keyword_scores[i] > 0]

        points = {}
        for ranking in (by_meaning, by_keywords):
            for rank, i in enumerate(ranking, start=1):
                points[int(i)] = points.get(int(i), 0) + 1 / (60 + rank)
        return sorted(points, key=points.get, reverse=True)

    def search(self, question):
        """The chunks that best match a question, best first."""
        ranked = [self.chunks[i] for i in self.rank(question)]

        references = find_references(question)
        named = [c for c in ranked if any(c[kind] == number for kind, number in references)]
        if not named:
            return ranked[: config.TOP_K]

        # The question names an Article, Part or Schedule. Embeddings are bad with numbers
        # ("Article 17" looks almost the same as "Article 71"), so fetch those chunks directly.
        named.sort(key=lambda c: c["kind"] != "toc")  # a Part's list of Articles first
        return sorted(named[: config.MAX_REFERENCE_CHUNKS], key=lambda c: c["id"])  # in document order


# ---------- Answer ----------

def llm_options():
    return {"reasoning_effort": config.REASONING_EFFORT} if config.REASONING_EFFORT else {}


def rewrite_question(client, question, history):
    """Turn a follow-up like "What are its exceptions?" into a question that can be searched alone."""
    conversation = "\n".join(f"{m['role'].title()}: {m['content']}" for m in history)
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": f"Conversation:\n{conversation}\n\nLast question: {question}"},
        ],
        **llm_options(),
    )
    return (response.choices[0].message.content or question).strip().strip('"')


def format_context(chunks):
    blocks = []
    for chunk in chunks:
        page = f" | page {chunk['page']}" if chunk["page"] else ""
        blocks.append(f"[{chunk['label']}{page}]\n{chunk['text']}")
    return "\n\n".join(blocks)


def ask(client, retriever, question, history):
    """Answer a question, as a stream of text pieces. The answer cites the Articles it uses.
    history holds the earlier turns as {"role": "user" or "assistant", "content": ...} dicts."""
    history = history[-2 * config.MAX_HISTORY_TURNS :]
    search_query = rewrite_question(client, question, history) if history else question
    # These chunks are what the LLM reads, not a list of sources: most are only near matches.
    chunks = retriever.search(search_query)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"},
    ]
    response = client.chat.completions.create(
        model=config.LLM_MODEL, messages=messages, stream=True, **llm_options()
    )
    return stream_text(response)


def stream_text(response):
    """The answer text piece by piece, without 【...】 citation marks. Text from a "【" on is
    held back until its "】" arrives, because a mark can be split over several pieces."""
    pending = ""
    for event in response:
        if not (event.choices and event.choices[0].delta.content):
            continue
        pending = CITATION_MARK_RE.sub("", pending + event.choices[0].delta.content)
        start = pending.find("【")
        if start == -1:
            ready, pending = pending, ""
        else:  # the space before a mark goes with it
            ready, pending = pending[:start].rstrip(), pending[start:]
        if ready:
            yield ready
    if pending:
        yield pending


# ---------- API key and errors ----------

def load_api_key():
    """The Groq API key from the .env file (or the environment), or None."""
    load_dotenv(config.ROOT / ".env")
    return os.getenv("GROQ_API_KEY") or None


def explain_error(error):
    if isinstance(error, groq.AuthenticationError):
        return "Groq rejected the API key. Check GROQ_API_KEY in the .env file."
    if isinstance(error, groq.RateLimitError):
        return "Groq's free-tier rate limit was reached. Wait a minute and try again."
    if isinstance(error, groq.NotFoundError):
        return f"Groq can't find the model '{config.LLM_MODEL}'. Change LLM_MODEL in config.py."
    if isinstance(error, groq.APIConnectionError):
        return "Couldn't reach Groq. Check your internet connection."
    return f"Groq error: {error}"


# ---------- Terminal chat ----------

def chat_in_terminal():
    client = groq.Groq(api_key=load_api_key() or getpass("Enter your Groq API key: "))
    print("Loading the Constitution ...")
    retriever = Retriever()
    history = []
    print("Ask anything about the Constitution of Nepal. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            print("\nBot: ", end="", flush=True)
            answer = ""
            for text in ask(client, retriever, question, history):
                print(text, end="", flush=True)
                answer += text
            print()
        except groq.APIError as error:
            print(f"\n{explain_error(error)}")
            continue
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


if __name__ == "__main__":
    chat_in_terminal()
