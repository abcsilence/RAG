"""Check that search finds the right part of the Constitution for known questions.
No API key needed: this only tests retrieval, which is where most RAG mistakes start.

    python eval.py

Run it after every change to chunking, the embedding model or TOP_K, and compare the score.
"""
import config
from rag import Retriever

# (question, sources that answer it). A question passes if any of them is in the search results.
TESTS = [
    ("What is the official language of Nepal?", ["Article 7"]),
    ("How many members does the House of Representatives have?", ["Article 84"]),
    ("Who appoints the Prime Minister?", ["Article 76"]),
    ("Is education free in Nepal?", ["Article 31"]),
    ("How can the Constitution be amended?", ["Article 274"]),
    ("What are the qualifications to become President?", ["Article 64"]),
    ("How many members are in the National Assembly?", ["Article 86"]),
    ("What does the national flag look like?", ["Article 8", "Schedule 1"]),
    ("What are the fundamental rights?", ["Part 3"]),
    ("How many provinces does Nepal have?", ["Schedule 4"]),
    ("Which province is Kathmandu in?", ["Schedule 4"]),
    ("Who can declare a state of emergency?", ["Article 273"]),
    ("How can the President be removed from office?", ["Article 101"]),
    ("What is the term of office of the President?", ["Article 63"]),
    ("Do citizens have a right to information?", ["Article 27"]),
    ("Can a Nepali citizen be exiled?", ["Article 45"]),
    ("At what age can a citizen vote?", ["Article 84"]),
    ("What are the duties of citizens?", ["Article 48"]),
    ("What is the national anthem of Nepal?", ["Article 9", "Schedule 2"]),
    ("What does secular mean in the constitution?", ["Article 4"]),
    ("What powers do local governments have?", ["Schedule 8"]),
    ("When was the constitution published?", ["The Constitution of Nepal"]),
    ("How many Articles does the constitution have?", ["Structure of the Constitution"]),
    ("What did the First Amendment change?", ["Amendments to the Constitution"]),
    ("Is torture allowed in Nepal?", ["Article 22"]),
    ("Is untouchability a crime?", ["Article 24"]),
    ("What rights do women have?", ["Article 38"]),
    ("What rights do children have?", ["Article 39"]),
    ("How many judges can the Supreme Court have?", ["Article 129"]),
    ("What is the capital of Nepal?", ["Article 288"]),
    ("Can a foreign woman married to a Nepali man become a citizen?", ["Article 11"]),
    ("Can the government take away private property?", ["Article 25"]),
    ("What happens if the Prime Minister loses a vote of confidence?", ["Article 100"]),
    ("What does Article 17 say?", ["Article 17"]),
    ("Summarize Schedule 5", ["Schedule 5"]),
    # Same topics asked in other words: search should not depend on the exact wording.
    ("Who appoints the Prime Minister of Nepal?", ["Article 76"]),
    ("What happens if no party has a majority in the House of Representatives?", ["Article 76"]),
    ("How is the Prime Minister chosen?", ["Article 76"]),
    ("Can the Prime Minister dissolve the House of Representatives?", ["Article 76", "Article 85"]),
    ("How is the President elected?", ["Article 62"]),
    ("Who is the head of state of Nepal?", ["Article 61"]),
    ("Who is the supreme commander of the Nepal Army?", ["Article 267"]),
    ("Who appoints the Chief Justice of Nepal?", ["Article 129"]),
    ("At what age do Supreme Court judges retire?", ["Article 131"]),
    ("How long is the term of the House of Representatives?", ["Article 85"]),
    ("What is the national animal of Nepal?", ["Article 9"]),
    ("Is there a death penalty in Nepal?", ["Article 16"]),
    ("is there death sentence in nepal", ["Article 16"]),
    ("Is capital punishment allowed?", ["Article 16"]),
    ("Is Nepal a Hindu state?", ["Article 4"]),
    ("Can a province choose its own official language?", ["Article 7"]),
    ("Can a child get citizenship through the mother?", ["Article 11"]),
]


def short_name(chunk):
    """ "Article 17 – Right to Freedom" -> "Article 17" """
    return chunk["source"].split(" – ")[0]


def main():
    retriever = Retriever()
    passed = 0
    for question, expected in TESTS:
        found = [short_name(chunk) for chunk in retriever.search(question)]
        rank = next((i for i, name in enumerate(found, start=1) if name in expected), None)
        if rank:
            passed += 1
            print(f"PASS #{rank}  {question}")
        else:
            print(f"FAIL     {question}\n         wanted {expected}, got {list(dict.fromkeys(found))}")
    print(f"\n{passed}/{len(TESTS)} questions found the right source ({config.EMBEDDING_MODEL}, TOP_K={config.TOP_K})")


if __name__ == "__main__":
    main()
