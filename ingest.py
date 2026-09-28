"""Build the search index for the Constitution of Nepal chatbot.

Run it once, and again whenever you change the chunking:
    python ingest.py

PDF -> clean lines -> sections (Preamble, 308 Articles, 9 Schedules) -> chunks -> embeddings -> FAISS
Creates data/chunks.json and data/constitution.faiss.
"""
import json
import re

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import config

PART_RE = re.compile(r"^Part\s*-\s*(\d+)$")          # "Part-3"
ARTICLE_RE = re.compile(r"^(\d{1,3})\.\s*[A-Z]")     # "17. Right to Freedom: (1) No person ..."
SCHEDULE_RE = re.compile(r"^Schedule\s*-\s*(\d+)$")  # "Schedule-4"
# Amendment footnote at the bottom of a page, e.g. " Amended by the First Amendment."
# The same marker symbol (a private-use character) appears in the text where the change was made.
FOOTNOTE_RE = re.compile(r"^([-])\s*((?:Amended|Inserted) by the \w+ Amendment)\.?$")
NOTE_PREFIX_RE = re.compile(r"^\[[^\]]*\]\s*")       # "[Amended by ...] " at the start of a line
# A new clause starts at "(1)", "(a)", "12.", "Provided that", "Explanation", "Province No."
# (Schedule 4) or an ALL-CAPS word (Preamble).
CLAUSE_START_RE = re.compile(r"^(\(\w{1,5}\)\s|\d{1,3}\.\s|Provided that|Explanation|Province No\.|[A-Z]{4,}\s)")
# A line ending like this continues on the next line ("... pursuant to clause" / "(2) within thirty
# days ..."), so a "(2)" at the start of that next line is a reference, not a new clause.
CROSS_REFERENCE_RE = re.compile(r"\b(sub-)?clauses?$", re.IGNORECASE)
# A lettered group heading inside a long Article, e.g. "(h) Policies relating to Basic Needs of Citizens:"
GROUP_RE = re.compile(r"^\([a-z]{1,2}\)\s.{3,120}:-?$")


# ---------- 1. Read and clean ----------

def read_lines(reader):
    """All text lines of the PDF as (page_number, line), without headers, page numbers and footnotes."""
    raw = []
    for page_no, page in enumerate(reader.pages, start=1):
        # Layout mode keeps words whole (the default mode breaks some, like "ELIMINA TING").
        for line in page.extract_text(extraction_mode="layout").split("\n"):
            line = " ".join(line.split())  # collapse the extra spaces of justified text
            if line:
                raw.append((page_no, line))

    # Replace each footnote marker in the text with its note, e.g. "[Amended by the First Amendment]".
    notes = {m[1]: m[2] for _, line in raw if (m := FOOTNOTE_RE.match(line))}
    lines = []
    for page_no, line in raw:
        if is_noise(line):
            continue
        for marker, note in notes.items():
            line = line.replace(marker, f"[{note}] ")
        lines.append((page_no, " ".join(line.split())))
    return lines


def is_noise(line):
    return bool(
        "lawcommission.gov.np" in line                   # page header
        or re.fullmatch(r"\d{1,3}", line)                # page number
        or FOOTNOTE_RE.match(line)                       # amendment footnote (moved into the text)
        or re.fullmatch(r"S\.\s*N\.\s+Matters", line)    # table header in Schedules 5-9
    )


# ---------- 2. Split into sections ----------

def new_section(kind, number, title, part=None):
    return {"kind": kind, "number": number, "title": title, "part": part, "lines": []}


def split_sections(lines):
    """Group the lines into sections: front page, Preamble, Articles 1-308 and Schedules 1-9."""
    sections = [new_section("front", None, "Publication and amendment dates")]
    part = None  # (number, title) of the current Part
    last_article = 0
    i = 0
    while i < len(lines):
        page, line = lines[i]
        bare = NOTE_PREFIX_RE.sub("", line)  # "[Amended by ...] Schedule-3" -> "Schedule-3"
        part_match = PART_RE.match(bare)
        article_match = ARTICLE_RE.match(bare)
        schedule_match = SCHEDULE_RE.match(bare)

        if line == "Preamble:" and len(sections) == 1:
            sections.append(new_section("preamble", None, "Preamble"))
        elif part_match and last_article < 308:
            part = (int(part_match[1]), lines[i + 1][1])  # the Part title is on the next line
            i += 2
            continue
        # Article numbers go up one by one, which filters out numbered list items.
        elif part and article_match and last_article < int(article_match[1]) <= last_article + 3:
            title = article_title(lines, i)
            if title:
                last_article = int(article_match[1])
                sections.append(new_section("article", last_article, title, part))
        elif schedule_match and last_article == 308:
            # "Schedule-4" / "(Relating to clause (3) of Article 56)" / "Provinces, and Districts ..."
            # The "(Relating to ...)" note can wrap onto a second line.
            title_line = i + 1
            if lines[title_line][1].startswith("(Relating"):
                while not lines[title_line][1].endswith(")") and title_line < i + 4:
                    title_line += 1
                title_line += 1
            sections.append(new_section("schedule", int(schedule_match[1]), lines[title_line][1]))

        sections[-1]["lines"].append((page, line))
        i += 1
    return sections


def article_title(lines, i):
    """Title of the Article whose heading is lines[i] (e.g. "Right to Freedom"), or None if
    the line is not really a heading. The title ends at the first ":" and may wrap onto the
    next line. Two headings (242 and 245) have no ":" and go straight on to "(1)"."""
    heading = NOTE_PREFIX_RE.sub("", lines[i][1])
    for _, next_line in lines[i + 1 : i + 3]:
        if ":" in heading or " (1)" in heading:
            break
        heading += " " + next_line
    match = re.match(r"\d{1,3}\.\s*(.+?)\s*(?::|(?=\(1\)))", heading)
    return match[1] if match else None


def fix_schedule_4(section, reader):
    """Schedule 4 is a table with the provinces side by side, which layout mode mixes up.
    Plain mode reads it one column at a time, and the numbering restarts at 1 for every
    province, so the district list of each province can be rebuilt from it."""
    pages = sorted({page for page, _ in section["lines"]})
    provinces, districts = [], []  # districts: one (page, [names]) per province
    for page in pages:
        for line in reader.pages[page - 1].extract_text().split("\n"):
            line = " ".join(line.split())
            provinces += re.findall(r"Province No\. (\d)", line)
            item = re.match(r"(\d{1,2})\.\s*(.+)", line)
            if item:
                if item[1] == "1":
                    districts.append((page, []))
                districts[-1][1].append(item[2])
            elif districts and line and not is_noise(line) and "Province No." not in line:
                districts[-1][1][-1] += " " + line  # a district name that wrapped onto the next line

    if len(provinces) != len(districts):
        print(f"WARNING: Schedule 4 has {len(provinces)} province headings but {len(districts)} district lists")
    total = sum(len(names) for _, names in districts)
    heading = section["lines"][:3]  # "Schedule-4", "(Relating to ...)", title
    summary = (heading[0][0], f"There are {len(provinces)} Provinces with {total} districts in total.")
    section["lines"] = heading + [summary] + [
        (page, f"Province No. {number} ({len(names)} districts): {', '.join(names)}.")
        for number, (page, names) in zip(provinces, districts)
    ]


def add_summaries(sections):
    """Add generated sections: a list of Articles for each Part, an overview of the whole
    Constitution and a list of amended provisions. Questions like "What are the fundamental
    rights?", "How many Parts are there?" or "What did the First Amendment change?" then
    find a direct answer."""
    articles = [s for s in sections if s["kind"] == "article"]
    schedules = [s for s in sections if s["kind"] == "schedule"]
    parts = {}
    for article in articles:
        parts.setdefault(article["part"], []).append(article)

    overview = new_section("toc", None, "Structure of the Constitution")
    overview_lines = [
        f"The Constitution of Nepal has a Preamble, {len(parts)} Parts with {len(articles)} Articles, "
        f"and {len(schedules)} Schedules."
    ]
    for (number, title), part_articles in parts.items():
        span = article_span(part_articles)
        overview_lines.append(f"Part {number} – {title}: {span}")
        toc = new_section("toc", None, f"Part {number} – {title} (list of Articles)", (number, title))
        first_page = part_articles[0]["lines"][0][0]
        toc_lines = [f"Part {number} – {title} contains {span}:"]
        toc_lines += [f"Article {a['number']} – {a['title']}" for a in part_articles]
        toc["lines"] = [(first_page, line) for line in toc_lines]
        sections.append(toc)
    overview_lines += [f"Schedule {s['number']} – {s['title']}" for s in schedules]
    overview["lines"] = [(None, line) for line in overview_lines]

    changed = {}  # "Amended by the First Amendment" -> ["Article 42 – Right to Social Justice", ...]
    for section in sections:
        text = "\n".join(line for _, line in section["lines"])
        for note in dict.fromkeys(re.findall(r"\[((?:Amended|Inserted) by the \w+ Amendment)\]", text)):
            changed.setdefault(note, []).append(describe(section)[0])
    amendments = new_section("toc", None, "Amendments to the Constitution")
    amendments["lines"] = [(None, "Provisions changed by amendments, as marked in the text:")] + [
        (None, f"{note}: {'; '.join(sources)}.") for note, sources in changed.items()
    ]
    sections += [overview, amendments]


def article_span(articles):
    first, last = articles[0]["number"], articles[-1]["number"]
    return f"Article {first}" if first == last else f"Articles {first} to {last}"


def check_structure(sections):
    articles = [s["number"] for s in sections if s["kind"] == "article"]
    parts = {s["part"] for s in sections if s["kind"] == "article"}
    schedules = [s["number"] for s in sections if s["kind"] == "schedule"]
    print(f"Found {len(parts)} Parts, {len(articles)} Articles and {len(schedules)} Schedules")
    missing = sorted(set(range(1, max(articles) + 1)) - set(articles))
    if missing:
        print("WARNING: these Articles were not found:", missing)


# ---------- 3. Chunk ----------

def clause_units(lines):
    """Group a section's lines into clauses, as [page, text] pairs."""
    units = []
    for page, line in lines:
        if not units or (CLAUSE_START_RE.match(line) and not CROSS_REFERENCE_RE.search(units[-1][1])):
            units.append([page, line])
        else:
            units[-1][1] += "\n" + line
    return units


def pack(units, max_tokens, count_tokens):
    """Join clauses into chunks of at most max_tokens tokens. Returns (page, text, group)
    tuples, where group is the lettered group heading (see GROUP_RE) the chunk starts in."""
    chunks = []
    current = []  # (page, text, group, tokens) of each piece in the chunk being built
    group = None
    for page, unit in units:
        is_group_heading = bool(GROUP_RE.match(" ".join(unit.split())))
        if is_group_heading:
            group = " ".join(unit.split())
        for piece in cut(unit, max_tokens, count_tokens):
            tokens = count_tokens(piece)
            if current and sum(p[3] for p in current) + tokens > max_tokens:
                # Keep an intro line like "Provided that:" together with the clauses it introduces.
                carry = [current.pop()] if len(current) > 1 and ends_with_colon(current[-1][1]) else []
                chunks.append(finish(current))
                current = carry
            current.append((page, piece, None if is_group_heading else group, tokens))
    if current:
        chunks.append(finish(current))
    return chunks


def cut(text, max_tokens, count_tokens):
    """Split a clause that is too long on its own into word windows of at most max_tokens."""
    if count_tokens(text) <= max_tokens:
        return [text]
    pieces, words, size = [], [], 0
    for word in text.split():
        tokens = count_tokens(word)
        if words and size + tokens > max_tokens:
            pieces.append(" ".join(words))
            words, size = [], 0
        words.append(word)
        size += tokens
    pieces.append(" ".join(words))
    return pieces


def ends_with_colon(text):
    return text.rstrip(" -").endswith(":")


def finish(pieces):
    page, _, group, _ = pieces[0]
    return page, "\n".join(p[1] for p in pieces), group


def describe(section):
    """(source, label): a short name to show as a citation, and the full path used as context."""
    kind, number, title = section["kind"], section["number"], section["title"]
    if kind == "article":
        part_number, part_title = section["part"]
        source = f"Article {number} – {title}"
        return source, f"Part {part_number} – {part_title} > {source}"
    if kind == "schedule":
        source = f"Schedule {number} – {title}"
    elif kind == "front":
        source = f"The Constitution of Nepal – {title.lower()}"
    else:  # preamble, toc
        source = title
    return source, source


def make_chunks(sections, count_tokens):
    chunks = []
    for section in sections:
        source, label = describe(section)
        # Table-of-contents lines are chunked line by line; everything else clause by clause.
        units = section["lines"] if section["kind"] == "toc" else clause_units(section["lines"])
        for page, text, group in pack(units, config.MAX_CHUNK_TOKENS, count_tokens):
            chunks.append({
                "id": len(chunks),
                "kind": section["kind"],
                "source": source,
                "label": f"{label} > {group.rstrip(':-')}" if group else label,
                "part": section["part"][0] if section["part"] else None,
                "article": section["number"] if section["kind"] == "article" else None,
                "schedule": section["number"] if section["kind"] == "schedule" else None,
                "page": page,
                "text": text,
            })
    return chunks


# ---------- 4. Embed and save ----------

def main():
    reader = PdfReader(config.PDF_PATH)
    sections = split_sections(read_lines(reader))
    for section in sections:
        if section["kind"] == "schedule" and section["number"] == 4:
            fix_schedule_4(section, reader)
    check_structure(sections)
    add_summaries(sections)

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    # Some text (like district names) has many tokens per word, so chunk sizes are measured
    # in tokens, with the model's own tokenizer.
    chunks = make_chunks(sections, lambda text: len(model.tokenizer.tokenize(text)))

    print(f"Embedding {len(chunks)} chunks with {config.EMBEDDING_MODEL} ...")
    texts = [f"{chunk['label']}\n{chunk['text']}" for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True).astype("float32")

    too_long = [c["source"] for c, t in zip(chunks, texts)
                if len(model.tokenizer(t)["input_ids"]) > model.max_seq_length]
    if too_long:
        print(f"WARNING: {len(too_long)} chunks are longer than {model.max_seq_length} tokens and get "
              f"cut off when embedded (lower MAX_CHUNK_TOKENS in config.py): {too_long[:5]}")

    # Inner product of normalized vectors = cosine similarity. Exact search is instant for ~600 vectors.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    config.DATA_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(config.INDEX_PATH))
    config.CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=1))
    print(f"Saved {index.ntotal} vectors to {config.INDEX_PATH.relative_to(config.ROOT)} "
          f"and the chunks to {config.CHUNKS_PATH.relative_to(config.ROOT)}")

    example = next(c for c in chunks if c["article"] == 17)
    print(f"\nExample chunk (page {example['page']}):\n{example['label']}\n{example['text']}")


if __name__ == "__main__":
    main()
