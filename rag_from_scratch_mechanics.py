"""rag_from_scratch_mechanics.py

RAG implemented without a framework, to expose the mechanics a framework hides.

    [ Raw Documents ] ──> Chunking & Windowing ──> [ Text Chunks ]
                                                        │
                                                Embedding API
                                                        ▼
    [ User Query ] ─────> Embedding API ───────> Cosine Similarity
                                                        │
                                                 Selection Policy
                                                 (top-k | threshold)
                                                        ▼
    [ LLM Generation ] <── Context Prompting <── [ Selected Chunks ]

Invariant that governs the whole file: documents and queries MUST be embedded
with the SAME model. Different models are different vector spaces, and cosine
between two different spaces is noise centred on zero, not low similarity.
"""

import math
import os
import statistics
from collections.abc import Sequence

import voyageai
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

anthropic_client = Anthropic()
voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

# One model for documents AND queries. This is the invariant.
EMBED_MODEL = "voyage-3.5"
LLM_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------
# 1. CHUNKING
# ---------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 120,
    overlap: int = 25,
    min_chunk_size: int = 40,
) -> list[str]:
    """Split text into fixed-size windows with overlap.

    step = chunk_size - overlap  →  how far the window advances each iteration.

    Two guards the naive `range(0, len(text), step)` version lacks:
      - the loop stops as soon as a window has reached the end of the text,
        instead of emitting a runt chunk made only of the remainder;
      - windows shorter than `min_chunk_size` are dropped, so a stray tail
        like " balance reversal." never becomes an indexable unit.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    if not text:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        if start + chunk_size >= len(text):
            break
        start += step

    return [c for c in chunks if len(c) >= min_chunk_size] or chunks[:1]


# ---------------------------------------------------------------------
# 2. EMBEDDING
# ---------------------------------------------------------------------

def embed_texts(texts: list[str], input_type: str) -> list[list[float]]:
    """Batch-embed with EMBED_MODEL.

    `input_type` is "document" or "query". Voyage encodes the two asymmetrically
    *within the same space* — that asymmetry is trained and intentional. The
    model, unlike the input_type, must never differ between the two calls.
    """
    if input_type not in ("document", "query"):
        raise ValueError('input_type must be "document" or "query"')
    return voyage_client.embed(texts, model=EMBED_MODEL, input_type=input_type).embeddings


def build_index_from_docs(docs: list[str]) -> list[dict]:
    """Index a list of already-separated documents. No chunking."""
    return [
        {"text": text, "embedding": embedding}
        for text, embedding in zip(docs, embed_texts(docs, input_type="document"))
    ]


def build_index_from_text(text: str, **chunk_kwargs) -> list[dict]:
    """Chunk a single long document, then index the chunks."""
    return build_index_from_docs(chunk_text(text, **chunk_kwargs))


# ---------------------------------------------------------------------
# 3. COSINE SIMILARITY
# ---------------------------------------------------------------------

def cosine(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """Cosine similarity, with the dimension guard `zip` does not give you.

    Without this check, comparing vectors of different lengths silently
    truncates to the shorter one and returns a plausible-looking float.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError(
            f"dimension mismatch: {len(vector_a)} vs {len(vector_b)} — "
            "documents and query were probably embedded with different models"
        )

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("zero-norm vector — the embedding call returned nothing usable")

    return dot_product / (norm_a * norm_b)


def text_similarity(a: str, b: str, types=("document", "document")) -> float:
    va = embed_texts([a], input_type=types[0])[0]
    vb = embed_texts([b], input_type=types[1])[0]
    return cosine(va, vb)

# ---------------------------------------------------------------------
# 4. RETRIEVAL — scoring is one step, selection policy is another
# ---------------------------------------------------------------------

def score_all(query: str, index: list[dict]) -> list[tuple[float, str]]:
    """Score every chunk against the query. Sorted descending. Embeds once."""
    query_embedding = embed_texts([query], input_type="query")[0]
    scored = [(cosine(query_embedding, item["embedding"]), item["text"]) for item in index]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored


def select_top_k(scored: list[tuple[float, str]], k: int = 3) -> list[tuple[float, str]]:
    """Fixed-size selection. Always returns min(k, len) results — never empty."""
    return scored[:k]


def select_above_threshold(
    scored: list[tuple[float, str]], threshold: float
) -> list[tuple[float, str]]:
    """Score-gated selection. CAN return empty — that is the whole point."""
    return [pair for pair in scored if pair[0] >= threshold]


# ---------------------------------------------------------------------
# 5. AUGMENTATION + GENERATION
# ---------------------------------------------------------------------

NO_CONTEXT_ANSWER = (
    "No chunk cleared the similarity threshold. "
    "Abstaining without calling the model."
)

SYSTEM_PROMPT = (
    "You answer questions about DistributedOrderSystem orders using ONLY the "
    "context provided by the user. If the answer is not in the context, say "
    "that there is not enough information — never invent data that does not "
    "appear in the context."
)


def generate_answer(query: str, selected: list[tuple[float, str]]) -> str:
    """Assemble the grounded prompt and generate.

    Short-circuits on empty selection: with nothing retrieved there is nothing
    to ground on, so the call is skipped entirely. The threshold is not only a
    quality gate — it is the only place in this pipeline that can decline.
    """
    if not selected:
        return NO_CONTEXT_ANSWER

    context = "\n".join(f"- {text}" for _, text in selected)
    user_prompt = f"Retrieved context:\n{context}\n\nQuery: {query}"

    response = anthropic_client.messages.create(
        model=LLM_MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------
# 6. SELF-CHECK — run before trusting any number below
# ---------------------------------------------------------------------

def self_check() -> None:
    """Four assertions that separate a broken retriever from a weak one.

    A retriever whose scores hover around zero with mixed signs is not finding
    'low similarity' — its vectors are unrelated. For random vectors in R^d,
    cosine has mean 0 and standard deviation 1/sqrt(d), so on a 1024-dim model
    anything inside roughly +/-0.03 is indistinguishable from noise.
    """
    print("=== SELF-CHECK ===")

    v = embed_texts(["refund"], input_type="document")[0]
    dim = len(v)
    noise_sigma = 1 / math.sqrt(dim)
    print(f"model={EMBED_MODEL}  dim={dim}  noise sigma=1/sqrt(d)={noise_sigma:.4f}")

    same = cosine(v, v)
    print(f"1. cosine(v, v)                       = {same:.6f}   (must be 1.0)")
    assert same > 0.9999, "cosine() itself is wrong"

    determinism = text_similarity("refund", "refund")
    print(f"2. cosine of two identical embeds     = {determinism:.6f}   (must be ~1.0)")
    assert determinism > 0.99, "the embedding API is not deterministic — check parsing"

    unrelated = text_similarity("refund policy", "volcano geology")
    print(f"3. cosine of unrelated texts          = {unrelated:.6f}   (low, but > {3*noise_sigma:.3f})")
    assert unrelated > 3 * noise_sigma, (
        "unrelated texts score at noise level — documents and query are probably "
        "in different vector spaces (different models)"
    )

    related = text_similarity(
        "How do I process a refund?",
        "To initiate a return payout, execute a credit balance reversal.",
    )
    print(f"4. cosine of related texts            = {related:.6f}   (must beat #3)")
    assert related > unrelated, "related pair scores below unrelated pair — index is broken"

    print("self-check passed\n")


def describe(scored: list[tuple[float, str]], label: str) -> None:
    """Print the score distribution. Thresholds get chosen from this, not guessed."""
    values = [s for s, _ in scored]
    print(
        f"  [{label}] n={len(values)}  max={max(values):.4f}  "
        f"median={statistics.median(values):.4f}  min={min(values):.4f}"
    )


def show(query: str, selected: list[tuple[float, str]]) -> None:
    print(f"Query: {query}\n")
    print("Selected chunks (context sent to the LLM):")
    if not selected:
        print("  <empty — nothing cleared the threshold>")
    for score, text in selected:
        print(f"  [{score:+.4f}] {text!r}")
    print(f"\nAnswer:\n{generate_answer(query, selected)}\n")


# ---------------------------------------------------------------------
# EXPERIMENTS
# ---------------------------------------------------------------------

CORPUS = (
    "The policy states that every refund must be logged in the refund database "
    "under the refund tab. Refund entries older than ninety days are archived "
    "automatically and can no longer be edited in place. "
    "To initiate a return payout, open the ledger, select the customer account, "
    "and execute a credit balance reversal. The reversal posts to the customer "
    "ledger on the following business day. "
    "Shipping to international destinations outside the domestic zone requires "
    "custom duty forms, filed before the parcel leaves the origin warehouse. "
    "Warehouse staff scan every outbound parcel at the loading bay, and the scan "
    "event is what marks the order as dispatched in the order service. "
    "Order cancellation before dispatch releases the reserved stock immediately; "
    "after dispatch it becomes a return, not a cancellation. "
    "Payment capture happens at dispatch, not at checkout, so an authorised order "
    "that never dispatches expires without ever being charged."
)


def experiment_1_lexical_overlap_is_not_relevance() -> None:
    """The winning chunk need not contain the query's keywords."""
    print("=== 1. Lexical overlap is not relevance ===")
    index = build_index_from_text(CORPUS)
    print(f"  corpus: {len(CORPUS)} chars → {len(index)} chunks")

    query = "How do I process a refund for a returned item?"
    scored = score_all(query, index)
    describe(scored, "all chunks")
    show(query, select_top_k(scored, k=3))


def experiment_2_semantic_proximity_is_not_propositional_relevance() -> None:
    """Negation barely moves the vector. Measured directly, as a pair."""
    print("=== 2. Semantic proximity is not propositional relevance ===")

    affirmative = "The customer requested a refund."
    negated = "The customer did not request a refund."
    delta = text_similarity(affirmative, negated)
    print(f"  cosine(affirmative, negated) = {delta:.4f}")
    print("  ^ if this is high, the embedding does not encode the proposition,")
    print("    only the topic. That is why cross-encoder rerankers exist.\n")

    # Same two sentences as separate documents — never chunked together.
    index = build_index_from_docs([affirmative, negated])
    query = "Did the customer ask for their money back?"
    scored = score_all(query, index)
    for score, text in scored:
        print(f"  [{score:+.4f}] {text!r}")
    print("  ^ the gap between these two is what retrieval has to work with.\n")


def experiment_3_there_is_no_null_result(threshold: float) -> None:
    """Top-k always returns k. A threshold is the only way to decline."""
    print("=== 3. There is no null result ===")
    index = build_index_from_text(CORPUS)

    query = "What is the capital of France?"
    scored = score_all(query, index)
    describe(scored, "off-domain query")

    print("\n-- top-k selection (no gate) --")
    show(query, select_top_k(scored, k=3))

    print(f"-- threshold selection (>= {threshold}) --")
    show(query, select_above_threshold(scored, threshold))


def calibrate_threshold(in_domain: list[str], off_domain: list[str]) -> None:
    """Pick the gate from data instead of guessing it.

    Any value between the off-domain max and the in-domain min separates the
    two sets. A negative margin means no single global threshold works, which
    is itself a finding worth reporting.
    """
    print("=== Threshold calibration ===")
    index = build_index_from_text(CORPUS)

    in_tops = [score_all(q, index)[0][0] for q in in_domain]
    off_tops = [score_all(q, index)[0][0] for q in off_domain]

    print(f"  in-domain  top scores: {[f'{s:.4f}' for s in in_tops]}")
    print(f"  off-domain top scores: {[f'{s:.4f}' for s in off_tops]}")

    floor, ceiling = min(in_tops), max(off_tops)
    margin = floor - ceiling
    print(f"  usable band: ({ceiling:.4f}, {floor:.4f})   margin={margin:+.4f}")
    if margin <= 0:
        print("  no global threshold separates the two sets — report this, do not hide it")
    else:
        print(f"  suggested threshold: {(floor + ceiling) / 2:.4f}")
    print()


if __name__ == "__main__":

    self_check()

    experiment_1_lexical_overlap_is_not_relevance()
    experiment_2_semantic_proximity_is_not_propositional_relevance()

    calibrate_threshold(
        in_domain=[
            "How do I process a refund for a returned item?",
            "When is payment actually captured?",
            "What happens if I cancel after dispatch?",
        ],
        off_domain=[
            "What is the capital of France?",
            "How do I bake sourdough bread?",
        ],
    )

    # Replace with the value calibrate_threshold() suggests on your run.
    experiment_3_there_is_no_null_result(threshold=0.45)