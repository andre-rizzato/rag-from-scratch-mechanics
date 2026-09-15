# RAG from scratch — the mechanics a framework hides

A retrieval-augmented generation pipeline implemented without LangChain, LlamaIndex,
or a vector database, so that every decision a framework makes by default has to be
made explicitly — and can be measured.

This is a learning exercise on synthetic data, not a production component. Its value
is the measurements, not the code.

Companion article: **Retrieval Is Not a Lookup** *(https://www.linkedin.com/pulse/retrieval-lookup-andre-rizzato-wp9kf/)*

## The invariant

> Documents and queries **must** be embedded with the same model.

Different models are different vector spaces. Cosine similarity between two of them
is not "low similarity" — it is noise centred on zero. For two random unit vectors in
`d` dimensions, cosine has mean 0 and standard deviation `1/√d`, so on a 1024-dimension
model anything inside roughly ±0.03 is indistinguishable from an unrelated pair.

`self_check()` enforces this in four assertions and would have caught the bug that
started this project in about four seconds.

## What is implemented here vs. called out to

**Implemented:** chunking with overlap and boundary guards, cosine similarity with a
dimension check, top-k selection, threshold gating with abstention, prompt assembly.

**External HTTP calls:** Voyage AI (`voyage-3.5`, 1024 dimensions) for embeddings,
Anthropic for generation. Nothing else.

## Pipeline

```
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
```

Scoring and selection are separate steps on purpose. `score_all()` ranks everything;
`select_top_k()` and `select_above_threshold()` are the policies applied on top. That
split is where the interesting decisions live — one of them can never return empty,
the other can.

## Running it

```bash
pip install voyageai anthropic python-dotenv
```

Create a `.env` file in the repository root:

```
VOYAGE_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

Then:

```bash
python rag_from_scratch_mechanics.py
```

This runs the self-check, the three experiments, and the threshold calibration.

## What the experiments show

**Self-check** — four assertions that separate a broken retriever from a weak one:
cosine of a vector with itself, determinism of the embedding API, an unrelated pair
against the noise floor, and a related pair that must beat it.

**1. Lexical overlap is not relevance.** Query: *"How do I process a refund for a
returned item?"* against a 10-chunk support document. The winning chunk repeats
"refund" three times and describes the logging policy; the chunk that actually answers
the question describes the procedure and never uses the word. Scores across all chunks:
max 0.5557, median 0.4941, min 0.3607 — the whole top-3 fits inside 0.05, which is
worth reporting honestly rather than presenting as a clean result.

**2. Semantic proximity is not propositional relevance.**

```
cosine("The customer requested a refund.",
       "The customer did not request a refund.")  =  0.9492
```

Two sentences with opposite truth values, 0.95 similar. In retrieval, against
*"Did the customer ask for their money back?"*, the negated sentence ranks **above**
the affirmative one (0.6011 vs 0.5811). No threshold fixes this and no re-ranking of
the dense scores fixes it — the embedding encodes the topic, not the proposition.
This is what cross-encoder rerankers and hybrid retrieval exist for.

**3. There is no null result.** An off-domain query — *"What is the capital of
France?"* — against a logistics document returns three chunks at 0.3596, 0.3204 and
0.2989, with no signal that retrieval failed. BM25 would have returned an empty set.
Dense retrieval has no such state, so the threshold is the only component that can
decline — and it does so without paying for the model call.

**Threshold calibration** derives the gate from data instead of guessing it:

```
in-domain  top scores:  0.5557  0.4940  0.4871
off-domain top scores:  0.3594  0.3388
usable band: (0.3594, 0.4871)   margin = +0.128
```

The margin is the number worth reporting: it says how separable this domain is on this
model. A margin at or below zero would mean no single global threshold works.

## Deliberately not implemented

No cross-encoder reranker, no hybrid (BM25 + dense) retrieval, no ANN index. Those are
the layers that address the failure in experiment 2, and naming the gap is more useful
than pretending it isn't there.

## License

MIT — see [LICENSE](LICENSE).
