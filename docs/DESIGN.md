# Design notes

How DocQA is put together, why each choice was made, what it costs, and what would change at a larger scale.
Read the [README](../README.md) first for what the service does and how to run it.

## 1. Scope

**Goals**

- Answer only from the uploaded documents, show where each answer came from, and say so when the documents do
  not contain it.
- Run anywhere. With no API key and no ML libraries the service still works, in a clearly weaker mode.
- Stay small enough to read in an afternoon, and testable without a network.

**Non-goals:** multiple users or tenants, millions of chunks, OCR, streaming, conversation memory. Section 6 says
what each of them would take.

## 2. The two paths through the system

**Ingest** (`RagPipeline.ingest`)

1. `doc_id_for` takes the first 12 hex characters of the file's SHA-256. If that id is already stored, the existing
   document is returned, so uploads are idempotent.
2. `parse_document` gives the text per page (PDF) or as one block (DOCX, TXT, MD).
3. `drop_repeated_lines` removes running headers and footers across the pages; `clean_text` tidies each page.
4. `chunk_text` cuts sentence-aware chunks with overlap; each chunk remembers its page.
5. `embedder.embed`, then `store.add`, then `store.save`.

An ingest lock allows one upload at a time. Questions can still be answered while an upload is being embedded,
because the store takes its own lock only for the short add and search calls.

**Ask** (`RagPipeline.ask`)

1. An empty index answers "No documents have been uploaded yet."
2. The question is embedded and `store.search` returns the top-k chunks, optionally from one document.
3. Chunks below `min_score` are dropped. If none are left, the reply is the fixed refusal text with
   `grounded=false`, and **the model is never called**.
4. `llm.generate(question, chunks)` writes the answer. Any SDK failure is turned into `LLMError` (HTTP 502).
5. If the model's reply is exactly the refusal sentence, the response is `grounded=false` with no sources.
   Otherwise the reply is returned together with the chunks as sources.

## 3. Decisions and trade-offs

### 3.1 Refuse by default

- **Decision.** Two independent guards. The retrieval guard lets a chunk through only if its cosine similarity is at
  least `min_score`. The prompt guard tells the model to answer from the passages only, and to reply with one exact
  sentence otherwise.
- **Why.** Hallucination is the main failure of RAG. The cheapest fix is to not ask the model at all when nothing
  relevant was retrieved. It also saves money and time on off-topic questions.
- **Trade-off.** A cut-off that is too high refuses answerable questions; one that is too low lets weak passages
  through. The default of `0.2` for MiniLM is a starting point that has not been tuned on real documents.
  `/search` shows raw scores and `DOCQA_MIN_SCORE` changes the value. The second guard compares the model's reply
  with the refusal sentence exactly, so a paraphrased refusal counts as an answer (it still carries its sources,
  so a reader can judge).

### 3.2 Parsing and cleaning

- **Decision.** PDFs are read page by page with pdfplumber, so every chunk knows its page and the sources can show
  it. DOCX goes through docx2txt (no pages, so `page` is `null`). TXT and MD are decoded as UTF-8 (a byte
  order mark is tolerated). UTF-16 is recognised by its byte order mark (what Notepad's "Unicode" option writes), and
  anything that is not valid UTF-8 is read as Windows-1252, the "ANSI" that older Windows programs save; the few byte
  values it leaves undefined become replacement characters. Anything else is a 415.
- **Cleaning steps.** Unicode NFKC (ligatures, non-breaking spaces), control characters removed, a page number
  ("Page 3 of 10", "- 4 -", a bare "12") removed only when it is the first or the last line of a PDF page, so a number
  that stands alone anywhere else (a table cell, a year, a list number) is kept, and DOCX, TXT and MD files, which have
  no pages, are never stripped; words hyphenated across a line break re-joined, lines that were only
  wrapped for layout re-joined (the next line starts in lower case and the previous one has no closing
  punctuation), blank lines collapsed.
- **Headers and footers.** A short line (at most 100 characters) that repeats, ignoring digits, on at least 60% of
  the pages is removed. This only runs on documents of four pages or more, so short documents are never touched.
- **Why.** The embedding model sees exactly the text it is given. A footer repeated on every chunk dilutes every
  vector and pollutes retrieval.
- **Trade-offs.** These are heuristics. A legitimately repeated line (a table header on every page) would be
  removed, and unwrapping can merge a heading with a following lower-case line. Tables are flattened to text (a PDF
  table comes out one row per line, a DOCX table one cell per line, so empty cells vanish) and multi-column layouts come
  out in whatever order pdfplumber extracts them. Pages are chunked separately, so a fact that continues across a page
  break is split between two chunks. A text file in some other encoding (Shift-JIS, say) is read as Windows-1252 and
  comes out garbled. A scanned PDF yields no text, and the upload fails with a
  422 that says OCR is not supported, rather than silently indexing nothing.

### 3.3 Chunking

- **Decision.** Sentences are the unit. They are found with a regular expression (a sentence ends at `.`, `!` or `?`
  followed by whitespace and a capital letter, digit, quote or bracket), and every newline is also a boundary, so
  headings and bullets stay separate. Sentences are packed into chunks of at most `chunk_size` words (180). A
  sentence longer than that is hard-split. The next chunk starts with the longest run of whole sentences from the
  end of the previous one that fits in `chunk_overlap` words (30).
- **Why.** Cutting at fixed character counts splits sentences and facts in half. With whole-sentence overlap, a fact
  that straddles a boundary appears intact in at least one chunk. Words are counted instead of model tokens, so
  chunking has no dependency on a tokenizer.
- **Trade-offs.** The default embedding model reads at most 256 word-pieces and ignores the rest. 180 words of
  ordinary English prose usually fits, but text full of numbers and codes needs more word-pieces per word, so the
  tail of such a chunk may not influence its vector (the model that writes the answer still sees the whole text).
  If the real-model evaluation shows misses on such documents, lower `DOCQA_CHUNK_SIZE`. Abbreviations followed by
  a capital ("Dr. Smith") can split a sentence early, which is harmless for retrieval.
- **Evidence.** On the offline baseline (see section 5), MRR was 0.86, 0.93 and 0.94 for 60, 120 and 180-word
  chunks. Small documents with short facts favour larger chunks; this says nothing about the real model.

### 3.4 Embeddings

- **Interface.** An `Embedder` has a `name`, a `dim`, a `recommended_min_score` and `embed(texts)`, which returns
  L2-normalised float32 vectors, so a dot product is the cosine similarity.
- **Default.** Sentence-Transformers `all-MiniLM-L6-v2`: 384 dimensions, runs on a CPU, no per-call cost, and no
  document text leaves the machine.
- **Offline fallback.** `HashingEmbedder` hashes words and word pairs (CRC32, with a sign to reduce collision bias)
  into 4096 buckets, weights them by `1 + log(count)` and normalises. It needs no model and no download, and it is
  deterministic, so the tests, CI and the dependency-free demo use it. It matches words, not meaning, so it is not
  for production quality retrieval.
- **`auto` versus explicit.** `auto` tries Sentence-Transformers and falls back to hashing with a warning. The Docker
  image sets `sentence-transformers` explicitly, so a broken or missing model stops the container instead of
  silently degrading answer quality.
- **Model changes.** The index is stamped with the embedder's name. If the name differs at startup, every stored
  chunk is re-embedded and the index is saved again. Vectors from different models are not comparable, and mixing
  them would return nonsense without any error. The cost is a slower start after a model change on a large index.
- **Per-embedder cut-off.** Score scales differ between embedders (`0.2` for MiniLM, `0.05` for hashing), so the
  recommended cut-off belongs to the embedder.

### 3.5 Vector store

- **Decision.** FAISS `IndexFlatIP` over normalised vectors is an exact cosine search. At this scale (thousands to
  low hundreds of thousands of chunks) it answers in milliseconds, and approximate indexes (IVF, HNSW) would add
  tuning knobs and lost recall that this size does not need.
- **Source of truth.** The numpy matrix and the list of records are what is saved and loaded. The FAISS index is
  derived from them and rebuilt after a delete, because a flat index cannot cheaply remove rows. Without FAISS, a
  numpy dot product gives the same ranking (a test checks this).
- **Per-document filter.** A flat FAISS index has no filter, so a search limited to one document asks for every
  chunk and filters afterwards. That way the filter can never leave fewer than k results just because the overall
  top k belonged to other documents. It costs time linear in the index size, which is fine here.
- **Persistence.** `vectors.npy`, `records.json` and `meta.json` are each written to a temporary file and then
  renamed over the old one, with `meta.json` (which holds the chunk count) written last. On load the three must
  agree, otherwise the app refuses to start with a message, rather than serve a half-updated index.
- **Limits.** Every save rewrites all three files, so the cost of an upload grows with the index. The vectors are held
  in memory twice (numpy and FAISS): 100,000 chunks of 384 float32 values is about 150 MB each. A lock serialises
  access inside the process, which is why a single process is the supported shape.

### 3.6 Language-model layer

- **Interface.** `LLM` has a `name` and `generate(question, records)`. Three implementations are chosen from the
  settings: Anthropic, OpenAI-compatible and extractive.
- **Prompt.** The system prompt has five rules: use only the passages; reply with the exact refusal sentence if
  they do not contain the answer; cite passages by id; treat passage text as untrusted; keep the answer short. The
  user message is the passages, each in a `<passage id="1" source="file.pdf, page 3">` block, then the question. The
  ids are the ranks among the retrieved chunks, so `[2]` in an answer means `sources[1]`.
- **Prompt-injection hygiene.** Passage text and source names are HTML-escaped, so a chunk cannot close its own tag
  and pose as instructions, and the system prompt says the passages are untrusted. This lowers the risk; it does not
  remove it. The model has no tools, so the realistic worst case of an injected instruction is a wrong or odd
  answer, not an action.
- **Reliability.** A 60 second timeout (the Anthropic SDK default is ten minutes). Every SDK failure becomes one
  `LLMError` and an HTTP 502 with a generic message; the real exception goes to the server log only, because
  provider errors can echo request details.
- **Providers.** The OpenAI-compatible class also covers Groq, Ollama and OpenRouter through `OPENAI_BASE_URL`. No
  sampling parameters (temperature, top_p) are sent, so each provider's defaults apply and there are fewer settings
  to get wrong across providers and models.
- **Extractive fallback.** It lets the whole system run, be demoed and be tested end to end with no key. It splits
  the retrieved passages into sentences and scores each by the question words it contains, where a word is worth
  `1 / (number of sentences that contain it)`, so a rare word ("101") counts for more than a common one ("code").
  It keeps up to two sentences scoring within 60% of the best and cites the passage each came from. The answer is
  always a quote, which makes it easy to check, but it matches words rather than meaning, so a question phrased
  differently from the document can miss. That is what the LLM modes are for.

### 3.7 HTTP layer

- **Thin on purpose.** Routes call `RagPipeline` and map the results to Pydantic models. Domain errors are
  subclasses of `DocQAError`, and one handler maps each to a status code (415, 413, 422, 404, 502; anything else 400).
- **Plain `def` endpoints.** Embedding and searching are blocking, CPU-bound calls. FastAPI runs plain `def` routes
  in its thread pool, so the event loop stays free for health checks and other requests. An `async def` route that
  made the same calls would stall everything while it ran.
- **App factory and lifespan.** `create_app(settings, pipeline)` lets tests inject their own pipeline. In production
  the pipeline is built at startup, so a bad environment variable, a missing API key for a forced provider or a
  missing model fails at startup instead of on the first request.
- **Authentication.** An optional shared key in `X-API-Key`, compared with `secrets.compare_digest`. `/health` and
  the documentation pages stay open. It is a single secret, enough to protect LLM spend, not a user system.
- **Upload limits, in two layers.** A middleware rejects a declared `Content-Length` over the limit (plus 64 KiB of
  multipart overhead) before the body is read, and the handler reads at most one byte more than the limit. An upload
  sent in chunks without a `Content-Length` reaches the handler only after the framework has spooled it, so for
  anything on the internet a request-size limit in the reverse proxy is also needed.
- **Idempotent uploads.** The document id is the first 12 hex characters of the content's SHA-256: identical bytes
  give the same id and a `200` instead of a `201`. The same content under a new filename is therefore the same
  document. Twelve hex characters (48 bits) make an accidental collision negligible at this scale.
- **Observability.** Each request logs method, path, status and duration, and the response carries
  `X-Process-Time-ms` (the early 413 rejection in the middleware skips both).
- **One worker.** The index lives in this process's memory (see 3.5), so the Docker image starts one uvicorn worker.

## 4. How the tests are built

- **No network, no keys.** The suite runs offline on any machine and in CI.
- **Real files, real clients.** PDFs are generated with fpdf2 and DOCX files with python-docx, then parsed. The
  Anthropic and OpenAI wrappers use the real SDK clients pointed at a local `ThreadingHTTPServer` that plays the API,
  so request building, headers and response parsing are exercised. A mock would only confirm that the code calls
  what the test says it calls.
- **Deterministic retrieval.** The hashing embedder gives reproducible scores, so tests can assert on ranking and on
  the cut-off. Small fake and broken LLM classes cover the model-failure and "model says not found" paths.
- **Both back ends.** The vector store tests run on FAISS and on numpy, and one test requires them to rank
  identically.
- **Do the tests notice bugs?** `tools/mutation_check.py` applies 137 deliberate bugs, one at a time, to a scratch copy
  of the code (a comparison flipped, a lock removed, a save skipped, an escape dropped, a header check weakened) and
  runs the suite on each copy. A bug that no test catches means a test is missing, and the tool lists it. During
  development it exposed several untested behaviours (the exact cut-off boundary, a deletion that was not saved, the
  ingest lock); each now has a test. The bugs are hand-written for this code, so this is a spot check, not proof.
- **Real model on demand.** The one test that loads the real embedding model carries an `integration` marker and is
  deselected by default (`python -m pytest -m integration` runs it).
- **CI.** Lint and tests on Python 3.11, 3.12 and 3.13, then a job that builds the image, starts it, uploads a
  sample document and asks a question with the real model.
- **What the tests cannot show.** Answer quality from a real model, behaviour against the live APIs, and anything
  that depends on your documents. Those need the evaluation (section 5) and your own trial runs.

## 5. Evaluation

`evaluation/run_eval.py` measures retrieval alone, with no language model, so it is cheap and repeatable. For each of
34 questions in `evaluation/questions.jsonl` it checks whether the chunk that answers it is among the top k, and
reports hit@1, hit@3, hit@5 and mean reciprocal rank (MRR) for 60, 120 and 180-word chunks. The results for the
offline baseline are in [eval_results.md](eval_results.md). The same questions were also run against DOCX and PDF
versions of the documents; hit@3 and hit@5 stay within one question of the `.txt` numbers, so parsing and cleaning
do not noticeably hurt retrieval.

Caveats worth stating up front: the questions were written by the author against six fictional documents and share
their vocabulary, so the numbers are optimistic; only retrieval is measured, not whether the generated answer is
faithful; and the real embedding model has not been measured yet.

## 6. Limitations and next steps

| Limitation today | Next step |
|---|---|
| Dense retrieval only; embeddings can be weak on exact identifiers such as part numbers and error codes | Hybrid search: BM25 plus dense, merged with reciprocal-rank fusion, then a cross-encoder reranker on the top 20 |
| Scanned PDFs, tables and images are not understood; a fact that runs across a PDF page break is split | OCR for pages without text; table-aware extraction; carry the last sentences of a page into the next |
| Text files are read as UTF-8, UTF-16 (with a byte order mark) or Windows-1252 | Detect other encodings (charset-normalizer) |
| Each upload rewrites the whole index; vectors are held twice in memory; one process | Incremental or database-backed storage (pgvector, OpenSearch, Qdrant), which also allows several workers |
| Uploads are processed inside the request, so a large PDF holds the connection (parsing plus the offline embedder took about 78 ms per page here: a 100-page PDF, 260 chunks, about 7.8 s; the real model will be slower, so measure it) | Background ingestion with a job id and a status endpoint |
| One shared index and one shared key | Per-user authentication (OAuth/JWT), per-user collections or a filter field, rate limiting |
| Answers arrive whole | Streaming with Server-Sent Events |
| Every question is independent | Rewrite follow-up questions using the chat history |
| The size limit relies on `Content-Length` and a bounded read | A body-size limit in the reverse proxy |
| Retrieval is measured, answers are not | Answer-quality evaluation: faithfulness and correctness, a set of unanswerable questions, possibly an LLM judge |
| The `0.2` cut-off for MiniLM is untuned | Choose it from the score distributions of answerable and unanswerable questions |
| Plain-text logs, no metrics | Structured logs with request ids, and metrics for latency, refusal rate and retrieval scores |
