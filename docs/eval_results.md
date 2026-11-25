# Retrieval evaluation results

Measured on 2026-10-24 with

```bash
python -m evaluation.run_eval --embedder hashing --show-misses
```

over 34 questions about the 6 fictional documents in `sample_docs/` (`evaluation/questions.jsonl`).

## What is measured

For each question the retriever returns the top-k chunks. The question counts as *found at rank r* when
the r-th chunk comes from the right file **and** contains the phrase that answers it. Retrieval is measured
on its own, without a language model, so the numbers are cheap and repeatable.

- **hit@k** - share of questions found within the first k chunks (higher is better)
- **MRR** - mean reciprocal rank, 1/rank averaged over the questions (1.0 = always found at rank 1)

## Results: offline baseline (hashing embedder)

| embedder | chunk size (words) | overlap | chunks | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|---|---|
| hashing-4096 | 60 | 9 | 32 | 0.79 | 0.91 | 0.94 | 0.86 |
| hashing-4096 | 120 | 18 | 18 | 0.88 | 0.97 | 1.00 | 0.93 |
| hashing-4096 | 180 | 27 | 13 | 0.91 | 0.97 | 0.97 | 0.94 |

How to read this:

- Moving from 60-word to 120-word chunks raised hit@3 from 0.91 to 0.97 (a likely reason: small chunks split a fact from its context).
- 180-word chunks gave the best hit@1 and MRR, but missed one question that 120-word chunks still found
  within the top five (hit@5 0.97 against 1.00).
- With 34 questions, one question moves a score by about 0.03. Differences smaller than that are noise,
  so this set cannot name a single "best" chunk size; it shows that very small chunks are worse.
- The hashing embedder only matches shared words, not meaning. It is the *baseline* the service falls back to when
  `sentence-transformers` is not installed, not what a real deployment should run.

## Questions not found at rank 1

```text
Not found at rank 1 (chunk size 60):
  - rank 2: How many restart attempts are allowed after a motor overcurrent trip?
  - rank >5: What happens when communication with the remote terminal unit is lost?
  - rank 2: Who is allowed to restart a pump after a critical alarm?
  - rank 2: How often does lockout training have to be renewed?
  - rank 4: How do you confirm that equipment is properly isolated?
  - rank >5: When must a drifting sensor be recalibrated?
  - rank 2: Which metrics does the monthly asset health report contain?

Not found at rank 1 (chunk size 120):
  - rank 2: Who is allowed to restart a pump after a critical alarm?
  - rank 2: At what bearing temperature does the pump alarm become critical?
  - rank 3: How do you confirm that equipment is properly isolated?
  - rank 5: When must a drifting sensor be recalibrated?

Not found at rank 1 (chunk size 180):
  - rank 2: Who is allowed to restart a pump after a critical alarm?
  - rank 2: How do you confirm that equipment is properly isolated?
  - rank >5: When must a drifting sensor be recalibrated?
```

Most of these are questions worded differently from the text (for example "restart" against "restarted").
A semantic embedding model is expected to handle that better, but that has not been measured yet.

## Real file formats (DOCX and PDF)

The baseline above reads plain `.txt` files, but people upload Word and PDF files, which bring headers, footers, page
numbers, tables and page breaks. To check that parsing and cleaning do not hurt retrieval, the six sample documents were
rebuilt as real files and the same 34 questions were asked again on 2026-11-12, with the same offline embedder and the
same command, pointing `--corpus` and `--questions` at the new files:

- **DOCX**: six Word files with headings, bullet lists, a table, a header, a footer and a page-number field.
- **PDF**: the same six files exported to PDF with LibreOffice, so they have real pages, a running header and footer,
  and page numbers.
- **Handbook**: the six documents joined into one 9-page PDF with the running header and footer on every page, so the
  header and footer removal has to do its job.

DOCX:

| embedder | chunk size (words) | overlap | chunks | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|---|---|
| hashing-4096 | 60 | 9 | 34 | 0.85 | 0.91 | 0.94 | 0.88 |
| hashing-4096 | 120 | 18 | 20 | 0.88 | 0.97 | 0.97 | 0.92 |
| hashing-4096 | 180 | 27 | 13 | 0.94 | 0.97 | 0.97 | 0.96 |

PDF:

| embedder | chunk size (words) | overlap | chunks | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|---|---|
| hashing-4096 | 60 | 9 | 36 | 0.85 | 0.91 | 0.94 | 0.88 |
| hashing-4096 | 120 | 18 | 20 | 0.85 | 0.94 | 0.97 | 0.90 |
| hashing-4096 | 180 | 27 | 14 | 0.91 | 0.97 | 0.97 | 0.94 |

Handbook PDF:

| embedder | chunk size (words) | overlap | chunks | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|---|---|
| hashing-4096 | 60 | 9 | 34 | 0.79 | 0.91 | 0.94 | 0.86 |
| hashing-4096 | 120 | 18 | 20 | 0.88 | 0.94 | 1.00 | 0.92 |
| hashing-4096 | 180 | 27 | 15 | 0.91 | 0.97 | 0.97 | 0.94 |

Compared with the `.txt` table above:

- hit@3 and hit@5 differ from the `.txt` results by at most one question (0.03) at every chunk size, and MRR by at
  most 0.03.
- hit@1 moves by up to two questions (0.06), mostly upward at 60 words. On a 34-question set that is noise, not a sign
  that PDFs retrieve better.
- So reading real files does not noticeably damage retrieval. It does not show that retrieval is good in general: the
  questions still share the documents' vocabulary, and this is still the offline embedder.

A separate run against a real server uploaded the same files over HTTP. The page numbers in the returned sources
matched the pages found independently with poppler's `pdftotext` for 31 of 31 checked questions.

## Run it with the real embedding model
Run on 22 November 2026 with `python -m evaluation.run_eval --embedder sentence-transformers --show-misses`.
Computer: Intel Core i7-1355U (13th gen), 16 GB RAM, Windows 11, Python 3.13.5. 34 questions over the 6 sample documents.

| embedder | chunk size (words) | overlap | chunks | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | 60 | 9 | 32 | 0.79 | 0.94 | 0.94 | 0.86 |
| sentence-transformers/all-MiniLM-L6-v2 | 120 | 18 | 18 | 0.79 | 0.94 | 1.00 | 0.88 |
| sentence-transformers/all-MiniLM-L6-v2 | 180 | 27 | 13 | 0.85 | 0.97 | 1.00 | 0.92 |

One question moves a score by 0.03, so differences smaller than that are noise. Larger chunks did better here: at 60 words two questions fell outside the top 5, while at 120 and 180 words the right chunk was always in the top 5. At 180 words the right chunk was first for 29 of 34 questions and in the top 3 for 33 of 34. The offline baseline scored hit@3 0.97 and MRR 0.94 at 180 words, so on this small, keyword-friendly question set the real model matches it within noise rather than beating it.
These numbers were produced without the real model (it could not be downloaded where this project was built).
On your own machine:

```bash
pip install -r requirements-ml.txt
python -m evaluation.run_eval --embedder sentence-transformers --show-misses
```

Compare the table with the baseline above and keep the new numbers here. Also try
`--chunk-sizes 80 120 150` and `--top-k 1 3 5` to tune `DOCQA_CHUNK_SIZE` and `DOCQA_TOP_K`.
The embedding model reads at most 256 word-pieces, so chunks much longer than about 180 words are partly ignored.

## Evaluating on your own documents

Put your files in a folder and write a JSON Lines file with one object per line:

```json
{"question": "How often is the oil sampled?", "evidence": "sampled every 12 months", "source": "manual.txt"}
```

`evidence` is a short phrase that must appear in the chunk that answers the question; `source` is the file it comes from.
Then run:

```bash
python -m evaluation.run_eval --corpus path/to/folder --questions path/to/questions.jsonl
```
