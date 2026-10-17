# Retrieval eval set

`questions.jsonl` — 50 questions over `data/Calculus-Lecture.pdf`
(S. Kim, *Calculus I*, 228 PDF pages, ~165k characters). It is **frozen**: it was
committed before any retrieval experiment was run and must not be edited to
change results. (If it is ever revised, make it a new versioned file.)

## Composition

| Kind | Count |
|---|---:|
| Answerable from one specific passage | 44 |
| Not answerable from the corpus (abstention probes) | 6 |

Topics follow the chapters: functions, limits/continuity, derivatives,
applications of derivatives (up to antiderivatives). Question wording is
paraphrased rather than copied from the text where that was natural, but several
questions still share vocabulary with their passage (e.g. "Rolle's Theorem"),
so lexical methods are not handicapped.

## How it was built

1. The corpus text is the output of the app's own `ingest.extract_text`.
2. An AI coding agent read the relevant passages and wrote each question, a gold
   answer, and an `evidence` snippet copied from the PDF text.
3. `build_questions.py` is the single source of truth: it verifies each evidence
   snippet occurs in the extracted text (whitespace-insensitive) and records
   every matching character span (`spans`) and PDF page (`pdf_pages`, 1-based
   PDF index, not the printed page number). One snippet (q09) occurs in two
   places (a definition and its restatement in 3.9); both spans count.
4. For unanswerable questions it verifies that characteristic terms
   (`riemann`, `taylor`, `partial deriv`, `fundamental theorem`, `series`,
   `laplace`, ...) do not occur anywhere in the corpus.

Regenerate with `python eval/build_questions.py` (output must be byte-identical).

## Relevance rule

A retrieved chunk is **relevant** if its character range overlaps the gold span
by at least 50% of the span's length. Spans are <= 140 chars, below the 300-char
chunk overlap, so at least one chunk of the default chunking always contains a
span entirely. Judging by character offsets (not by chunk id) makes the labels
independent of the chunk size, so chunking can be varied without relabelling.

## Known limitations / needs owner review

- Gold labels were written by an AI agent and checked mechanically (the evidence
  text exists) but **not independently reviewed by a human**. Please skim
  `gold_answer`/`evidence` pairs, especially for q27 (strategy list: the span is
  one step of several), q37 (span covers the 0/0 case only), q04 (area formula).
- Only the *primary* passage is labelled. The same fact is sometimes restated
  elsewhere (chapter summaries, exercises, the book index), so a retrieved chunk
  can be useful without being counted. This understates every method.
- Abstention questions do not score retrieval recall; they are reported
  separately (top-1 score distribution only, no generation was run).
- 44 answerable questions is small: a difference of one question is 0.023 recall.
