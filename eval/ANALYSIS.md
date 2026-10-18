# Error analysis

Source: `results/retrieval_20261017.json` (baseline) and
`results/ligature_fix_20261017.json` (after the ligature fix). Ranks below are the
rank of the first relevant chunk (1500/300 chunking); "miss" = not in the top 10.
Chunk ranges are character offsets in the extracted text. All findings come from
looking at the saved top-10 lists; causes are my reading of those chunks, not
proven by ablation unless stated.

## Worst questions (baseline; ranks for dense / BM25 / hybrid / dense+rerank / hybrid+rerank)

| Q | Ranks | Finding |
|---|---|---|
| q12 "squeezed between two functions…" | 6 / miss / miss / 4 / 3 | Vocabulary gap: the passage says "Sandwich Theorem", the question says "squeezed". BM25 finds nothing relevant. Dense returns neighbouring limit sections (sin θ/θ, which *uses* the squeeze) — topically close, not the statement. |
| q10 "average change … across an interval" | 9 / 6 / 4 / 6 / 8 | The definition is a few lines inside a chunk dominated by worked examples (Example 2.1/2.2). Dense preferred chunks from section 4.2 that restate "average rate of change" inside the Mean Value Theorem remarks. Fixed by the ligature change for dense (rank 1 after) — most likely because "deﬁnition"-heavy text shifted the chunk embedding, not because of any lexical effect; not verified. |
| q24 "term for the derivative of the derivative" | 5 / 10 / 4 / 7 / 5 | The term "second derivative" is not in the question; dense returned the section 3.2 chunks about the derivative as a function. The answer sits mid-chunk among velocity/acceleration examples. |
| q25 "derivative of a composition" | 1 / miss / 3 / 6 / 6 | Dense was right at rank 1 but the **reranker demoted it to rank 6**, promoting a table-of-contents chunk (range 4800–6300, a list of section titles and page numbers) to rank 1. I did not test why; a plausible reading is that section titles look topically relevant to the cross-encoder. |
| q38 "point of inflection" | 2 / miss / 6 / 1 / 1 | **BM25 miss is an extraction artifact**: the PDF text contains the ligature glyph in "inﬂection", so the token "inflection" in the query never matches. See below. |
| q29 related-rates strategy | 2 / 2 / 2 / 2 / 2 | Rank 2 for every method. Likely a labelling limitation: the gold span is only step 3 of a 4-step list, and an adjacent chunk containing the earlier steps is also plausible. Needs owner review. |

## Systematic finding: ligatures break lexical matching

`extract_text` emits `ﬁ` (310×) and `ﬂ` (39×). The word "definition" occurs
**0 times** in plain form and 87 times as "deﬁnition". Any query containing
"definition", "inflection", "differentiable"-style words with fi/fl cannot match
by keyword. Dense retrieval is less affected but still sees odd tokens.

Fix (own commit): expand ligatures after extraction (`ingest.normalize_ligatures`,
enabled in `index.py`). The eval maps the frozen gold spans onto the normalised
text (`--fix-ligatures`), so the question set is unchanged.

| Method | MRR@10 before | MRR@10 after | recall@1 before→after | recall@5 before→after |
|---|---:|---:|---:|---:|
| BM25 | 0.698 | 0.747 | 0.591 → 0.659 | 0.795 → 0.841 |
| Hybrid | 0.857 | 0.898 | 0.795 → 0.841 | 0.955 → 0.977 |
| Dense | 0.825 | 0.831 | 0.705 → 0.727 | 0.955 → 0.955 |

Caveats: n=44, so each question is ±0.023 recall. Per-question ranks moved both
ways (dense recall@3 fell 0.932 → 0.886: q15, q33 got worse; q10, q19, q26, q32 got
better), so the dense change is within noise; the BM25/hybrid improvement is the
one with a mechanistic explanation (q38 went from miss to rank 2). Reranker rows
were **not** re-run after the fix (about 1 h of CPU per config).

## Abstention (6 unanswerable questions)

Top-1 dense cosine: answerable mean 0.634, unanswerable mean 0.529 (AUROC 0.94
baseline, 0.95 after fix), but the highest unanswerable score (0.599) exceeds the
lowest answerable score (0.542 / 0.561), so no threshold separates them cleanly
on this data. With 6 negatives this is anecdotal. No generation-level abstention
was evaluated.

## Not investigated

Chunk boundaries splitting tables or equations; the exercise/answer-key text
(which duplicates some definitions); whether answers span two chunks.
