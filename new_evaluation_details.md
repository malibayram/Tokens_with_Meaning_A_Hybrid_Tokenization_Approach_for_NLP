# New Evaluation Details (Updated)

This document explains how the evaluation artifacts in this repo are produced from the latest runs (models recreated from scratch).

---

## 1) STS (STSb-TR)

**Evaluator**
- Script: `evaluate_sts_tr.py`
- Dataset: `figenfikri/stsb_tr`
- Metric: Pearson / Spearman correlation between cosine similarity of embeddings and gold scores (normalized from 0–5 to 0–1).

**Where results live**
- Raw log of runs: `sts_benchmark_results.json`

**Paper artifacts**
- Figure(s) + LaTeX table are generated from the JSON (no re-eval) via:
  - `python3 generate_sts_artifacts.py`
- Outputs:
  - `figures/sts_benchmark_chart_test.png`
  - `figures/sts_benchmark_chart_train.png`
  - `tables/sts_results.tex`

---

## 2) MTEB-TR

**Where results live**
- MTEB JSON outputs are under: `results/<model>/<revision>/*.json`

**Markdown report**
- `python3 generate_mteb_report.py`
- Output: `MTEB_BENCHMARK_RESULTS.md`

**Charts (JPEG)**
- `python3 visualize_mteb.py`
- Outputs:
  - `figures/mteb_comparison.jpg` (overall)
  - `figures/mteb_comparison_by_category.jpg` (by category)

**Paper LaTeX tables**
- `python3 generate_mteb_latex_tables.py`
- Outputs:
  - `tables/mteb_category_averages.tex`
  - `tables/mteb_detailed.tex`

---

## 3) TurBLiMP (centroid proxy, with colors sourced from HTML)

TurBLiMP is a Turkish benchmark of linguistic minimal pairs. Since our models are **embedding encoders** (not generative LMs), we use a **centroid-based acceptability proxy**:

For each phenomenon CSV containing `(good_sentence, bad_sentence)` pairs:
1) Embed all sentences with normalized embeddings.
2) Compute the centroid of the **good** sentence embeddings (then normalize the centroid).
3) Score each sentence by cosine similarity to the centroid: `score = embedding · centroid`.
4) Count the pair as correct iff `score(good) > score(bad)`.
5) Report **pairwise accuracy (%)** per phenomenon, plus a model average.

### 3.1 Run the centroid evaluation

Script:
- `eval_embeddings_turblimp.py`

Expected folder layout (configurable):
- `data/experimental/*.csv`

Run (example with the 4 random-init models):
```bash
python3 eval_embeddings_turblimp.py \
  --data_dir data \
  --experimental_subdir experimental \
  --models \
    alibayram/mft-random \
    alibayram/newmindaiMursit-random \
    alibayram/cosmosGPT2-random \
    alibayram/tabi-random \
  --output_dir turblimp_results_tables
```

Outputs:
- `turblimp_results_tables/results_table_centroid.csv`
- `turblimp_results_tables/results_table_centroid.html`
- `turblimp_results_tables/results_table_centroid.tex` (plain, no cell colors)

### 3.2 Use HTML to carry cell colors into the paper

The paper uses the **HTML** as the source of truth for cell background colors (Pandas Styler gradient), and converts it to colored LaTeX.

Run:
```bash
python3 generate_turblimp_tables_from_html.py
```

Outputs:
- `turblimp_results_tables/results_table_centroid_colored.tex` (uses `\cellcolor[RGB]{...}`)
- `turblimp_results_tables/results_table_centroid_inlined.html` (same table, inline `style=...`)

The paper includes:
- `turblimp_results_tables/results_table_centroid_colored.tex`

If you just want to view the colored table in a browser/editor, open:
- `turblimp_results_tables/results_table_centroid_inlined.html`

