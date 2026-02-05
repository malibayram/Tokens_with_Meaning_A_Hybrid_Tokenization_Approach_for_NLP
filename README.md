# Tokens with Meaning: A Hybrid Tokenization Approach for Turkish

This repository contains the LaTeX sources for the paper and the scripts/artifacts used to train and evaluate the embedding models referenced in the experiments (STS-TR, MTEB-TR, TurBLiMP proxy).

## Paper

- Main TeX: `main.tex` (includes sections from `chapters/`)
- Style: `neurips_2024.sty`
- References: `tokenizer.bib`
- Prebuilt PDFs (if present in the repo): `main.pdf`, `2025.emnlp-main.834.pdf`

### Build the PDF locally

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

## Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Notes:

- Some scripts fetch datasets/models from Hugging Face at runtime and may require credentials (see `.env` usage in `train.py`).
- `setup_server.sh` is an Ubuntu/Debian convenience script (uses `apt`) for remote machines.

## Experiments and artifacts

Details on how the repo’s evaluation artifacts are produced are in `new_evaluation_details.md`.

### STS benchmark (STSb-TR)

- Evaluate one or more models:
  ```bash
  python3 evaluate_sts_tr.py --model alibayram/mft-random
  ```
- Results JSON: `sts_benchmark_results.json`
- Generate paper charts/tables (from the JSON, no re-eval):
  ```bash
  python3 generate_sts_artifacts.py
  ```

### MTEB benchmark (MTEB-TR)

- Evaluate one:

  ```bash
  git clone https://github.com/selmanbaysan/mteb_tr.git
  python3 mteb_tr/mteb_tr_cli.py alibayram/mft-random
  ```

- MTEB raw outputs live under `results/<model>/<revision>/*.json`.
- Generate the markdown report:
  ```bash
  python3 generate_mteb_report.py
  ```
  Output: `MTEB_BENCHMARK_RESULTS.md`

### TurBLiMP (centroid proxy for embedding models)

The TurBLiMP scripts expect a local TurBLiMP/data folder:

- Expected layout: `TurBLiMP/data/experimental/*.csv` with `(good_sentence, bad_sentence)` pairs.
- Run evaluation:
  ```bash
  git clone https://github.com/ezgibasar/TurBLiMP.git
  python3 eval_embeddings_turblimp.py --models alibayram/mft-random
  ```
- Convert the styled HTML table to colored LaTeX:
  ```bash
  python3 generate_turblimp_tables_from_html.py
  ```

### Long-text roundtrip check

`pipi install turkish_tokenizer` to get the required tokenizer for this script. Then:

`evaluate_long_text.py` If you have that tokenizer available (e.g., as a local module or installed package), the script writes `LONG_TEXT_EVAL_REPORT_RUST.md`.

## Repository layout

- `chapters/`: paper sections (`abstract.tex`, `methodology.tex`, `results_and_analysis.tex`, etc.)
- `figures/`: generated figures used in the paper (e.g., MTEB charts)
- `results/`: stored evaluation outputs (MTEB JSONs by model/revision)
- `turblimp_results_tables/`: generated TurBLiMP summary tables
- `*.py`: training/evaluation and report-generation scripts

## Authors / Contact

Authors (as listed in `main.tex`): M. Ali Bayram, Ali Arda Fincan, Ahmet Semih Gümüş, Sercan Karakaş, Banu Diri, Savaş Yıldırım, Demircan Çelik.

Contact: `malibayram20@gmail.com`

## Citation

If you use this repository, please cite the paper. (Add the final venue/year once published.)

## License

No `LICENSE` file is currently provided. If you need reuse permissions for the paper/code, please contact the authors.
