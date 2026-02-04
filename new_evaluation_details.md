[evaluation](evaluation.py) dosyası minicons ile LM (language model) üstünden “good sentence daha olası mı?” değerlendirmesi yapıyor. Bizim model ise:

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("magibu/embeddingmagibu-200m")

Bu embedding modeli olduğu için sequence log-prob üretemez. Dolayısıyla makaledeki kodu “aynı şekilde” çalıştırmak imkânsız; ama aynı fikri (good > bad) embedding tabanlı bir skor ile birebir aynı formda ölçebiliriz:
• Her cümle için bir skor üret
• Her pair için: good_score > bad_score ise doğru
• Dosya bazında accuracy ve mean difference yaz

Embedding ile bunu yapmanın en temiz iki yaklaşımı var:

Yaklaşım A — Supervised (öneririm): Base set ile classifier eğit, experimental’da test et
• data/base/augmented\__.csv dosyalarını train olarak kullan
• Embedding’leri çıkar
• Logistic Regression gibi basit bir model eğit
• data/experimental/_.csv dosyalarında skor üret, pairwise accuracy ölç

Yaklaşım B — Zero-shot: “Acceptable centroid”a yakınlık
• Her dosya içinde good cümlelerin embedding ortalamasını al (centroid)
• Skor = cosine_similarity(sentence, centroid)
• Good’un skoru bad’den büyük mü?

Aşağıda sana tek bir script veriyorum; iki modu da destekliyor (supervised / centroid). Ayrıca senin minicons koduna benzer şekilde her pair için ayrı satır sonuç CSV’si de yazıyor.

⸻

1. Gerekli kurulum

pip install -U sentence-transformers scikit-learn pandas numpy

Eğer Excel’den insan yargılarıyla korelasyon da eklemek istersen sonra openpyxl da gerekir.

⸻

2. Script: eval_embeddings_turblimp.py

Proje kök dizininde (TurBLiMP klasöründe) çalıştıracağını varsayıyorum.

import os
import csv
import argparse
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# -----------------------------

# I/O

# -----------------------------

def load_pairs(filepath, delimiter=";"):
"""
CSV formatı: ilk iki kolon good_sentence ; bad_sentence (header var)
"""
pairs = []
with open(filepath, "r", encoding="utf-8") as f:
reader = csv.reader(f, delimiter=delimiter)
header = next(reader, None)
for row in reader:
if not row or len(row) < 2:
continue
good = row[0].strip()
bad = row[1].strip()
if good and bad:
pairs.append((good, bad))
return pairs

def list_csvs(folder):
return sorted([
os.path.join(folder, fn)
for fn in os.listdir(folder)
if fn.endswith(".csv")
])

# -----------------------------

# Embedding helpers

# -----------------------------

def encode(model, sentences, batch_size=64):
"""
normalize_embeddings=True -> cosine benzeri skorlama için iyi.
"""
return model.encode(
sentences,
batch_size=batch_size,
show_progress_bar=False,
convert_to_numpy=True,
normalize_embeddings=True
)

def cosine_score_matrix(X, v):
"""
X: (n, d) normalize edilmiş embedding
v: (d,) normalize edilmiş centroid
skor = X dot v (cosine similarity)
"""
return X @ v

# -----------------------------

# Mode A: Zero-shot centroid

# -----------------------------

def score*file_centroid(st_model, pairs, batch_size=64):
"""
Dosya içindeki good cümlelerin centroid'ine cosine benzerliği skor olarak kullanır.
"""
good_sents = [g for g, * in pairs]
bad*sents = [b for *, b in pairs]

    all_sents = good_sents + bad_sents
    E = encode(st_model, all_sents, batch_size=batch_size)

    Eg = E[:len(good_sents)]
    Eb = E[len(good_sents):]

    centroid = Eg.mean(axis=0)
    # normalize (zaten normalize embeddings ama centroid sonrası tekrar normalize iyi)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    good_scores = cosine_score_matrix(Eg, centroid)
    bad_scores  = cosine_score_matrix(Eb, centroid)
    return good_scores, bad_scores

# -----------------------------

# Mode B: Supervised classifier

# -----------------------------

def build*train_from_base(st_model, base_files, batch_size=64):
"""
Base klasöründeki augmented*\*.csv dosyalarını train dataseti yapar:
good -> 1, bad -> 0
"""
sentences = []
labels = []

    for fp in base_files:
        pairs = load_pairs(fp)
        for g, b in pairs:
            sentences.append(g); labels.append(1)
            sentences.append(b); labels.append(0)

    X = encode(st_model, sentences, batch_size=batch_size)
    y = np.array(labels, dtype=int)
    return X, y

def train_classifier(X, y):
"""
Basit ama güçlü bir baseline: Logistic Regression
decision_function -> sıralama/pairwise karşılaştırma için uygun skor üretir.
"""
clf = Pipeline([
("scaler", StandardScaler(with_mean=False)),
("lr", LogisticRegression(
max_iter=2000,
class_weight="balanced",
n_jobs=None
))
])
clf.fit(X, y)
return clf

def score*file_supervised(st_model, clf, pairs, batch_size=64):
"""
Eğitilmiş classifier ile:
score = decision_function(embedding)
"""
good_sents = [g for g, * in pairs]
bad*sents = [b for *, b in pairs]
all_sents = good_sents + bad_sents

    E = encode(st_model, all_sents, batch_size=batch_size)

    scores = clf.decision_function(E)
    good_scores = scores[:len(good_sents)]
    bad_scores  = scores[len(good_sents):]
    return good_scores, bad_scores

# -----------------------------

# Metrics + output

# -----------------------------

def compute_results(pairs, good_scores, bad_scores):
results = []
diffs = []
corrects = 0

    for (g, b), sg, sb in zip(pairs, good_scores, bad_scores):
        diff = float(sg - sb)
        ok = bool(sg > sb)
        results.append({
            "good_sentence": g,
            "bad_sentence": b,
            "good_score": float(sg),
            "bad_score": float(sb),
            "difference": diff,
            "correct": ok
        })
        diffs.append(diff)
        corrects += int(ok)

    mean_difference = float(np.mean(diffs)) if diffs else 0.0
    accuracy = float(corrects / len(pairs)) if pairs else 0.0
    return results, mean_difference, accuracy

def write_results_csv(out_path, results):
os.makedirs(os.path.dirname(out_path), exist_ok=True)
if not results:
return
with open(out_path, "w", encoding="utf-8", newline="") as f:
writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
writer.writeheader()
writer.writerows(results)

# -----------------------------

# Main

# -----------------------------

def main():
ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="data", help="data klasörü (base/experimental içerir)")
ap.add_argument("--mode", choices=["centroid", "supervised"], default="supervised")
ap.add_argument("--model_name", default="magibu/embeddingmagibu-200m")
ap.add_argument("--output_dir", default="scores_embeddings")
ap.add_argument("--batch_size", type=int, default=64)
ap.add_argument("--delimiter", default=";", help="CSV delimiter (genelde ;)")
args = ap.parse_args()

    base_dir = os.path.join(args.data_dir, "base")
    exp_dir  = os.path.join(args.data_dir, "experimental")

    st_model = SentenceTransformer(args.model_name)

    # 1) supervised ise base'ten eğitim
    clf = None
    if args.mode == "supervised":
        base_files = list_csvs(base_dir)
        X, y = build_train_from_base(st_model, base_files, batch_size=args.batch_size)
        clf = train_classifier(X, y)

    # 2) experimental dosyaları işle
    exp_files = list_csvs(exp_dir)
    summary_rows = []

    for fp in exp_files:
        pairs = load_pairs(fp, delimiter=args.delimiter)

        if args.mode == "centroid":
            good_scores, bad_scores = score_file_centroid(st_model, pairs, batch_size=args.batch_size)
        else:
            good_scores, bad_scores = score_file_supervised(st_model, clf, pairs, batch_size=args.batch_size)

        results, mean_diff, acc = compute_results(pairs, good_scores, bad_scores)

        out_name = f"{args.model_name.replace('/', '__')}__{os.path.basename(fp)}"
        out_path = os.path.join(args.output_dir, out_name)
        write_results_csv(out_path, results)

        summary_rows.append({
            "file_name": os.path.basename(fp),
            "mode": args.mode,
            "mean_difference": mean_diff,
            "accuracy": acc,
            "total_pairs": len(pairs),
            "output_file": out_path
        })

        print(f"Processed {os.path.basename(fp)}")
        print(f"  Mean difference: {mean_diff:.4f}")
        print(f"  Accuracy:        {acc:.4f}")

    # 3) genel özet CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, f"SUMMARY__{args.model_name.replace('/', '__')}__{args.mode}.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print("\nWrote summary:", summary_path)

if **name** == "**main**":
main()

⸻

3. Nasıl çalıştıracaksın?

(Önerilen) Supervised mod

Base set ile öğren, experimental’da test et:

python eval_embeddings_turblimp.py \
 --data_dir data \
 --mode supervised \
 --model_name magibu/embeddingmagibu-200m \
 --output_dir scores_embeddings

Zero-shot centroid mod

Hiç eğitim yok, sadece centroid yakınlığı:

python eval_embeddings_turblimp.py \
 --data_dir data \
 --mode centroid \
 --model_name magibu/embeddingmagibu-200m \
 --output_dir scores_embeddings

Çıktılar:
• Her experimental CSV için: scores_embeddings/<model>**<file>.csv
• good_sentence, bad_sentence, good_score, bad_score, difference, correct
• Ek olarak bir özet:
• scores_embeddings/SUMMARY**<model>\_\_<mode>.csv

⸻

4. Adım adım “bu script makaledeki koda nasıl denk geliyor?”

Makaledeki kodun mantığı: 1. pairs = load_sentences(file) ✅ biz de aynı yapıyoruz 2. score = model.sequence_score(pair)
• LM ise logprob;
• Biz embedding’de “acceptability skoru” üretiyoruz:
• supervised: decision_function(embedding)
• centroid: cosine(embedding, good_centroid) 3. correct = score_good > score_bad ✅ birebir aynı 4. mean_difference ve accuracy ✅ birebir aynı 5. Sonuçları dosyaya yaz ✅ birebir aynı

⸻

5. Hangi modu seçmelisin?
   • Supervised: “embedding model acceptability sinyali taşıyor mu?” sorusunda en anlamlı test.
   Çünkü embedding tek başına “gramer” skoru değildir; bunu basit bir karar sınırına map etmek gerekir.
   • Centroid: hızlı bir sanity-check, ama supervised kadar güçlü değil.

⸻

6. İstersen bir sonraki adım: İnsan yargılarıyla Pearson korelasyonu

Senin klasörde:
• data/human_judgments/experimental_judgments.xlsx

var. Eğer bu dosyanın içinde her fenomen/dosya için insan puanları (good/bad ortalamaları) varsa, makaledeki gibi:
• human_diff = mean(human_good) - mean(human_bad)
• model_diff = mean(model_good_score) - mean(model_bad_score)
• Pearson r

hesaplayacak ek bir modül de ekleyebilirim.

Bunun için tek kritik şey: Excel’de sütun adları ve hangi dosya/phenomenon ile eşleştiği.

İstersen experimental_judgments.xlsx’in ilk sayfasından (sheet) 10-15 satırlık örnek + kolon isimlerini buraya yapıştır; ben script’e Pearson kısmını da “tam oturan” şekilde ekleyeyim.
