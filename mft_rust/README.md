# Turkish Tokenizer

**PyPI version** | **Python 3.8+** | **License: MIT**

`turkish-tokenizer`, Türkçe doğal dil işleme görevleri için geliştirilmiş, dilbilim kurallarına (ünlü uyumu, ünsüz yumuşaması vb.) sadık kalan, yüksek performanslı ve hibrid (Rule-Based + BPE) bir tokenizer kütüphanesidir.

Rust ile yeniden yazılarak Python versiyonuna kıyasla **~100x performans artışı** sağlanmıştır.

## Özellikler

- **Hibrid Yapı**: Kök bulma ve ek analizi ile BPE (Byte Pair Encoding) yöntemlerini birleştirir.
- **Dilbilimsel Doğruluk**:
  - Ünlü uyumu (Vowel Harmony)
  - Ünsüz yumuşaması (Consonant Softening)
  - Ünsüz düşmesi ve daralması
- **Yüksek Performans**: Rust tabanlı motor ile milisaniyeler içinde büyük metinleri tokenize eder.
- **Tam Uyumluluk**: Orijinal Python implementasyonu ile %99.8 oranında çıktı benzerliği.

## Kurulum

PyPI üzerinden en güncel versiyonu indirebilirsiniz:

```bash
pip install turkish-tokenizer
```

## Hızlı Başlangıç

### Tokenizasyon (Encode)

Metinleri modelin anlayabileceği ID listelerine dönüştürmek için:

```python
from turkish_tokenizer import TurkishTokenizer

# Tokenizer'ı başlat
tokenizer = TurkishTokenizer()

text = "Bugün hava çok güzel."
tokens = tokenizer.encode(text)

print(f"Text: {text}")
print(f"Token IDs: {tokens}")
# Çıktı: [2, 0, 1234, 5678, ...]
```

### Geri Dönüştürme (Decode)

ID listelerini tekrar okunabilir metne dönüştürmek için. Decoder, kelimelerin kök ve eklerini birleştirirken ses olaylarını otomatik olarak uygular.

```python
decoded_text = tokenizer.decode(tokens)
print(f"Decoded: {decoded_text}")
# Çıktı: Bugün hava çok güzel.
```

## Gelişmiş Kullanım

Tokenizer, cümleleri parçalara ayırırken (tokenization) ve birleştirirken (decoding) bağlama duyarlı işlemler yapar.

```python
# Örnek: Yumuşama ve Ünlü Uyumu
ids = tokenizer.encode("Kitabı bana ver")
# "Kitap" + "ı" -> "Kitabı" (p -> b yumuşaması)

decoded = tokenizer.decode(ids)
assert decoded == " Kitabı bana ver"
```

## Performans

Bu kütüphane Rust dili (`pyo3` ve `maturin`) kullanılarak geliştirilmiştir. Büyük veri setlerinde (örneğin Cosmos Corpus) saniyede yüz binlerce kelimeyi işleyebilir.

## Lisans

Bu proje MIT lisansı ile lisanslanmıştır.
