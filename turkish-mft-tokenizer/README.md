---
license: cc-by-nc-4.0
---

# Turkish MFT Tokenizer

This is a custom Turkish tokenizer based on Meaningful Formal Tokenization (MFT).
It is compatible with Hugging Face Transformers.

## Usage

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("alibayram/turkish-mft-tokenizer", trust_remote_code=True)

text = "Merhaba nasılsın?"
tokens = tokenizer.tokenize(text)
print(tokens)

ids = tokenizer.encode(text)
print(ids)

decoded_text = tokenizer.decode(ids)
print(decoded_text)
```

## 🚀 High Performance Version (Rust)

If you need higher performance (~100x faster) for large-scale processing, you can use the Rust-based PyPI package which implements the exact same logic.

**Installation:**

```bash
pip install turkish-tokenizer
```

**Usage:**

```python
from turkish_tokenizer import TurkishTokenizer

# Initialize the Rust-backed tokenizer
tokenizer = TurkishTokenizer()

tokens = tokenizer.encode("Bugün hava çok güzel.")
print(tokens)
# Output: [2, 0, 1234, ... ]

decoded = tokenizer.decode(tokens)
print(decoded)
# Output: Bugün hava çok güzel.
```

See [turkish-tokenizer on PyPI](https://pypi.org/project/turkish-tokenizer/) for more details.

## Structure

- `tokenization_turkish_mft.py`: The tokenizer implementation and HF wrapper.
- `vocabs/`: Directory containing vocabulary and morphological rules JSON files.
