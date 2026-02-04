import difflib

from datasets import load_dataset

from turkish_tokenizer import TurkishTokenizer


def evaluate_long_text():
    tokenizer = TurkishTokenizer()
    print("Loading dataset 'alibayram/cosmos-corpus-00-5'...")
    try:
        ds = load_dataset("alibayram/cosmos-corpus-00-5", split="train", streaming=True)
        it = iter(ds)
        first_item = next(it)
        text_column = "text" if "text" in first_item else list(first_item.keys())[0]

        rows = []
        print("Fetching 100 rows to concatenate...")
        count = 0
        for _ in range(500):  # Fetch enough to get 100 good rows
            if count >= 100:
                break
            try:
                item = next(it)
                text = item[text_column]
                if text and len(text.strip()) > 5:
                    rows.append(text.strip())
                    count += 1
            except StopIteration:
                break
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Concatenate rows
    long_text = "\n\n".join(rows)
    print(f"Total Text Length: {len(long_text)} characters")

    # 1. Tokenize
    print("Tokenizing...")
    ids = tokenizer.encode(long_text)
    print(f"Total Tokens: {len(ids)}")

    # 2. Decode
    print("Decoding...")
    decoded_text = tokenizer.decode(ids)

    # 3. Calculate Word-Based Accuracy
    # Normalize by splitting (ignoring multiple spaces/newlines differences if any)
    original_words = long_text.split()
    decoded_words = decoded_text.split()

    print(f"Original Word Count: {len(original_words)}")
    print(f"Decoded Word Count: {len(decoded_words)}")

    matcher = difflib.SequenceMatcher(None, original_words, decoded_words)

    match_count = 0
    errors = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            match_count += i2 - i1
        else:
            errors.append((tag, original_words[i1:i2], decoded_words[j1:j2]))

    accuracy = (match_count / len(original_words)) * 100

    print(f"\nWord-Based Accuracy: {accuracy:.2f}%")
    print(f"Correct Words: {match_count}")
    print(f"Total Reference Words: {len(original_words)}")

    print("\n--- Sample Errors (First 10) ---")
    for i, (tag, orig, dec) in enumerate(errors[:10]):
        print(f"{i+1}. {tag}: {orig} -> {dec}")

    # Generate Report
    report = f"""# Long Text Word-Based Evaluation (RUST)

**Dataset:** alibayram/cosmos-corpus-00-5 (First 100 non-empty rows)
**Total Characters:** {len(long_text)}
**Total Tokens:** {len(ids)}

## Results
- **Word Accuracy:** {accuracy:.2f}%
- **Correct Words:** {match_count}
- **Total Words:** {len(original_words)}

## Mismatches (Sample)
"""
    for tag, orig, dec in errors[:20]:
        report += f"- **{tag}**: `{orig}` -> `{dec}`\n"

    with open("LONG_TEXT_EVAL_REPORT_RUST.md", "w") as f:
        f.write(report)
    print("Report saved to LONG_TEXT_EVAL_REPORT_RUST.md")


if __name__ == "__main__":
    evaluate_long_text()
