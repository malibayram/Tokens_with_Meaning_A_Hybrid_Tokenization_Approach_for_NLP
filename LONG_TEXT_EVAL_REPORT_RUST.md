# Roundtrip Reconstruction Evaluation (RUST)

**Dataset:** alibayram/cosmos-corpus-00-5 (First 100 non-empty rows)
**Total Characters:** 547765
**Total Tokens:** 189983
**Total Words:** 66547

## Results
- **Word-level exact-match accuracy (decode(encode(w))):** **99.48%** (66,200 / 66,547 words)
- **Full-text word-alignment accuracy (encode/decode the concatenated text once):** 99.84% (66,442 / 66,547 words)

Notes:
- The paper reports the **word-level exact-match** metric (per-word roundtrip).
- The full-text alignment metric is included as an additional diagnostic; it can be higher because it measures word sequence alignment after a single full-text encode/decode pass rather than strict per-word exact reconstruction.

## Mismatches (Sample)
- **replace**: `['tetkiki']` -> `['tetkiği']`
- **replace**: `['çağırıyordu']` -> `['çağrıyordu']`
- **replace**: `['çağırıyordu']` -> `['çağrıyordu']`
- **replace**: `['hand,']` -> `['hant,']`
- **replace**: `['ittifakından']` -> `['ittifağından']`
- **replace**: `['sonuclarina']` -> `['sonuçlarina']`
- **replace**: `['kınıma']` -> `['kınama']`
- **replace**: `['emiri!']` -> `['emri!']`
- **replace**: `['kınıma']` -> `['kınama']`
- **replace**: `['emiri!']` -> `['emri!']`
- **replace**: `['kınıma']` -> `['kınama']`
- **replace**: `['emiri!']` -> `['emri!']`
- **replace**: `['gerekecektir.Girdi']` -> `['gereğecektir.Girdi']`
- **replace**: `['gerekebilir.']` -> `['gereğebilir.']`
- **replace**: `['sürecleri']` -> `['süreçleri']`
- **replace**: `['türü']` -> `['türe']`
- **replace**: `['gerekecektir.\\n\\nGirdi']` -> `['gereğecektir.\\n\\nGirdi']`
- **replace**: `['gerekebilir.']` -> `['gereğebilir.']`
- **replace**: `['sürecleri']` -> `['süreçleri']`
- **replace**: `['türü']` -> `['türe']`
