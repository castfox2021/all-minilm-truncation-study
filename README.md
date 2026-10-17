# Reproduction Package: Xenova/all-MiniLM-L6-v2 Truncation & Chunking Study

This reproduction package contains code, dependencies, and verified results demonstrating the empirical impact of sequence truncation (input truncated to 512 tokens including special tokens) and passage chunking on dense retrieval in a Node.js / Transformers.js environment using `Xenova/all-MiniLM-L6-v2`.

## 1. Pinned Metadata & Integrity Hashes

- **Target Model:** `Xenova/all-MiniLM-L6-v2`
- **Pinned Commit SHA:** `751bff37182d3f1213fa05d7196b954e230abad9`
- **Tokenizer Config:** `model_max_length: 512` ([Hugging Face Tokenizer Config](https://huggingface.co/Xenova/all-MiniLM-L6-v2/raw/751bff37182d3f1213fa05d7196b954e230abad9/tokenizer_config.json))
- **Upstream PyTorch Model:** `sentence-transformers/all-MiniLM-L6-v2` (`max_seq_length: 256` default in `sentence_bert_config.json`)
- **Runtime Library:** `@xenova/transformers` (installed version: `v2.17.2`)
- **Runtime Environment:** Node.js v22.14.0 (macOS Darwin x64)
- **Validation Scope:** Repeated Local Execution (Same machine / environment). Independent external cross-validation remains pending.

### Model File SHA-256 Hashes (Recorded at runtime)
The script downloads the pinned revision (`751bff37182d3f1213fa05d7196b954e230abad9`) and records the following SHA-256 file hashes:

| File Path | Size | SHA-256 Checksum |
| :--- | :---: | :--- |
| `config.json` | 0.6 KB | `7135149f7cffa1a573466c6e4d8423ed73b62fd2332c575bf738a0d033f70df7` |
| `onnx/model_quantized.onnx` | 22.4 MB | `afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1` |
| `tokenizer.json` | 695.0 KB | `da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0` |
| `tokenizer_config.json` | 0.4 KB | `9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3` |

> **Synthetic String Notice:** The query string `"What is the secret deployment key for Project Lavenbee?"` and target fact `"The secret deployment key for Project Lavenbee is OMEGA-7749-DELTA."` use a synthetic test token (`OMEGA-7749-DELTA`) created solely for this controlled benchmark experiment.

## 2. Experimental Setup & Comparison Design

- **Test Document:** An 804-token technical document regarding distributed systems background.
- **Three Placement Variants:**
  - Case A (Front): Target fact at token offset 1 (immediately after `[CLS]`).
  - Case B (Middle): Target fact placed between paragraphs at token offset 315.
  - Case C (End): Target fact at token offset 786 (strictly beyond the 512 truncation cutoff).
- **Distractor Documents:** 5 unrelated domain passages (Kubernetes, PostgreSQL, Redis, OAuth/PKCE, S3).
- **Comparison Methodology:**
  - **Unchunked:** Document encoded with `truncation: true, max_length: 512`.
  - **Chunked (MaxP):** Document segmented using a sliding window (80 words, 20 words overlap; ~105 tokens/chunk). Document score is evaluated via Max Passage (`max(chunk_scores)`), a standard dense retrieval aggregation strategy.
  - Distractor passages (shorter than chunk size) receive single-passage embeddings under the same scoring rule.

## 3. Reproduction Instructions

### Prerequisites
- Node.js >= 18.0.0
- npm

### Installation & Execution (Clean Fresh Run)
```bash
# 0. Clone repository
git clone https://github.com/castfox2021/all-minilm-truncation-study.git
cd all-minilm-truncation-study

# 1. Install exact dependencies using package-lock.json
npm ci

# 2. Run reproduction script (automatically downloads pinned revision into .cache on first run)
node reproduce.mjs

# 3. (Optional) Run in offline mode once cache is populated
node reproduce.mjs --offline
```

The script will:
1. Download the pinned revision (`751bff37182d3f1213fa05d7196b954e230abad9`) if not present in `.cache/`.
2. Compute and print the SHA-256 integrity hash of each downloaded model component.
3. Verify token lengths, special token insertion (`[CLS]`, `[SEP]`), and fact token subsequence preservation.
4. Compute cosine similarity against query for unchunked documents, chunked passages, and distractors.
5. Validate assertion criteria (exit code `0` on verified pass, `1` on failure).
6. Save raw structured results to `reproduction_results.json`.

## 4. Measured Results (Repeated Local Execution)

```
┌─────────┬────────────────────────────────┬───────────┬────────────────────┬──────────────┬───────────────┬────────────┬──────────────┬───────────────┬────────────────┬────────────────┬───────────────────┐
│ (index) │ placement                      │ rawTokens │ factPreservedIn512 │ unchunkedSim │ unchunkedRank │ isTop1Miss │ maxPChunkSim │ maxPChunkRank │ targetChunkIdx │ targetChunkSim │ isBestChunkTarget │
├─────────┼────────────────────────────────┼───────────┼────────────────────┼──────────────┼───────────────┼────────────┼──────────────┼───────────────┼────────────────┼────────────────┼───────────────────┤
│ 0       │ 'Fact at Beginning (Front)'    │ 804       │ 'YES'              │ 0.3027       │ '#1 / 6'      │ false      │ 0.5107       │ '#1 / 6'      │ 0              │ 0.5107         │ true              │
│ 1       │ 'Fact at Middle (Tokens ~315)' │ 804       │ 'YES'              │ 0.3177       │ '#1 / 6'      │ false      │ 0.5402       │ '#1 / 6'      │ 3              │ 0.5402         │ true              │
│ 2       │ 'Fact at End (Tokens ~786)'    │ 804       │ 'NO (LOST)'        │ 0.1759       │ '#4 / 6'      │ true       │ 0.5132       │ '#1 / 6'      │ 8              │ 0.5132         │ true              │
└─────────┴────────────────────────────────┴───────────┴────────────────────┴──────────────┴───────────────┴────────────┴──────────────┴───────────────┴────────────────┴────────────────┴───────────────────┘
```

- **Token Truncation:** For Case C, the fact at token 786 is strictly truncated out of the 512-token input tensor (`factPreservedIn512 = NO`).
- **Top-1 Miss:** Case C unchunked similarity drops to `0.1759`, falling below Distractor 4 (`0.2680`) and Distractor 3 (`0.2375`), resulting in Rank #4.
- **Chunking Benefit:** In this synthetic case study, chunking isolates the fact into Chunk 8 (similarity `0.5132`) and improves document retrieval to Rank #1.
