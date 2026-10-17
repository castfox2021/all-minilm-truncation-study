import { pipeline, env, AutoTokenizer } from '@xenova/transformers';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Command-line flag or env var for offline mode
const isOffline = process.argv.includes('--offline') || process.env.OFFLINE === '1';

// Cache directory: default to local .cache directory in current package
const cacheDir = process.env.CACHE_DIR || path.resolve(__dirname, '.cache');
env.cacheDir = cacheDir;

if (isOffline) {
  env.allowRemoteModels = false;
  env.localModelPath = cacheDir;
} else {
  // Allow downloading the pinned revision from Hugging Face Hub if not cached
  env.allowRemoteModels = true;
}

const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
const PINNED_REVISION = '751bff37182d3f1213fa05d7196b954e230abad9';

function sha256File(filePath) {
  const fileBuffer = fs.readFileSync(filePath);
  return crypto.createHash('sha256').update(fileBuffer).digest('hex');
}

function findModelFiles(dir) {
  const results = [];
  if (!fs.existsSync(dir)) return results;
  function traverse(current) {
    const entries = fs.readdirSync(current, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        traverse(fullPath);
      } else if (entry.isFile()) {
        results.push(fullPath);
      }
    }
  }
  traverse(dir);
  return results;
}

async function main() {
  const startTime = new Date().toISOString();
  console.log('================================================================');
  console.log('MiniLM Input Truncation & Chunking Reproduction Study');
  console.log('================================================================');
  console.log(`Execution Time:          ${startTime}`);
  console.log(`Execution Mode:          ${isOffline ? 'OFFLINE (cache only)' : 'ONLINE / CACHED (downloads pinned revision if needed)'}`);
  console.log(`Node.js Version:         ${process.version}`);
  console.log(`Platform / Arch:         ${process.platform} (${process.arch})`);
  console.log(`Target Model:            ${MODEL_ID}`);
  console.log(`Pinned Revision:         ${PINNED_REVISION}`);
  console.log(`Cache Directory:         ${cacheDir}`);

  // Resolve actual installed library version from node_modules
  let installedTransformersVersion = 'unknown';
  try {
    const pkgPath = path.resolve(__dirname, 'node_modules', '@xenova', 'transformers', 'package.json');
    if (fs.existsSync(pkgPath)) {
      const pkgJson = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
      installedTransformersVersion = pkgJson.version;
    }
  } catch (e) {
    // fallback
  }
  console.log(`Installed Library:       @xenova/transformers v${installedTransformersVersion}`);
  console.log(`Validation Scope:        Repeated Local Execution (Same machine / environment)`);
  console.log(`Independent Cross-Valid: Pending external contributor environment`);

  console.log('\nLoading tokenizer and ONNX pipeline (quantized)...');
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { revision: PINNED_REVISION });
  const extractor = await pipeline('feature-extraction', MODEL_ID, {
    revision: PINNED_REVISION,
    quantized: true,
  });

  console.log(`Tokenizer Model Max Length: ${tokenizer.model_max_length} (Xenova ONNX config)`);
  console.log(`Upstream Model Max Length:  256 (sentence-transformers default max_seq_length)`);

  // Record SHA-256 of all cached model files
  console.log('\n----------------------------------------------------------------');
  console.log('Cached Model Files & SHA-256 Integrity Hashes:');
  console.log('----------------------------------------------------------------');
  const modelDir = path.join(cacheDir, MODEL_ID);
  const modelFiles = findModelFiles(modelDir);
  const fileHashRecords = [];

  for (const f of modelFiles) {
    const relPath = path.relative(modelDir, f);
    const hash = sha256File(f);
    const size = fs.statSync(f).size;
    fileHashRecords.push({ file: relPath, sizeBytes: size, sha256: hash });
    console.log(`  ${relPath.padEnd(55)} (${(size / 1024).toFixed(1)} KB) -> SHA256: ${hash}`);
  }

  // Helper: Cosine similarity
  function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  async function getEmbedding(text, options = {}) {
    const out = await extractor(text, {
      pooling: 'mean',
      normalize: true,
      truncation: true,
      ...options,
    });
    return Array.from(out.data);
  }

  // Synthetic Query and Target Fact (clearly labeled as synthetic benchmark strings)
  const query = "What is the secret deployment key for Project Lavenbee?";
  const targetFact = "The secret deployment key for Project Lavenbee is OMEGA-7749-DELTA.";

  // Distractors (technical passages unrelated to the query)
  const distractors = [
    { id: "Distractor 1", text: "Kubernetes pod lifecycle and container networking interface setup in microservice topologies." },
    { id: "Distractor 2", text: "PostgreSQL transactional isolation levels, MVCC vacuuming, and serialized snapshot isolation." },
    { id: "Distractor 3", text: "Redis in-memory pub-sub messaging, cluster sharding, and sentinel failover coordination." },
    { id: "Distractor 4", text: "OAuth 2.0 authorization code flow with PKCE verification and JSON Web Key Sets." },
    { id: "Distractor 5", text: "S3 compatible blob storage bucket replication, multipart upload concurrency, and lifecycle policies." },
  ];

  // Filler paragraph generator (~150 words per paragraph, ~160 tokens each)
  const makeParagraph = (title) => `Section ${title}: Distributed coordination systems require robust consensus mechanisms like Raft or Paxos to guarantee consistency across multi-region datacenters. When leader elections occur, network partitions can cause transient latency spikes. System operators must carefully monitor heartbeat intervals, lease renewals, and quorum acknowledgments. Data persistence layers utilize write-ahead logging (WAL) combined with log-structured merge (LSM) trees to maximize sequential write throughput on non-volatile memory express storage devices. Cache eviction algorithms such as W-TinyLFU offer high hit ratios under skewed access workloads. Failure detection protocols rely on phi-accrual detectors rather than static heartbeat timeouts to adapt dynamically to varying network conditions across interconnected cloud availability zones. `;

  const p1 = makeParagraph('A');
  const p2 = makeParagraph('B');
  const p3 = makeParagraph('C');
  const p4 = makeParagraph('D');
  const p5 = makeParagraph('E');

  // Documents with target fact at beginning, middle, and end
  const docFront = `${targetFact}\n\n${p1}\n\n${p2}\n\n${p3}\n\n${p4}\n\n${p5}`;
  const docMiddle = `${p1}\n\n${p2}\n\n${targetFact}\n\n${p3}\n\n${p4}\n\n${p5}`;
  const docEnd = `${p1}\n\n${p2}\n\n${p3}\n\n${p4}\n\n${p5}\n\n${targetFact}`;

  // Step 1: Token Analysis & Truncation Boundary Verification
  console.log('\n----------------------------------------------------------------');
  console.log('1. Token & Tensor Truncation Verification');
  console.log('----------------------------------------------------------------');

  const factTokenTensor = await tokenizer(targetFact, { add_special_tokens: false });
  const factTokenIds = Array.from(factTokenTensor.input_ids.data).map(Number);
  console.log(`Target Fact Token Count (excl special tokens): ${factTokenIds.length}`);
  console.log(`Target Fact Token IDs: [${factTokenIds.join(', ')}]`);

  function findSubsequence(arr, sub) {
    for (let i = 0; i <= arr.length - sub.length; i++) {
      let match = true;
      for (let j = 0; j < sub.length; j++) {
        if (arr[i + j] !== sub[j]) { match = false; break; }
      }
      if (match) return i;
    }
    return -1;
  }

  const cases = [
    { name: 'Fact at Beginning (Front)', doc: docFront, expectedOffset: 1 },
    { name: 'Fact at Middle (Tokens ~315)', doc: docMiddle, expectedOffset: 315 },
    { name: 'Fact at End (Tokens ~786)', doc: docEnd, expectedOffset: 786 },
  ];

  const tokenVerificationResults = [];

  for (const c of cases) {
    const rawTokens = Array.from((await tokenizer(c.doc, { truncation: false })).input_ids.data).map(Number);
    const truncTokens = Array.from((await tokenizer(c.doc, { truncation: true, max_length: 512 })).input_ids.data).map(Number);

    const rawPos = findSubsequence(rawTokens, factTokenIds);
    const truncPos = findSubsequence(truncTokens, factTokenIds);
    const isPreserved = truncPos !== -1;

    console.log(`- ${c.name}:`);
    console.log(`    Total raw tokens:   ${rawTokens.length}`);
    console.log(`    Truncated tokens:   ${truncTokens.length} (boundary = 512)`);
    console.log(`    Raw token offset:   ${rawPos}`);
    console.log(`    Trunc token offset: ${truncPos === -1 ? 'NOT FOUND (TRUNCATED)' : truncPos}`);
    console.log(`    Status:             ${isPreserved ? 'PRESERVED in 512-window' : 'LOST beyond 512-window'}`);

    tokenVerificationResults.push({
      name: c.name,
      rawLen: rawTokens.length,
      truncLen: truncTokens.length,
      rawPos,
      truncPos,
      isPreserved,
    });
  }

  // Step 2: Retrieval Evaluation (Unchunked vs Chunked MaxP)
  console.log('\n----------------------------------------------------------------');
  console.log('2. Embedding & Retrieval Ranking Evaluation');
  console.log('----------------------------------------------------------------');

  const queryEmb = await getEmbedding(query);
  const factEmb = await getEmbedding(targetFact);
  const directSim = cosineSimilarity(queryEmb, factEmb);
  console.log(`Direct Query <-> Fact Cosine Similarity: ${directSim.toFixed(4)}`);

  // Distractor embeddings
  const distractorScores = [];
  for (const d of distractors) {
    const emb = await getEmbedding(d.text);
    const sim = cosineSimilarity(queryEmb, emb);
    distractorScores.push({ name: d.id, sim });
    console.log(`  ${d.id}: sim = ${sim.toFixed(4)}`);
  }
  const maxDistractorScore = Math.max(...distractorScores.map(d => d.sim));
  console.log(`Max Distractor Baseline Score: ${maxDistractorScore.toFixed(4)}`);

  // Sliding window chunker (words: 80 words, overlap: 20 words)
  function chunkText(text, chunkSizeWords = 80, overlapWords = 20) {
    const words = text.split(/\s+/);
    const chunks = [];
    let start = 0;
    while (start < words.length) {
      const end = Math.min(start + chunkSizeWords, words.length);
      chunks.push(words.slice(start, end).join(' '));
      if (end === words.length) break;
      start += (chunkSizeWords - overlapWords);
    }
    return chunks;
  }

  const evaluationResults = [];

  for (let idx = 0; idx < cases.length; idx++) {
    const c = cases[idx];
    const tokVer = tokenVerificationResults[idx];

    // Method 1: Unchunked (truncated at 512)
    const unchunkedEmb = await getEmbedding(c.doc, { truncation: true });
    const unchunkedSim = cosineSimilarity(queryEmb, unchunkedEmb);

    const unchunkedPool = [{ name: 'Target Document', sim: unchunkedSim }, ...distractorScores]
      .sort((a, b) => b.sim - a.sim);
    const unchunkedRank = unchunkedPool.findIndex(item => item.name === 'Target Document') + 1;

    // Method 2: Chunked (MaxP aggregation)
    const chunks = chunkText(c.doc, 80, 20);
    const chunkScores = [];
    let targetChunkIndex = -1;
    let targetChunkSim = -1;

    for (let cIdx = 0; cIdx < chunks.length; cIdx++) {
      const chunkTextStr = chunks[cIdx];
      const hasFact = chunkTextStr.includes(targetFact);
      const cEmb = await getEmbedding(chunkTextStr, { truncation: true });
      const cSim = cosineSimilarity(queryEmb, cEmb);

      chunkScores.push({ index: cIdx, sim: cSim, hasFact });

      if (hasFact) {
        targetChunkIndex = cIdx;
        targetChunkSim = cSim;
      }
    }

    if (targetChunkIndex === -1) {
      throw new Error(`Integrity error: target fact not found in any chunk for ${c.name}`);
    }

    const maxPChunk = chunkScores.slice().sort((a, b) => b.sim - a.sim)[0];
    const chunkedPool = [{ name: 'Target Document (MaxP)', sim: maxPChunk.sim }, ...distractorScores]
      .sort((a, b) => b.sim - a.sim);
    const chunkedRank = chunkedPool.findIndex(item => item.name === 'Target Document (MaxP)') + 1;

    evaluationResults.push({
      placement: c.name,
      rawTokens: tokVer.rawLen,
      factPreservedIn512: tokVer.isPreserved ? 'YES' : 'NO (LOST)',
      unchunkedSim: Number(unchunkedSim.toFixed(4)),
      unchunkedRank: `#${unchunkedRank} / ${unchunkedPool.length}`,
      isTop1Miss: unchunkedRank > 1,
      maxPChunkSim: Number(maxPChunk.sim.toFixed(4)),
      maxPChunkRank: `#${chunkedRank} / ${chunkedPool.length}`,
      targetChunkIdx: targetChunkIndex,
      targetChunkSim: Number(targetChunkSim.toFixed(4)),
      isBestChunkTarget: maxPChunk.index === targetChunkIndex,
    });
  }

  // Step 3: Print Consolidated Results Table
  console.log('\n================================================================');
  console.log('RESULTS SUMMARY TABLE');
  console.log('================================================================');
  console.table(evaluationResults);

  // Assertion Checks for Reproduction Package Validity
  console.log('\n----------------------------------------------------------------');
  console.log('Sanity & Assertion Checks');
  console.log('----------------------------------------------------------------');

  const caseEnd = evaluationResults[2];

  // 1. End case must have fact truncated in 512 window
  if (caseEnd.factPreservedIn512 !== 'NO (LOST)') {
    throw new Error('Assertion failed: End case fact should be truncated past 512 tokens');
  }
  console.log('[PASS] Token Truncation: End case fact is confirmed truncated outside 512-window.');

  // 2. End case unchunked rank must drop below distractor baseline (Top-1 Miss)
  if (!caseEnd.isTop1Miss || caseEnd.unchunkedSim >= maxDistractorScore) {
    throw new Error('Assertion failed: End case unchunked should suffer Top-1 Miss below max distractor');
  }
  console.log(`[PASS] Top-1 Miss: End case unchunked similarity (${caseEnd.unchunkedSim}) drops below Distractor 4 (${maxDistractorScore.toFixed(4)}), resulting in Rank ${caseEnd.unchunkedRank}.`);

  // 3. Chunking with MaxP improves ranking to #1 in this case study
  if (caseEnd.maxPChunkRank !== '#1 / 6' || !caseEnd.isBestChunkTarget) {
    throw new Error('Assertion failed: Chunking should place target chunk at Rank #1 for End case');
  }
  console.log(`[PASS] Chunking Benefit: In this case study, chunking isolates target fact (Chunk ${caseEnd.targetChunkIdx}, sim ${caseEnd.targetChunkSim}) and improves retrieval to Rank #1.`);

  // Write machine-readable JSON results
  const outputPayload = {
    metadata: {
      timestamp: startTime,
      validationType: "repeated_local_execution",
      environment: `${process.platform}-${process.arch}`,
      nodeVersion: process.version,
      installedLibrary: `@xenova/transformers@${installedTransformersVersion}`,
      independentCrossValidationStatus: "pending_external_environment",
      model: MODEL_ID,
      pinnedRevision: PINNED_REVISION,
      model_max_length_onnx: tokenizer.model_max_length,
      upstream_default_max_length: 256,
      syntheticTokenNotice: "OMEGA-7749-DELTA is a synthetic dummy token created for this benchmark experiment.",
      modelFiles: fileHashRecords,
    },
    tokenVerification: tokenVerificationResults,
    evaluation: evaluationResults,
  };

  const resultsJsonPath = path.resolve(__dirname, 'reproduction_results.json');
  fs.writeFileSync(resultsJsonPath, JSON.stringify(outputPayload, null, 2), 'utf8');
  console.log(`\nMachine-readable results written to: ${resultsJsonPath}`);
  console.log('\n================================================================');
  console.log('Reproduction execution completed successfully.');
  console.log('================================================================');
}

main().catch((err) => {
  console.error('\n[FATAL ERROR during reproduction run]:', err);
  process.exit(1);
});
