# RAG Chunking & Vector DB Simulator

A programmatic engine and CLI tool for modeling **Retrieval-Augmented Generation (RAG) chunking strategies**, calculating vector index memory footprints (HNSW, IVF-PQ, Flat), and projecting embedding ingestion costs across OpenAI, Cohere, and Open-Source models.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)

---

## Why This Exists

In enterprise RAG systems, **chunk size is the primary lever governing retrieval precision, context recall, and vector database RAM infrastructure costs**:

- **Micro-Chunks (128–256 tokens)**: Maximize semantic precision (exact answers are not diluted by surrounding text), but risk severing multi-sentence context unless high sliding overlap or hierarchical parent-document retrieval is implemented.
- **Macro-Chunks (1024–2048 tokens)**: Retain broad context, but dilute vector embeddings with multi-topic noise, causing top-k approximate nearest neighbor (ANN) retrieval to surface irrelevant passages.
- **Vector RAM Growth**: Vector databases (Pinecone, Qdrant, Milvus, pgvector) require holding index structures in memory. HNSW graphs add a ~1.5x memory multiplier over raw float32 vector dimensions, whereas quantized indices (IVF-PQ) compress memory down to ~0.3x at the cost of slight recall degradation.

This package lets engineering teams simulate and benchmark these trade-offs mathematically before committing to multi-million token ingestion pipelines.

---

## Architecture Flow

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 260" width="100%" height="260" role="img" aria-labelledby="rag-diag-title rag-diag-desc" style="background:#0f172a; border-radius:8px; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <title id="rag-diag-title">RAG Ingestion and Vector Index Memory Modeling</title>
  <desc id="rag-diag-desc">Diagram showing raw documents split into sliding window chunks, converted into dense vectors, and stored in indexed memory architectures.</desc>
  <defs>
    <linearGradient id="rag-blue" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8"/>
      <stop offset="100%" stop-color="#0284c7"/>
    </linearGradient>
  </defs>

  <!-- Document Ingestion -->
  <rect x="30" y="80" width="180" height="100" rx="8" fill="#1e293b" stroke="#38bdf8" stroke-width="2"/>
  <text x="120" y="115" fill="#f8fafc" font-size="14" font-weight="600" text-anchor="middle">Corpus Ingestion</text>
  <text x="120" y="138" fill="#94a3b8" font-size="12" text-anchor="middle">N Total Tokens</text>
  <text x="120" y="158" fill="#38bdf8" font-size="11" font-family="monospace" text-anchor="middle">Step = Chunk × (1 - Overlap)</text>

  <!-- Arrow 1 -> 2 -->
  <path d="M 210 130 L 280 130" stroke="#38bdf8" stroke-width="2" fill="none"/>

  <!-- Chunk Slicing -->
  <rect x="290" y="80" width="190" height="100" rx="8" fill="#1e293b" stroke="#34d399" stroke-width="2"/>
  <text x="385" y="115" fill="#f8fafc" font-size="14" font-weight="600" text-anchor="middle">Sliding Window</text>
  <text x="385" y="138" fill="#94a3b8" font-size="12" text-anchor="middle">Overlap Token Buffers</text>
  <text x="385" y="158" fill="#34d399" font-size="11" font-family="monospace" text-anchor="middle">Precision vs Recall</text>

  <!-- Arrow 2 -> 3 -->
  <path d="M 480 130 L 550 130" stroke="#34d399" stroke-width="2" fill="none"/>

  <!-- Vector Index -->
  <rect x="560" y="80" width="200" height="100" rx="8" fill="#1e293b" stroke="#c084fc" stroke-width="2"/>
  <text x="660" y="115" fill="#f8fafc" font-size="14" font-weight="600" text-anchor="middle">Vector Index RAM</text>
  <text x="660" y="138" fill="#94a3b8" font-size="12" text-anchor="middle">Float32 × Dims × Graph</text>
  <text x="660" y="158" fill="#c084fc" font-size="11" font-family="monospace" text-anchor="middle">HNSW (1.5x) / IVF-PQ (0.3x)</text>
</svg>

---

## Installation

```bash
# Clone repository
git clone https://github.com/gordongeraghty/rag-chunking-simulator.git
cd rag-chunking-simulator

# Install dependencies and build
npm install
npm run build
```

---

## CLI Usage

Run calculations directly from your terminal:

```bash
# Basic simulation for 5 million tokens with 512-token chunks and 15% overlap
npx rag-chunk-sim --tokens 5000000 --chunk 512 --overlap 15

# Dense technical Q&A using OpenAI text-embedding-3-large and HNSW index
npx rag-chunk-sim -t 10000000 -c 256 -o 20 -m text-embedding-3-large -i hnsw
```

### CLI Options

| Flag | Name | Description | Default |
|---|---|---|---|
| `-t`, `--tokens` | Total Corpus Tokens | Total source volume in tokens | `5000000` |
| `-c`, `--chunk` | Chunk Size | Segment size in tokens (128–2048) | `512` |
| `-o`, `--overlap` | Overlap % | Sliding window overlap percentage | `15` |
| `-m`, `--model` | Embedding Model | Model ID (`text-embedding-3-small`, `text-embedding-3-large`, `cohere-embed-v3`, `bge-large-en`) | `text-embedding-3-small` |
| `-i`, `--index` | Index Type | Vector index structure (`hnsw`, `ivf_pq`, `flat`) | `hnsw` |

---

## Programmatic TypeScript API

```typescript
import { calculateRagMetrics, simulateTextChunking } from 'rag-chunking-simulator'

// 1. Calculate Vector Database & Cost Topology
const metrics = calculateRagMetrics({
  corpusTokenCount: 5_000_000,
  chunkSize: 256,
  chunkOverlapPct: 20,
  embeddingModelId: 'text-embedding-3-large',
  vectorIndexType: 'hnsw',
})

console.log(metrics)
// {
//   totalChunks: 24510,
//   totalTokensEmbedded: 6274560,
//   embeddingCostUsd: 0.8157,
//   totalIndexMegabytes: 430.49,
//   estimatedPrecisionPct: 86,
//   estimatedRecallPct: 84,
//   model: { ... }
// }

// 2. Simulate Text Chunking Windows
const preview = simulateTextChunking(
  'Retrieval-Augmented Generation combines dense semantic search with LLMs...',
  128,
  15
)
```

---

## Math & Methodology

1. **Chunk Count Formula**:
   $$\text{Total Chunks} = \left\lceil \frac{\text{Corpus Tokens}}{\text{Chunk Size} \times (1 - \text{Overlap Pct})} \right\rceil$$

2. **Vector Index Memory Footprint**:
   $$\text{Memory (MB)} = \frac{\text{Total Chunks} \times \text{Dimensions} \times 4 \text{ bytes} \times \text{Index Multiplier}}{1024 \times 1024}$$
   - **HNSW**: Multiplier $= 1.5$ (Includes bidirectional layer graph adjacency lists).
   - **Flat**: Multiplier $= 1.0$ (Raw float32 vectors).
   - **IVF-PQ**: Multiplier $= 0.3$ (Product quantization compression).

---

## Limitations

- Token counts assume average English text tokenization (~0.75 words per token). CJK or code languages may exhibit different token density ratios.
- Precision and recall figures are parametric estimation curves based on standard information retrieval benchmarks (BEIR, MTEB); real domain recall depends on corpus embedding alignment.

---

## Related Tools

- [Hosted Interactive RAG Chunking Simulator](https://gordongeraghty.com/resources/ai-engineering/rag-chunking-embedding-benchmark) — Full visual web calculator with live sliding window boundary previews.
- [GA4 BigQuery SQL Builder](https://gordongeraghty.com/resources/gtm-analytics/ga4-bigquery-sql-query-library) — Query library for raw event analytics.

---

## Licence

Licensed under the [MIT License](LICENSE).
