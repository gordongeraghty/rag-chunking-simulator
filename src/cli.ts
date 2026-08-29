#!/usr/bin/env node

import { calculateRagMetrics } from './engine/metrics.ts'
import type { RagSimulationInputs } from './engine/metrics.ts'
import { EMBEDDING_MODELS } from './engine/models.ts'

function parseArgs(): RagSimulationInputs {
  const args = process.argv.slice(2)
  const inputs: RagSimulationInputs = {
    corpusTokenCount: 5_000_000,
    chunkSize: 512,
    chunkOverlapPct: 15,
    embeddingModelId: 'text-embedding-3-small',
    vectorIndexType: 'hnsw',
    topKRetrieval: 5,
  }

  for (let i = 0; i < args.length; i++) {
    const arg = args[i]
    if (arg === '--tokens' || arg === '-t') {
      inputs.corpusTokenCount = parseInt(args[++i], 10) || 5_000_000
    } else if (arg === '--chunk' || arg === '-c') {
      inputs.chunkSize = parseInt(args[++i], 10) || 512
    } else if (arg === '--overlap' || arg === '-o') {
      inputs.chunkOverlapPct = parseInt(args[++i], 10) || 15
    } else if (arg === '--model' || arg === '-m') {
      inputs.embeddingModelId = args[++i] || 'text-embedding-3-small'
    } else if (arg === '--index' || arg === '-i') {
      inputs.vectorIndexType = (args[++i] as any) || 'hnsw'
    } else if (arg === '--help' || arg === '-h') {
      console.log(`
RAG Chunking & Vector DB Simulator CLI

Usage:
  rag-chunk-sim [options]

Options:
  -t, --tokens <number>    Total corpus size in tokens (default: 5,000,000)
  -c, --chunk <number>     Chunk size in tokens [128-2048] (default: 512)
  -o, --overlap <number>   Sliding window overlap percentage [0-40] (default: 15)
  -m, --model <id>         Embedding model ID (default: text-embedding-3-small)
                           Models: ${Object.keys(EMBEDDING_MODELS).join(', ')}
  -i, --index <type>       Index structure: hnsw, ivf_pq, flat (default: hnsw)
  -h, --help               Display this help message
`)
      process.exit(0)
    }
  }

  return inputs
}

function run() {
  const inputs = parseArgs()
  const metrics = calculateRagMetrics(inputs)

  console.log(`
======================================================
RAG Chunking Architecture & Vector Topology Projection
======================================================
Corpus Size:           ${inputs.corpusTokenCount.toLocaleString()} tokens
Chunk Size:            ${inputs.chunkSize} tokens
Sliding Overlap:       ${inputs.chunkOverlapPct}%
Embedding Model:       ${metrics.model.name} (${metrics.model.dimensions} dims)
Vector Index Type:     ${inputs.vectorIndexType.toUpperCase()}
------------------------------------------------------
Total Chunks Created:  ${metrics.totalChunks.toLocaleString()}
Total Embedded Tokens: ${metrics.totalTokensEmbedded.toLocaleString()}
Ingestion API Cost:    $${metrics.embeddingCostUsd} USD
Vector RAM Footprint:  ${metrics.totalIndexMegabytes} MB
Estimated Precision:   ${metrics.estimatedPrecisionPct}%
Estimated Recall:      ${metrics.estimatedRecallPct}%
======================================================
`)
}

run()
