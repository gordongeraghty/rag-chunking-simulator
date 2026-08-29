import test from 'node:test'
import assert from 'node:assert'
import { calculateRagMetrics } from '../src/engine/metrics.ts'
import { simulateTextChunking } from '../src/engine/chunker.ts'

test('calculateRagMetrics returns correct chunk count and cost calculation', () => {
  const result = calculateRagMetrics({
    corpusTokenCount: 1_000_000,
    chunkSize: 500,
    chunkOverlapPct: 0,
    embeddingModelId: 'text-embedding-3-small',
    vectorIndexType: 'hnsw',
  })

  assert.strictEqual(result.totalChunks, 2000)
  assert.strictEqual(result.totalTokensEmbedded, 1_000_000)
  assert.strictEqual(result.embeddingCostUsd, 0.02)
  assert.ok(result.totalIndexMegabytes > 0)
  assert.ok(result.estimatedPrecisionPct >= 50 && result.estimatedPrecisionPct <= 100)
  assert.ok(result.estimatedRecallPct >= 50 && result.estimatedRecallPct <= 100)
})

test('calculateRagMetrics handles overlap correctly', () => {
  const noOverlap = calculateRagMetrics({
    corpusTokenCount: 1_000_000,
    chunkSize: 500,
    chunkOverlapPct: 0,
    embeddingModelId: 'text-embedding-3-small',
    vectorIndexType: 'flat',
  })

  const withOverlap = calculateRagMetrics({
    corpusTokenCount: 1_000_000,
    chunkSize: 500,
    chunkOverlapPct: 20,
    embeddingModelId: 'text-embedding-3-small',
    vectorIndexType: 'flat',
  })

  assert.ok(withOverlap.totalChunks > noOverlap.totalChunks)
})

test('simulateTextChunking generates valid chunk list with overlaps', () => {
  const sample = 'Retrieval Augmented Generation combines semantic vector search with large language models to provide accurate contextual answers.'
  const chunks = simulateTextChunking(sample, 8, 25, 4)

  assert.ok(chunks.length > 0)
  assert.strictEqual(chunks[0].id, 1)
  assert.ok(chunks[0].text.length > 0)
})
