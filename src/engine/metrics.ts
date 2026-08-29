import { EMBEDDING_MODELS } from './models.ts'
import type { EmbeddingModel } from './models.ts'

export type VectorIndexType = 'hnsw' | 'flat' | 'ivf_pq'

export interface RagSimulationInputs {
  corpusTokenCount: number
  chunkSize: number
  chunkOverlapPct: number
  embeddingModelId: string
  vectorIndexType: VectorIndexType
  topKRetrieval?: number
}

export interface RagSimulationResult {
  totalChunks: number
  totalTokensEmbedded: number
  embeddingCostUsd: number
  totalIndexMegabytes: number
  estimatedPrecisionPct: number
  estimatedRecallPct: number
  model: EmbeddingModel
}

/**
 * Calculates vector index footprint, chunk counts, and retrieval trade-offs.
 */
export function calculateRagMetrics(inputs: RagSimulationInputs): RagSimulationResult {
  const model = EMBEDDING_MODELS[inputs.embeddingModelId] || EMBEDDING_MODELS['text-embedding-3-small']
  const stepSize = Math.max(1, Math.round(inputs.chunkSize * (1 - inputs.chunkOverlapPct / 100)))
  const totalChunks = Math.ceil(inputs.corpusTokenCount / stepSize)
  const totalTokensEmbedded = totalChunks * inputs.chunkSize
  const embeddingCostUsd = Number(((totalTokensEmbedded / 1_000_000) * model.costPerMillionTokens).toFixed(4))

  // Float32 = 4 bytes per dimension
  const rawVectorBytes = totalChunks * model.dimensions * 4
  const indexMultiplier = inputs.vectorIndexType === 'hnsw' ? 1.5 : inputs.vectorIndexType === 'ivf_pq' ? 0.3 : 1.0
  const totalIndexMegabytes = Number(((rawVectorBytes * indexMultiplier) / (1024 * 1024)).toFixed(2))

  // Precision vs Recall trade-off curve estimation
  let estimatedPrecision = 90 - (inputs.chunkSize / 1024) * 18
  let estimatedRecall = 75 + (inputs.chunkSize / 1024) * 20
  if (inputs.chunkOverlapPct >= 20) {
    estimatedRecall += 4
  }

  return {
    totalChunks,
    totalTokensEmbedded,
    embeddingCostUsd,
    totalIndexMegabytes,
    estimatedPrecisionPct: Math.min(98, Math.max(50, Math.round(estimatedPrecision))),
    estimatedRecallPct: Math.min(98, Math.max(50, Math.round(estimatedRecall))),
    model,
  }
}
