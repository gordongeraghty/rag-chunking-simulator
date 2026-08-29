export interface EmbeddingModel {
  id: string
  name: string
  provider: string
  dimensions: number
  costPerMillionTokens: number
  maxContextTokens: number
}

export const EMBEDDING_MODELS: Record<string, EmbeddingModel> = {
  'text-embedding-3-small': {
    id: 'text-embedding-3-small',
    name: 'OpenAI text-embedding-3-small',
    provider: 'OpenAI',
    dimensions: 1536,
    costPerMillionTokens: 0.02,
    maxContextTokens: 8191,
  },
  'text-embedding-3-large': {
    id: 'text-embedding-3-large',
    name: 'OpenAI text-embedding-3-large',
    provider: 'OpenAI',
    dimensions: 3072,
    costPerMillionTokens: 0.13,
    maxContextTokens: 8191,
  },
  'cohere-embed-v3': {
    id: 'cohere-embed-v3',
    name: 'Cohere Embed English v3.0',
    provider: 'Cohere',
    dimensions: 1024,
    costPerMillionTokens: 0.1,
    maxContextTokens: 512,
  },
  'bge-large-en': {
    id: 'bge-large-en',
    name: 'BAAI BGE-Large-EN-v1.5',
    provider: 'Open Source',
    dimensions: 1024,
    costPerMillionTokens: 0.005,
    maxContextTokens: 512,
  },
}
