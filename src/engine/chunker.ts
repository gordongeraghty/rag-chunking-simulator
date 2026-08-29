export interface SimulatedChunk {
  id: number
  text: string
  overlapStart: string
  body: string
  estimatedTokens: number
}

/**
 * Splits raw text into overlapping token windows for simulation and preview.
 */
export function simulateTextChunking(
  text: string,
  chunkSize: number,
  chunkOverlapPct: number,
  maxPreviewChunks: number = 10
): SimulatedChunk[] {
  const words = text.trim().split(/\s+/)
  if (words.length === 0 || !text) return []

  // Approximate words to tokens ratio ~ 0.75 words per token (or ~1.3 tokens per word)
  const wordsPerChunk = Math.max(4, Math.round(chunkSize * 0.75))
  const overlapWords = Math.round(wordsPerChunk * (chunkOverlapPct / 100))
  const chunks: SimulatedChunk[] = []

  let i = 0
  let chunkId = 1

  while (i < words.length && chunkId <= maxPreviewChunks) {
    const slice = words.slice(i, i + wordsPerChunk)
    const chunkText = slice.join(' ')
    const overlapText = i > 0 && overlapWords > 0 ? words.slice(i, i + overlapWords).join(' ') : ''

    chunks.push({
      id: chunkId,
      text: chunkText,
      overlapStart: overlapText,
      body: chunkText,
      estimatedTokens: Math.round(slice.length / 0.75),
    })

    const step = Math.max(1, wordsPerChunk - overlapWords)
    i += step
    chunkId++
  }

  return chunks
}
