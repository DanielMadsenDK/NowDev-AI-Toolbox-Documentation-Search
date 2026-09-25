import type { FeatureExtractionPipeline } from "@huggingface/transformers";

export interface InferenceSettings {
  model: string;
  pooling: "mean" | "cls";
  layerNorm: boolean;
  dimensions: number;
}

/** Settings a pooled embedding worker loads its model with; see embedding-worker.ts. */
export interface EmbeddingWorkerData extends InferenceSettings {
  dtype: string;
  cacheDirectory?: string;
  threads: number;
}

export interface EmbeddingWorkerRequest {
  id: number;
  texts: string[];
}

export type EmbeddingWorkerResponse = { ready: true } | { id: number; data: Float32Array } | { id: number; error: string };

type LayerNorm = typeof import("@huggingface/transformers").layer_norm;

/**
 * Runs one batch through the model and returns its L2-normalized vectors as one row-major array. Shared by the
 * in-process pipeline and the pooled workers so both produce identical vectors for the same batch.
 */
export async function extractEmbeddings(extractor: FeatureExtractionPipeline, layerNorm: LayerNorm, texts: string[], settings: InferenceSettings): Promise<Float32Array> {
  let output = await extractor(texts, { pooling: settings.pooling, normalize: false });
  const sourceDimensions = output.dims[output.dims.length - 1]!;
  if (settings.dimensions > sourceDimensions) {
    throw new Error(`Embedding model ${settings.model} returned ${sourceDimensions} dimensions; expected at least ${settings.dimensions}`);
  }
  if (settings.layerNorm) output = layerNorm(output, [sourceDimensions]);
  if (settings.dimensions < sourceDimensions) output = output.slice(null, [0, settings.dimensions]);
  const data = output.normalize(2, -1).data as Float32Array;
  if (data.length !== texts.length * settings.dimensions) throw new Error(`Embedding model ${settings.model} returned an unexpected tensor shape`);
  return data;
}

export function splitVectors(data: Float32Array, count: number, dimensions: number): Float32Array[] {
  if (data.length !== count * dimensions) throw new Error(`Expected ${count * dimensions} embedding values, received ${data.length}`);
  return Array.from({ length: count }, (_, row) => data.slice(row * dimensions, (row + 1) * dimensions));
}
