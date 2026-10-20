import type { DeviceType, FeatureExtractionPipeline, Tensor } from "@huggingface/transformers";

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

export type EmbeddingWorkerRequest = { id: number; texts: string[] } | { shutdown: true };

export type EmbeddingWorkerResponse = { ready: true } | { id: number; data: Float32Array } | { id: number; error: string; cause?: string };

type LayerNorm = typeof import("@huggingface/transformers").layer_norm;

export interface LoadedExtractor {
  extractor: FeatureExtractionPipeline;
  layerNorm: LayerNorm;
}

/**
 * Loads a feature-extraction pipeline. Shared by the in-process sessions and the pooled workers so every session
 * resolves the same cached model files with the same options.
 */
export async function loadExtractor(model: string, options: { device: DeviceType; dtype: string; cacheDirectory?: string; threads?: number }): Promise<LoadedExtractor> {
  const transformers = await import("@huggingface/transformers");
  // env is process-global; every caller passes the same data directory's model cache.
  if (options.cacheDirectory) transformers.env.cacheDir = options.cacheDirectory;
  // onnxruntime-node's default intra-op thread count isn't always the full logical core count, so CPU sessions set it explicitly.
  const sessionOptions = options.device === "cpu" && options.threads ? { intraOpNumThreads: options.threads } : undefined;
  const extractor = await transformers.pipeline("feature-extraction", model, { device: options.device, dtype: options.dtype as "q8", session_options: sessionOptions });
  return { extractor, layerNorm: transformers.layer_norm };
}

/**
 * Runs one batch through the model and returns its L2-normalized vectors as one row-major array. Shared by the
 * in-process pipeline and the pooled workers so both produce identical vectors for the same batch.
 */
export async function extractEmbeddings(extractor: FeatureExtractionPipeline, layerNorm: LayerNorm, texts: string[], settings: InferenceSettings): Promise<Float32Array> {
  // Each step allocates a new tensor; intermediates are disposed as soon as the next one exists, which releases
  // native buffers promptly instead of waiting for garbage collection (and matters on GPU devices).
  const intermediates: Tensor[] = [];
  try {
    let output = await extractor(texts, { pooling: settings.pooling, normalize: false });
    const sourceDimensions = output.dims[output.dims.length - 1]!;
    if (settings.dimensions > sourceDimensions) {
      throw new Error(`Embedding model ${settings.model} returned ${sourceDimensions} dimensions; expected at least ${settings.dimensions}`);
    }
    if (settings.layerNorm) {
      intermediates.push(output);
      output = layerNorm(output, [sourceDimensions]);
    }
    if (settings.dimensions < sourceDimensions) {
      intermediates.push(output);
      output = output.slice(null, [0, settings.dimensions]);
    }
    intermediates.push(output);
    const data = output.normalize(2, -1).data as Float32Array;
    if (data.length !== texts.length * settings.dimensions) throw new Error(`Embedding model ${settings.model} returned an unexpected tensor shape`);
    return data;
  } finally {
    for (const tensor of intermediates) tensor.dispose();
  }
}

/** Splits a row-major batch into per-text views; the views share `data`, which the caller must own exclusively. */
export function splitVectors(data: Float32Array, count: number, dimensions: number): Float32Array[] {
  if (data.length !== count * dimensions) throw new Error(`Expected ${count * dimensions} embedding values, received ${data.length}`);
  return Array.from({ length: count }, (_, row) => data.subarray(row * dimensions, (row + 1) * dimensions));
}
