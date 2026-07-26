import type { FeatureExtractionPipeline } from "@huggingface/transformers";
import { DEFAULT_DIMENSIONS, DEFAULT_DOCUMENT_PREFIX, DEFAULT_MODEL, DEFAULT_POOLING, DEFAULT_QUERY_PREFIX } from "./config.js";

export interface EmbeddingProvider {
  readonly name: string;
  readonly model: string;
  readonly dimensions: number;
  readonly pooling?: "mean" | "cls";
  readonly layerNorm?: boolean;
  readonly documentPrefix?: string;
  readonly queryPrefix?: string;
  embed(texts: string[], onProgress?: (completed: number, total: number) => void): Promise<Float32Array[]>;
  embedQuery?(query: string): Promise<Float32Array>;
}

export interface TransformersEmbeddingOptions {
  model?: string;
  dimensions?: number;
  cacheDirectory?: string;
  dtype?: "auto" | "fp32" | "fp16" | "q8" | "q4" | "q4f16";
  pooling?: "mean" | "cls";
  layerNorm?: boolean;
  documentPrefix?: string;
  queryPrefix?: string;
  batchSize?: number;
  sortByLength?: boolean;
}

interface IndexedText {
  index: number;
  text: string;
}

export function createEmbeddingBatches(texts: string[], batchSize: number, sortByLength = true): IndexedText[][] {
  const indexed = texts.map((text, index) => ({ index, text }));
  if (sortByLength) indexed.sort((left, right) => left.text.length - right.text.length || left.index - right.index);
  const batches: IndexedText[][] = [];
  for (let start = 0; start < indexed.length; start += batchSize) batches.push(indexed.slice(start, start + batchSize));
  return batches;
}

export class TransformersEmbeddingProvider implements EmbeddingProvider {
  readonly name = "transformers.js";
  readonly model: string;
  readonly dimensions: number;
  readonly pooling: "mean" | "cls";
  readonly layerNorm: boolean;
  readonly documentPrefix: string;
  readonly queryPrefix: string;
  readonly batchSize: number;
  private readonly options: TransformersEmbeddingOptions;
  private pipeline?: Promise<FeatureExtractionPipeline>;
  private layerNormOperation?: typeof import("@huggingface/transformers").layer_norm;
  private inferenceQueue: Promise<void> = Promise.resolve();

  constructor(options: TransformersEmbeddingOptions = {}) {
    this.options = options;
    this.model = options.model ?? DEFAULT_MODEL;
    this.dimensions = options.dimensions ?? DEFAULT_DIMENSIONS;
    this.pooling = options.pooling ?? DEFAULT_POOLING;
    this.layerNorm = options.layerNorm ?? this.model === DEFAULT_MODEL;
    this.documentPrefix = options.documentPrefix ?? DEFAULT_DOCUMENT_PREFIX;
    this.queryPrefix = options.queryPrefix ?? DEFAULT_QUERY_PREFIX;
    this.batchSize = Math.max(1, Math.floor(options.batchSize ?? 32));
  }

  private async load(): Promise<FeatureExtractionPipeline> {
    if (this.pipeline) return this.pipeline;
    this.pipeline = import("@huggingface/transformers").then(async (transformers) => {
      if (this.options.cacheDirectory) transformers.env.cacheDir = this.options.cacheDirectory;
      this.layerNormOperation = transformers.layer_norm;
      return transformers.pipeline("feature-extraction", this.model, { dtype: this.options.dtype ?? "q8" });
    });
    return this.pipeline;
  }

  private async embedPrefixed(texts: string[], prefix: string, onProgress?: (completed: number, total: number) => void): Promise<Float32Array[]> {
    if (!texts.length) return [];
    const run = async () => {
      const extractor = await this.load();
      const vectors = new Array<Float32Array>(texts.length);
      let completed = 0;
      const prefixedTexts = texts.map((text) => `${prefix}${text}`);
      for (const batch of createEmbeddingBatches(prefixedTexts, this.batchSize, this.options.sortByLength ?? true)) {
        let output = await extractor(batch.map((item) => item.text), { pooling: this.pooling, normalize: false });
        const sourceDimensions = output.dims[output.dims.length - 1]!;
        if (this.dimensions > sourceDimensions) {
          throw new Error(`Embedding model ${this.model} returned ${sourceDimensions} dimensions; expected at least ${this.dimensions}`);
        }
        if (this.layerNorm) output = this.layerNormOperation!(output, [sourceDimensions]);
        if (this.dimensions < sourceDimensions) output = output.slice(null, [0, this.dimensions]);
        const rows = output.normalize(2, -1).tolist() as number[][];
        for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
          const row = rows[rowIndex]!;
          vectors[batch[rowIndex]!.index] = Float32Array.from(row);
        }
        completed += batch.length;
        onProgress?.(completed, texts.length);
      }
      return vectors;
    };
    const pending = this.inferenceQueue.then(run, run);
    this.inferenceQueue = pending.then(() => undefined, () => undefined);
    return pending;
  }

  async embed(texts: string[], onProgress?: (completed: number, total: number) => void): Promise<Float32Array[]> {
    return this.embedPrefixed(texts, this.documentPrefix, onProgress);
  }

  async embedQuery(query: string): Promise<Float32Array> {
    const [embedding] = await this.embedPrefixed([query], this.queryPrefix);
    return embedding!;
  }
}

export class HashEmbeddingProvider implements EmbeddingProvider {
  readonly name = "deterministic-hash";
  readonly model = "deterministic-hash-v1";
  readonly pooling = "mean" as const;
  readonly queryPrefix = "";

  constructor(readonly dimensions = 64) {}

  async embed(texts: string[], onProgress?: (completed: number, total: number) => void): Promise<Float32Array[]> {
    const vectors = texts.map((value) => {
      const vector = new Float32Array(this.dimensions);
      const tokens = value.toLowerCase().match(/[a-z0-9_]+/g) ?? [];
      for (const token of tokens) {
        let hash = 2166136261;
        for (let index = 0; index < token.length; index += 1) hash = Math.imul(hash ^ token.charCodeAt(index), 16777619);
        vector[Math.abs(hash) % this.dimensions]! += 1;
      }
      const norm = Math.sqrt(vector.reduce((total, item) => total + item * item, 0)) || 1;
      return vector.map((item) => item / norm);
    });
    onProgress?.(texts.length, texts.length);
    return vectors;
  }

  async embedQuery(query: string): Promise<Float32Array> {
    const [embedding] = await this.embed([query]);
    return embedding!;
  }
}