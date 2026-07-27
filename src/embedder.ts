import os from "node:os";
import type { FeatureExtractionPipeline } from "@huggingface/transformers";
import { DEFAULT_DEVICE, DEFAULT_DIMENSIONS, DEFAULT_DOCUMENT_PREFIX, DEFAULT_MAX_EMBEDDING_CHARACTERS, DEFAULT_MODEL, DEFAULT_POOLING, DEFAULT_QUERY_PREFIX } from "./config.js";

export type EmbeddingDevice = import("@huggingface/transformers").DeviceType;

export interface EmbeddingProvider {
  readonly name: string;
  readonly model: string;
  readonly dimensions: number;
  readonly device?: EmbeddingDevice;
  readonly pooling?: "mean" | "cls";
  readonly layerNorm?: boolean;
  readonly documentPrefix?: string;
  readonly queryPrefix?: string;
  readonly maxEmbeddingCharacters?: number;
  readonly activeDevice?: EmbeddingDevice;
  embed(texts: string[], onProgress?: (completed: number, total: number) => void): Promise<Float32Array[]>;
  embedBatched?(texts: string[], onBatch: (batch: EmbeddingBatch) => void | Promise<void>, onProgress?: (completed: number, total: number) => void): Promise<void>;
  embedQuery?(query: string): Promise<Float32Array>;
}

export interface EmbeddingBatch {
  indexes: number[];
  vectors: Float32Array[];
  completed: number;
  total: number;
}

export interface TransformersEmbeddingOptions {
  model?: string;
  dimensions?: number;
  cacheDirectory?: string;
  device?: EmbeddingDevice;
  dtype?: "auto" | "fp32" | "fp16" | "q8" | "q4" | "q4f16";
  pooling?: "mean" | "cls";
  layerNorm?: boolean;
  documentPrefix?: string;
  queryPrefix?: string;
  batchSize?: number;
  maxBatchTokens?: number;
  maxEmbeddingCharacters?: number;
  sortByLength?: boolean;
  /** ONNX Runtime intra-op thread count for CPU inference. Defaults to the host's logical core count; override on shared or constrained hosts. */
  threads?: number;
}

interface IndexedText {
  index: number;
  text: string;
}

function isOutOfMemoryError(error: unknown): boolean {
  const message = error instanceof Error ? `${error.message} ${error.cause instanceof Error ? error.cause.message : ""}` : String(error);
  return /out of memory|not enough memory resources|8007000e|resource[s]? are available to complete/i.test(message);
}

function isDeviceLostError(error: unknown): boolean {
  const message = error instanceof Error ? `${error.message} ${error.cause instanceof Error ? error.cause.message : ""}` : String(error);
  return /device instance has been suspended|device has been removed|will not respond to more commands|887a0005|887a0006/i.test(message);
}

function estimateTokenCount(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

export function truncateEmbeddingText(text: string, maxCharacters: number): string {
  if (!Number.isFinite(maxCharacters) || maxCharacters <= 0 || text.length <= maxCharacters) return text;
  return text.slice(0, Math.floor(maxCharacters));
}

export function createEmbeddingBatches(texts: string[], batchSize: number, sortByLength = true, maxBatchTokens = Number.POSITIVE_INFINITY): IndexedText[][] {
  const indexed = texts.map((text, index) => ({ index, text }));
  if (sortByLength) indexed.sort((left, right) => left.text.length - right.text.length || left.index - right.index);
  const batches: IndexedText[][] = [];
  let current: IndexedText[] = [];
  let currentTokens = 0;
  for (const item of indexed) {
    const itemTokens = estimateTokenCount(item.text);
    if (current.length && (current.length >= batchSize || currentTokens + itemTokens > maxBatchTokens)) {
      batches.push(current);
      current = [];
      currentTokens = 0;
    }
    current.push(item);
    currentTokens += itemTokens;
  }
  if (current.length) batches.push(current);
  return batches;
}

export class TransformersEmbeddingProvider implements EmbeddingProvider {
  readonly name = "transformers.js";
  readonly model: string;
  readonly dimensions: number;
  readonly device: EmbeddingDevice;
  readonly pooling: "mean" | "cls";
  readonly layerNorm: boolean;
  readonly documentPrefix: string;
  readonly queryPrefix: string;
  readonly batchSize: number;
  readonly maxBatchTokens: number;
  readonly maxEmbeddingCharacters: number;
  readonly threads: number;
  private readonly options: TransformersEmbeddingOptions;
  private pipeline?: Promise<FeatureExtractionPipeline>;
  private layerNormOperation?: typeof import("@huggingface/transformers").layer_norm;
  // Two lanes so a query embedded during a large bulk indexing pass only waits for the in-flight batch to finish, not the whole pass: bulk work is scheduled one batch at a time, and the drain loop always prefers a queued query over the next bulk batch.
  private readonly bulkTaskQueue: Array<() => Promise<void>> = [];
  private readonly priorityTaskQueue: Array<() => Promise<void>> = [];
  private draining = false;
  private runtimeDevice: EmbeddingDevice;

  constructor(options: TransformersEmbeddingOptions = {}) {
    this.options = options;
    this.model = options.model ?? DEFAULT_MODEL;
    this.dimensions = options.dimensions ?? DEFAULT_DIMENSIONS;
    this.device = options.device ?? DEFAULT_DEVICE;
    this.pooling = options.pooling ?? DEFAULT_POOLING;
    this.layerNorm = options.layerNorm ?? this.model === "nomic-ai/nomic-embed-text-v1.5";
    this.documentPrefix = options.documentPrefix ?? DEFAULT_DOCUMENT_PREFIX;
    this.queryPrefix = options.queryPrefix ?? DEFAULT_QUERY_PREFIX;
    this.batchSize = Math.max(1, Math.floor(options.batchSize ?? (this.device === "dml" ? 8 : 32)));
    const maxBatchTokens = options.maxBatchTokens ?? (this.device === "dml" ? 4096 : Number.POSITIVE_INFINITY);
    this.maxBatchTokens = Number.isFinite(maxBatchTokens) && maxBatchTokens > 0 ? Math.floor(maxBatchTokens) : Number.POSITIVE_INFINITY;
    const maxEmbeddingCharacters = options.maxEmbeddingCharacters ?? DEFAULT_MAX_EMBEDDING_CHARACTERS;
    this.maxEmbeddingCharacters = Number.isFinite(maxEmbeddingCharacters) && maxEmbeddingCharacters > 0 ? Math.floor(maxEmbeddingCharacters) : Number.POSITIVE_INFINITY;
    this.threads = Math.max(1, Math.floor(options.threads ?? os.cpus().length));
    this.runtimeDevice = this.device;
  }

  get activeDevice(): EmbeddingDevice {
    return this.runtimeDevice;
  }

  private async load(): Promise<FeatureExtractionPipeline> {
    if (this.pipeline) return this.pipeline;
    this.pipeline = import("@huggingface/transformers").then(async (transformers) => {
      if (this.options.cacheDirectory) transformers.env.cacheDir = this.options.cacheDirectory;
      this.layerNormOperation = transformers.layer_norm;
      // onnxruntime-node's default intra-op thread count isn't always the full logical core count; on a low-core machine that leaves real CPU throughput on the table for a compute-bound embedding pass. Configurable (this.threads) since host core counts vary.
      const sessionOptions = this.runtimeDevice === "cpu" ? { intraOpNumThreads: this.threads } : undefined;
      return transformers.pipeline("feature-extraction", this.model, { device: this.runtimeDevice, dtype: this.options.dtype ?? "q8", session_options: sessionOptions });
    });
    return this.pipeline;
  }

  private async useCpuFallback(): Promise<void> {
    if (this.runtimeDevice === "cpu") return;
    const currentPipeline = this.pipeline;
    this.pipeline = undefined;
    this.runtimeDevice = "cpu";
    if (currentPipeline) {
      try {
        await (await currentPipeline).dispose();
      } catch {
        // The native provider can be unavailable while its session is being released.
      }
    }
  }

  private schedule(priority: boolean, task: () => Promise<void>): Promise<void> {
    return new Promise((resolve, reject) => {
      (priority ? this.priorityTaskQueue : this.bulkTaskQueue).push(async () => {
        try {
          await task();
          resolve();
        } catch (error) {
          reject(error);
        }
      });
      void this.drain();
    });
  }

  private async drain(): Promise<void> {
    if (this.draining) return;
    this.draining = true;
    try {
      while (this.priorityTaskQueue.length || this.bulkTaskQueue.length) {
        const next = this.priorityTaskQueue.shift() ?? this.bulkTaskQueue.shift()!;
        await next();
      }
    } finally {
      this.draining = false;
    }
  }

  private async embedPrefixed(texts: string[], prefix: string, onProgress?: (completed: number, total: number) => void, onBatch?: (batch: EmbeddingBatch) => void | Promise<void>, priority = false): Promise<Float32Array[]> {
    if (!texts.length) return [];
    const vectors = onBatch ? undefined : new Array<Float32Array>(texts.length);
    let completed = 0;
    const prefixedTexts = texts.map((text) => truncateEmbeddingText(`${prefix}${text}`, this.maxEmbeddingCharacters));
    const processBatch = async (batch: IndexedText[]): Promise<void> => {
      const extractor = await this.load();
      try {
        let output = await extractor(batch.map((item) => item.text), { pooling: this.pooling, normalize: false });
        const sourceDimensions = output.dims[output.dims.length - 1]!;
        if (this.dimensions > sourceDimensions) {
          throw new Error(`Embedding model ${this.model} returned ${sourceDimensions} dimensions; expected at least ${this.dimensions}`);
        }
        if (this.layerNorm) output = this.layerNormOperation!(output, [sourceDimensions]);
        if (this.dimensions < sourceDimensions) output = output.slice(null, [0, this.dimensions]);
        const normalized = output.normalize(2, -1);
        const outputDimensions = normalized.dims[normalized.dims.length - 1]!;
        const data = normalized.data as Float32Array;
        if (data.length !== batch.length * outputDimensions) throw new Error(`Embedding model ${this.model} returned an unexpected tensor shape`);
        const batchVectors = new Array<Float32Array>(batch.length);
        for (let rowIndex = 0; rowIndex < batch.length; rowIndex += 1) {
          const start = rowIndex * outputDimensions;
          batchVectors[rowIndex] = Float32Array.from(data.subarray(start, start + outputDimensions));
          if (vectors) vectors[batch[rowIndex]!.index] = batchVectors[rowIndex]!;
        }
        completed += batch.length;
        await onBatch?.({ indexes: batch.map((item) => item.index), vectors: batchVectors, completed, total: texts.length });
        onProgress?.(completed, texts.length);
      } catch (error) {
        if (isDeviceLostError(error) && this.runtimeDevice !== "cpu") {
          await this.useCpuFallback();
          await processBatch(batch);
          return;
        }
        if (!isOutOfMemoryError(error) || batch.length <= 1) throw error;
        const midpoint = Math.ceil(batch.length / 2);
        await processBatch(batch.slice(0, midpoint));
        await processBatch(batch.slice(midpoint));
      }
    };
    for (const batch of createEmbeddingBatches(prefixedTexts, this.batchSize, this.options.sortByLength ?? true, this.maxBatchTokens)) {
      await this.schedule(priority, () => processBatch(batch));
    }
    return vectors ?? [];
  }

  async embed(texts: string[], onProgress?: (completed: number, total: number) => void): Promise<Float32Array[]> {
    return this.embedPrefixed(texts, this.documentPrefix, onProgress);
  }

  async embedBatched(texts: string[], onBatch: (batch: EmbeddingBatch) => void | Promise<void>, onProgress?: (completed: number, total: number) => void): Promise<void> {
    await this.embedPrefixed(texts, this.documentPrefix, onProgress, onBatch);
  }

  async embedQuery(query: string): Promise<Float32Array> {
    const [embedding] = await this.embedPrefixed([query], this.queryPrefix, undefined, undefined, true);
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

  async embedBatched(texts: string[], onBatch: (batch: EmbeddingBatch) => void | Promise<void>, onProgress?: (completed: number, total: number) => void): Promise<void> {
    const vectors = await this.embed(texts, onProgress);
    await onBatch({ indexes: vectors.map((_, index) => index), vectors, completed: texts.length, total: texts.length });
  }

  async embedQuery(query: string): Promise<Float32Array> {
    const [embedding] = await this.embed([query]);
    return embedding!;
  }
}