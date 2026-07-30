import os from "node:os";
import type { FeatureExtractionPipeline } from "@huggingface/transformers";
import { DEFAULT_DEVICE, DEFAULT_DIMENSIONS, DEFAULT_DOCUMENT_PREFIX, DEFAULT_MAX_EMBEDDING_CHARACTERS, DEFAULT_MODEL, DEFAULT_POOLING, DEFAULT_QUERY_PREFIX } from "./config.js";

export type EmbeddingDevice = import("@huggingface/transformers").DeviceType;

export interface EmbeddingProvider {
  readonly name: string;
  readonly model: string;
  readonly dimensions: number;
  readonly device?: EmbeddingDevice;
  readonly dtype?: string;
  readonly pooling?: "mean" | "cls";
  readonly layerNorm?: boolean;
  readonly documentPrefix?: string;
  readonly queryPrefix?: string;
  readonly maxEmbeddingCharacters?: number;
  readonly activeDevice?: EmbeddingDevice;
  readonly endpointModel?: string;
  readonly endpointConcurrency?: number;
  readonly metrics?: EmbeddingMetrics;
  embed(texts: string[], onProgress?: (completed: number, total: number) => void): Promise<Float32Array[]>;
  embedBatched?(texts: string[], onBatch: (batch: EmbeddingBatch) => void | Promise<void>, onProgress?: (completed: number, total: number) => void): Promise<void>;
  embedQuery?(query: string): Promise<Float32Array>;
}

export interface EmbeddingMetrics {
  requests: number;
  retries: number;
  truncations: number;
  batchSplits: number;
  deviceTransitions: number;
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

export interface EndpointEmbeddingOptions {
  endpoint: string;
  apiKey: string;
  model: string;
  batchSize?: number;
  concurrency?: number;
  timeoutMilliseconds?: number;
  fetch?: typeof globalThis.fetch;
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
  return Math.max(1, Math.ceil(Buffer.byteLength(text, "utf8") / 2));
}

function assertNonEmptyTexts(texts: string[]): void {
  const emptyIndex = texts.findIndex((text) => !text.trim());
  if (emptyIndex !== -1) throw new Error(`Embedding input at index ${emptyIndex} is empty`);
}

export function truncateEmbeddingText(text: string, maxCharacters: number): string {
  if (!Number.isFinite(maxCharacters) || maxCharacters <= 0 || text.length <= maxCharacters) return text;
  let end = Math.floor(maxCharacters);
  if (end > 0 && /[\uD800-\uDBFF]/.test(text[end - 1]!)) end -= 1;
  return text.slice(0, end);
}

export function createEmbeddingBatches(texts: string[], batchSize: number, sortByLength = true, maxBatchTokens = Number.POSITIVE_INFINITY): IndexedText[][] {
  const indexed = texts.map((text, index) => ({ index, text }));
  if (sortByLength) indexed.sort((left, right) => left.text.length - right.text.length || left.index - right.index);
  const batches: IndexedText[][] = [];
  let current: IndexedText[] = [];
  let currentTokens = 0;
  for (const item of indexed) {
    const itemTokens = estimateTokenCount(item.text);
    if (itemTokens > maxBatchTokens) {
      throw new Error(`Embedding input at index ${item.index} exceeds the configured batch token budget (${itemTokens} > ${maxBatchTokens})`);
    }
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
  readonly dtype: NonNullable<TransformersEmbeddingOptions["dtype"]>;
  readonly pooling: "mean" | "cls";
  readonly layerNorm: boolean;
  readonly documentPrefix: string;
  readonly queryPrefix: string;
  readonly batchSize: number;
  readonly maxBatchTokens: number;
  readonly maxEmbeddingCharacters: number;
  readonly threads: number;
  readonly metrics: EmbeddingMetrics = { requests: 0, retries: 0, truncations: 0, batchSplits: 0, deviceTransitions: 0 };
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
    this.dtype = options.dtype ?? "q8";
    this.pooling = options.pooling ?? DEFAULT_POOLING;
    this.layerNorm = options.layerNorm ?? this.model === "nomic-ai/nomic-embed-text-v1.5";
    this.documentPrefix = options.documentPrefix ?? DEFAULT_DOCUMENT_PREFIX;
    this.queryPrefix = options.queryPrefix ?? DEFAULT_QUERY_PREFIX;
    this.batchSize = Math.max(1, Math.floor(options.batchSize ?? (this.device === "dml" ? 8 : 32)));
    const maxBatchTokens = options.maxBatchTokens ?? (this.device === "dml" ? 4096 : Number.POSITIVE_INFINITY);
    this.maxBatchTokens = Number.isFinite(maxBatchTokens) && maxBatchTokens > 0 ? Math.floor(maxBatchTokens) : Number.POSITIVE_INFINITY;
    const maxEmbeddingCharacters = options.maxEmbeddingCharacters ?? DEFAULT_MAX_EMBEDDING_CHARACTERS;
    this.maxEmbeddingCharacters = Number.isFinite(maxEmbeddingCharacters) && maxEmbeddingCharacters > 0 ? Math.floor(maxEmbeddingCharacters) : Number.POSITIVE_INFINITY;
    if (Number.isFinite(this.maxEmbeddingCharacters) && this.maxEmbeddingCharacters <= Math.max(this.documentPrefix.length, this.queryPrefix.length)) {
      throw new Error("maxEmbeddingCharacters must exceed the configured embedding prefix length");
    }
    this.threads = Math.max(1, Math.floor(options.threads ?? os.availableParallelism()));
    this.runtimeDevice = this.device;
  }

  get activeDevice(): EmbeddingDevice {
    return this.runtimeDevice;
  }

  private async load(): Promise<FeatureExtractionPipeline> {
    if (!this.pipeline) {
      const loading = import("@huggingface/transformers").then(async (transformers) => {
        if (this.options.cacheDirectory) transformers.env.cacheDir = this.options.cacheDirectory;
        this.layerNormOperation = transformers.layer_norm;
        // onnxruntime-node's default intra-op thread count isn't always the full logical core count; on a low-core machine that leaves real CPU throughput on the table for a compute-bound embedding pass. Configurable (this.threads) since host core counts vary.
        const sessionOptions = this.runtimeDevice === "cpu" ? { intraOpNumThreads: this.threads } : undefined;
        return transformers.pipeline("feature-extraction", this.model, { device: this.runtimeDevice, dtype: this.dtype, session_options: sessionOptions });
      });
      this.pipeline = loading;
      void loading.catch(() => {
        if (this.pipeline === loading) this.pipeline = undefined;
      });
    }
    return this.pipeline;
  }

  private async useCpuFallback(): Promise<void> {
    if (this.runtimeDevice === "cpu") return;
    const currentPipeline = this.pipeline;
    this.pipeline = undefined;
    this.runtimeDevice = "cpu";
    this.metrics.deviceTransitions += 1;
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
    assertNonEmptyTexts(texts);
    const vectors = onBatch ? undefined : new Array<Float32Array>(texts.length);
    let completed = 0;
    const prefixedTexts = texts.map((text) => {
      const value = `${prefix}${text}`;
      const truncated = truncateEmbeddingText(value, this.maxEmbeddingCharacters);
      if (truncated.length < value.length) this.metrics.truncations += 1;
      return truncated;
    });
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
        if (isOutOfMemoryError(error) && batch.length === 1 && this.runtimeDevice !== "cpu") {
          await this.useCpuFallback();
          await processBatch(batch);
          return;
        }
        if (!isOutOfMemoryError(error) || batch.length <= 1) throw error;
        this.metrics.batchSplits += 1;
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

interface EndpointEmbeddingResponse {
  data?: Array<{ embedding?: number[]; index?: number }>;
  error?: { message?: string };
}

class EndpointEmbeddingError extends Error {
  constructor(readonly status: number, message: string, readonly retryAfterMilliseconds?: number) {
    super(message);
  }
}

function isTransientEndpointError(error: unknown): boolean {
  return error instanceof EndpointEmbeddingError && (error.status === 0 || error.status === 408 || error.status === 429 || error.status >= 500);
}

function retryDelay(error: unknown, attempt: number): number {
  if (error instanceof EndpointEmbeddingError && error.retryAfterMilliseconds !== undefined) return error.retryAfterMilliseconds;
  return 1000 * 2 ** attempt;
}

function isEndpointContextLimitError(error: unknown): error is EndpointEmbeddingError {
  return error instanceof EndpointEmbeddingError
    && error.status === 400
    && /context length|maximum input length|input_tokens/i.test(error.message);
}

export class EndpointDocumentEmbeddingProvider implements EmbeddingProvider {
  readonly name: string;
  readonly model: string;
  readonly dimensions: number;
  readonly device?: EmbeddingDevice;
  readonly dtype?: string;
  readonly pooling?: "mean" | "cls";
  readonly layerNorm?: boolean;
  readonly documentPrefix: string;
  readonly queryPrefix?: string;
  readonly maxEmbeddingCharacters?: number;
  readonly endpointModel: string;
  readonly batchSize: number;
  readonly endpointConcurrency: number;
  readonly timeoutMilliseconds: number;
  readonly endpoint: string;
  readonly metrics: EmbeddingMetrics = { requests: 0, retries: 0, truncations: 0, batchSplits: 0, deviceTransitions: 0 };
  private readonly apiKey: string;
  private readonly fetchImplementation: typeof globalThis.fetch;

  constructor(private readonly queryProvider: EmbeddingProvider, options: EndpointEmbeddingOptions) {
    const endpoint = new URL(options.endpoint);
    if (endpoint.username || endpoint.password || endpoint.search || endpoint.hash) {
      throw new Error("Embedding endpoint must not contain credentials, query parameters, or a fragment");
    }
    if (endpoint.protocol !== "https:" && !(endpoint.protocol === "http:" && ["localhost", "127.0.0.1", "::1"].includes(endpoint.hostname))) {
      throw new Error("Embedding endpoint must use HTTPS unless it targets localhost");
    }
    if (!options.apiKey.trim()) throw new Error("Embedding endpoint API key is empty");
    if (!options.model.trim()) throw new Error("Embedding endpoint model is empty");
    this.name = queryProvider.name;
    this.model = queryProvider.model;
    this.dimensions = queryProvider.dimensions;
    this.device = queryProvider.device;
    this.dtype = queryProvider.dtype;
    this.pooling = queryProvider.pooling;
    this.layerNorm = queryProvider.layerNorm;
    this.documentPrefix = queryProvider.documentPrefix ?? "";
    this.queryPrefix = queryProvider.queryPrefix;
    this.maxEmbeddingCharacters = queryProvider.maxEmbeddingCharacters;
    this.endpoint = endpoint.toString();
    this.apiKey = options.apiKey;
    this.endpointModel = options.model;
    this.batchSize = Math.max(1, Math.floor(options.batchSize ?? 64));
    this.endpointConcurrency = Math.max(1, Math.floor(options.concurrency ?? 4));
    this.timeoutMilliseconds = Math.max(1000, Math.floor(options.timeoutMilliseconds ?? 30_000));
    this.fetchImplementation = options.fetch ?? globalThis.fetch;
    if (Number.isFinite(this.maxEmbeddingCharacters) && this.maxEmbeddingCharacters! <= this.documentPrefix.length) {
      throw new Error("maxEmbeddingCharacters must exceed the configured document prefix length");
    }
  }

  get activeDevice(): EmbeddingDevice | undefined {
    return this.queryProvider.activeDevice;
  }

  private async request(texts: string[]): Promise<Float32Array[]> {
    this.metrics.requests += 1;
    let response: Response;
    try {
      response = await this.fetchImplementation(this.endpoint, {
        method: "POST",
        headers: {
          authorization: `Bearer ${this.apiKey}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: this.endpointModel,
          input: texts.map((text) => truncateEmbeddingText(`${this.documentPrefix}${text}`, this.maxEmbeddingCharacters ?? Number.POSITIVE_INFINITY)),
          encoding_format: "float",
        }),
        signal: AbortSignal.timeout(this.timeoutMilliseconds),
      });
    } catch (error) {
      throw new EndpointEmbeddingError(0, `Embedding endpoint request failed: ${error instanceof Error ? error.message : String(error)}`);
    }
    let payload: EndpointEmbeddingResponse;
    try {
      payload = await response.json() as EndpointEmbeddingResponse;
    } catch {
      throw new Error(`Embedding endpoint returned HTTP ${response.status} with an invalid JSON response`);
    }
    if (!response.ok) {
      const retryAfter = response.headers.get("retry-after");
      const retryAfterMilliseconds = retryAfter && /^\d+(?:\.\d+)?$/.test(retryAfter) ? Number(retryAfter) * 1000 : undefined;
      throw new EndpointEmbeddingError(response.status, `Embedding endpoint returned HTTP ${response.status}${payload.error?.message ? `: ${payload.error.message}` : ""}`, retryAfterMilliseconds);
    }
    if (!Array.isArray(payload.data) || payload.data.length !== texts.length) {
      throw new Error(`Embedding endpoint returned ${payload.data?.length ?? 0} vectors for ${texts.length} inputs`);
    }
    const indexedCount = payload.data.filter((item) => item.index !== undefined).length;
    if (indexedCount !== 0 && indexedCount !== payload.data.length) throw new Error("Embedding endpoint returned a mixture of indexed and unindexed vectors");
    let ordered = payload.data;
    if (indexedCount) {
      const indexes = payload.data.map((item) => item.index!);
      if (indexes.some((index) => !Number.isInteger(index) || index < 0 || index >= texts.length) || new Set(indexes).size !== texts.length) {
        throw new Error(`Embedding endpoint returned invalid vector indexes; expected a unique permutation of 0..${texts.length - 1}`);
      }
      ordered = [...payload.data].sort((left, right) => left.index! - right.index!);
    }
    return ordered.map((item) => {
      if (!Array.isArray(item.embedding) || item.embedding.length < this.dimensions) {
        throw new Error(`Embedding endpoint model ${this.endpointModel} returned ${item.embedding?.length ?? 0} dimensions; expected at least ${this.dimensions}`);
      }
      const vector = Float32Array.from(item.embedding.slice(0, this.dimensions));
      const norm = Math.sqrt(vector.reduce((total, value) => total + value * value, 0));
      if (!Number.isFinite(norm) || norm === 0) throw new Error(`Embedding endpoint model ${this.endpointModel} returned an invalid vector`);
      return vector.map((value) => value / norm);
    });
  }

  private async requestWithTransientRetry(texts: string[]): Promise<Float32Array[]> {
    for (let attempt = 0; ; attempt += 1) {
      try {
        return await this.request(texts);
      } catch (error) {
        const maximumRetries = error instanceof EndpointEmbeddingError && error.status === 0 ? 1 : 2;
        if (attempt >= maximumRetries || !isTransientEndpointError(error)) throw error;
        this.metrics.retries += 1;
        await new Promise((resolve) => setTimeout(resolve, retryDelay(error, attempt)));
      }
    }
  }

  private async requestWithContextLimitRetry(texts: string[], reductions = 0): Promise<Float32Array[]> {
    try {
      return await this.requestWithTransientRetry(texts);
    } catch (error) {
      if (!isEndpointContextLimitError(error)) throw error;
      if (texts.length > 1) {
        this.metrics.batchSplits += 1;
        const midpoint = Math.ceil(texts.length / 2);
        const left = await this.requestWithContextLimitRetry(texts.slice(0, midpoint), reductions);
        const right = await this.requestWithContextLimitRetry(texts.slice(midpoint), reductions);
        return [...left, ...right];
      }
      if (reductions >= 8) throw new Error("Embedding endpoint still rejected an input after 8 context-limit reductions", { cause: error });
      const text = texts[0]!;
      const currentLength = this.documentPrefix.length + text.length;
      const tokenCounts = error.message.match(/passed\s+(\d+)\s+input tokens[\s\S]*?context length is only\s+(\d+)\s+tokens/i);
      const ratio = tokenCounts ? (Number(tokenCounts[2]) - 8) / Number(tokenCounts[1]) : 0.5;
      const shortenedLength = Math.floor(currentLength * Math.min(0.9, Math.max(0.1, ratio))) - this.documentPrefix.length;
      if (shortenedLength < 1 || shortenedLength >= text.length) throw error;
      this.metrics.truncations += 1;
      return this.requestWithContextLimitRetry([truncateEmbeddingText(text, shortenedLength)], reductions + 1);
    }
  }

  async embed(texts: string[], onProgress?: (completed: number, total: number) => void): Promise<Float32Array[]> {
    const vectors: Float32Array[] = [];
    await this.embedBatched(texts, (batch) => {
      batch.indexes.forEach((index, batchIndex) => { vectors[index] = batch.vectors[batchIndex]!; });
    }, onProgress);
    return vectors;
  }

  async embedBatched(texts: string[], onBatch: (batch: EmbeddingBatch) => void | Promise<void>, onProgress?: (completed: number, total: number) => void): Promise<void> {
    if (!texts.length) return;
    assertNonEmptyTexts(texts);
    if (Number.isFinite(this.maxEmbeddingCharacters)) {
      this.metrics.truncations += texts.filter((text) => this.documentPrefix.length + text.length > this.maxEmbeddingCharacters!).length;
    }
    let completed = 0;
    let nextOffset = 0;
    const worker = async (): Promise<void> => {
      while (nextOffset < texts.length) {
        const offset = nextOffset;
        nextOffset += this.batchSize;
        const input = texts.slice(offset, offset + this.batchSize);
        const vectors = await this.requestWithContextLimitRetry(input);
        completed += input.length;
        await onBatch({ indexes: input.map((_, index) => offset + index), vectors, completed, total: texts.length });
        onProgress?.(completed, texts.length);
      }
    };
    const results = await Promise.allSettled(Array.from({ length: Math.min(this.endpointConcurrency, Math.ceil(texts.length / this.batchSize)) }, worker));
    const failures = results.filter((result): result is PromiseRejectedResult => result.status === "rejected");
    if (failures.length) {
      const firstFailure = failures[0]!.reason;
      const detail = firstFailure instanceof Error ? firstFailure.message : String(firstFailure);
      throw new AggregateError(failures.map((failure) => failure.reason), `${failures.length} embedding endpoint worker${failures.length === 1 ? "" : "s"} failed: ${detail}`, { cause: firstFailure });
    }
  }

  async embedQuery(query: string): Promise<Float32Array> {
    if (this.queryProvider.embedQuery) return this.queryProvider.embedQuery(query);
    return (await this.queryProvider.embed([query]))[0]!;
  }
}

export class HashEmbeddingProvider implements EmbeddingProvider {
  readonly name = "deterministic-hash";
  readonly model = "deterministic-hash-v1";
  readonly pooling = "mean" as const;
  readonly queryPrefix = "";

  constructor(readonly dimensions = 64) {}

  async embed(texts: string[], onProgress?: (completed: number, total: number) => void): Promise<Float32Array[]> {
    assertNonEmptyTexts(texts);
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