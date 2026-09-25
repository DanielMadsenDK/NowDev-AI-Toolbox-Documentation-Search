import os from "node:os";
import type { FeatureExtractionPipeline, PreTrainedTokenizer } from "@huggingface/transformers";
import { DEFAULT_DEVICE } from "./config.js";
import { extractEmbeddings, splitVectors, type EmbeddingWorkerData, type InferenceSettings } from "./embedding-inference.js";
import { EmbeddingWorkerPool } from "./embedding-pool.js";
import { DEFAULT_EMBEDDING_PROFILE, EMBEDDING_PROFILES } from "./embedding-profiles.js";

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
  /** Parallel model sessions used for bulk document embedding. */
  readonly workers?: number;
  readonly metrics?: EmbeddingMetrics;
  embed(texts: string[], onProgress?: (completed: number, total: number) => void): Promise<Float32Array[]>;
  embedBatched?(texts: string[], onBatch: (batch: EmbeddingBatch) => void | Promise<void>, onProgress?: (completed: number, total: number) => void): Promise<void>;
  embedQuery?(query: string): Promise<Float32Array>;
  /** Releases background resources such as worker threads. */
  dispose?(): Promise<void>;
}

export interface EmbeddingMetrics {
  requests: number;
  retries: number;
  truncations: number;
  batchSplits: number;
  deviceTransitions: number;
  /** Bulk passes that ran in-process because the embedding worker pool could not start. */
  workerFallbacks: number;
}

export interface EmbeddingBatch {
  indexes: number[];
  vectors: Float32Array[];
  completed: number;
  total: number;
}

export const EMBEDDING_DTYPES = ["auto", "fp32", "fp16", "q8", "q4", "q4f16"] as const;
export type EmbeddingDtype = typeof EMBEDDING_DTYPES[number];

export function isEmbeddingDtype(value: string): value is EmbeddingDtype {
  return (EMBEDDING_DTYPES as readonly string[]).includes(value);
}

export interface TransformersEmbeddingOptions {
  model?: string;
  dimensions?: number;
  cacheDirectory?: string;
  /** Device for bulk document embedding. Queries always run on the CPU with the same model and dtype. */
  device?: EmbeddingDevice;
  dtype?: EmbeddingDtype;
  pooling?: "mean" | "cls";
  layerNorm?: boolean;
  documentPrefix?: string;
  queryPrefix?: string;
  batchSize?: number;
  maxBatchTokens?: number;
  maxEmbeddingCharacters?: number;
  sortByLength?: boolean;
  /** Total CPU threads for inference, split across the bulk embedding workers. Defaults to the host's logical core count; override on shared or constrained hosts. */
  threads?: number;
  /** Worker threads for bulk CPU embedding, each running its own model session (default: min(threads, 8), capped by memory). 1 embeds in-process. */
  workers?: number;
  /** Approximate resident memory of one embedding worker; caps the default worker count. */
  workerMemoryMegabytes?: number;
  /** Overrides the embedding worker script; used by tests. */
  workerUrl?: URL;
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

// Each worker holds its own copy of the model (about 160-300 MB for MiniLM), so the default stops here even on hosts with more cores.
export const DEFAULT_MAX_EMBEDDING_WORKERS = 8;
// Share of physical memory the default worker count may claim. Six nomic workers (about 1.4 GB each) on a 16 GB Mac
// drove it into 12 GB of swap and cut throughput from ~35 to ~3 chunks per second.
const WORKER_MEMORY_SHARE = 0.35;
const DEFAULT_WORKER_MEMORY_MEGABYTES = 1500;

function isOutOfMemoryError(error: unknown): boolean {
  const message = error instanceof Error ? `${error.message} ${error.cause instanceof Error ? error.cause.message : ""}` : String(error);
  return /out of memory|not enough memory resources|8007000e|resource[s]? are available to complete/i.test(message);
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

/**
 * Splits texts into batches of similar length so little of each batch is padding. `tokenCounts`, when given, orders
 * texts by real token count: character length is a poor proxy (identifier-heavy text tokenizes far denser), and
 * sorting by it left about half of every padded batch empty. The token budget keeps using the conservative byte
 * estimate it was tuned against.
 */
export function createEmbeddingBatches(texts: string[], batchSize: number, sortByLength = true, maxBatchTokens = Number.POSITIVE_INFINITY, tokenCounts?: number[]): IndexedText[][] {
  if (tokenCounts && tokenCounts.length !== texts.length) throw new Error(`Expected ${texts.length} token counts, received ${tokenCounts.length}`);
  const indexed = texts.map((text, index) => ({ index, text }));
  const length = tokenCounts ? (item: IndexedText) => tokenCounts[item.index]! : (item: IndexedText) => item.text.length;
  if (sortByLength) indexed.sort((left, right) => length(left) - length(right) || left.index - right.index);
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
  /** Bulk CPU embedding workers. With a GPU device they take over only after a GPU failure. */
  readonly workers: number;
  readonly threadsPerWorker: number;
  readonly workerMemoryMegabytes: number;
  readonly metrics: EmbeddingMetrics = { requests: 0, retries: 0, truncations: 0, batchSplits: 0, deviceTransitions: 0, workerFallbacks: 0 };
  private readonly options: TransformersEmbeddingOptions;
  // GPU session for bulk embedding; unused on the CPU.
  private pipeline?: Promise<FeatureExtractionPipeline>;
  // CPU session for queries, and for bulk embedding that isn't pooled or on a GPU.
  private cpuPipeline?: Promise<FeatureExtractionPipeline>;
  private tokenizer?: Promise<PreTrainedTokenizer>;
  private layerNormOperation?: typeof import("@huggingface/transformers").layer_norm;
  private pool?: EmbeddingWorkerPool;
  private poolUnavailable = false;
  // Two lanes so a query embedded during a large in-process bulk pass only waits for the in-flight batch to finish, not the whole pass: bulk work is scheduled one batch at a time, and the drain loop always prefers a queued query over the next bulk batch.
  private readonly bulkTaskQueue: Array<() => Promise<void>> = [];
  private readonly priorityTaskQueue: Array<() => Promise<void>> = [];
  private draining = false;
  private runtimeDevice: EmbeddingDevice;

  constructor(options: TransformersEmbeddingOptions = {}) {
    this.options = options;
    // Unset options come from the default embedding profile, so a bare provider matches a default index.
    const defaults = EMBEDDING_PROFILES[DEFAULT_EMBEDDING_PROFILE];
    this.model = options.model ?? defaults.model;
    this.dimensions = options.dimensions ?? defaults.dimensions;
    this.device = options.device ?? DEFAULT_DEVICE;
    this.dtype = options.dtype ?? "q8";
    this.pooling = options.pooling ?? defaults.pooling;
    this.layerNorm = options.layerNorm ?? this.model === "nomic-ai/nomic-embed-text-v1.5";
    this.documentPrefix = options.documentPrefix ?? defaults.documentPrefix;
    this.queryPrefix = options.queryPrefix ?? defaults.queryPrefix;
    this.threads = Math.max(1, Math.floor(options.threads ?? os.availableParallelism()));
    this.workerMemoryMegabytes = options.workerMemoryMegabytes ?? (options.model === undefined ? defaults.workerMemoryMegabytes : DEFAULT_WORKER_MEMORY_MEGABYTES);
    const memoryLimitedWorkers = Math.floor((os.totalmem() / 1024 / 1024) * WORKER_MEMORY_SHARE / this.workerMemoryMegabytes);
    this.workers = Math.max(1, Math.floor(options.workers ?? Math.min(this.threads, DEFAULT_MAX_EMBEDDING_WORKERS, memoryLimitedWorkers)));
    this.threadsPerWorker = Math.max(1, Math.floor(this.threads / this.workers));
    // Pooled workers take smaller batches: with token-sorted batches, 16 and 32 run at the same speed, and smaller batches leave workers idle for less time at the end of each pass.
    this.batchSize = Math.max(1, Math.floor(options.batchSize ?? (this.device === "dml" ? 8 : this.workers > 1 ? 16 : 32)));
    const maxBatchTokens = options.maxBatchTokens ?? (this.device === "dml" ? 4096 : Number.POSITIVE_INFINITY);
    this.maxBatchTokens = Number.isFinite(maxBatchTokens) && maxBatchTokens > 0 ? Math.floor(maxBatchTokens) : Number.POSITIVE_INFINITY;
    const maxEmbeddingCharacters = options.maxEmbeddingCharacters ?? defaults.maxEmbeddingCharacters;
    this.maxEmbeddingCharacters = Number.isFinite(maxEmbeddingCharacters) && maxEmbeddingCharacters > 0 ? Math.floor(maxEmbeddingCharacters) : Number.POSITIVE_INFINITY;
    if (Number.isFinite(this.maxEmbeddingCharacters) && this.maxEmbeddingCharacters <= Math.max(this.documentPrefix.length, this.queryPrefix.length)) {
      throw new Error("maxEmbeddingCharacters must exceed the configured embedding prefix length");
    }
    this.runtimeDevice = this.device;
  }

  get activeDevice(): EmbeddingDevice {
    return this.runtimeDevice;
  }

  private get inferenceSettings(): InferenceSettings {
    return { model: this.model, pooling: this.pooling, layerNorm: this.layerNorm, dimensions: this.dimensions };
  }

  private createPipeline(device: EmbeddingDevice): Promise<FeatureExtractionPipeline> {
    return import("@huggingface/transformers").then(async (transformers) => {
      if (this.options.cacheDirectory) transformers.env.cacheDir = this.options.cacheDirectory;
      this.layerNormOperation = transformers.layer_norm;
      // onnxruntime-node's default intra-op thread count isn't always the full logical core count; on a low-core machine that leaves real CPU throughput on the table for a compute-bound embedding pass. Configurable (this.threads) since host core counts vary.
      const sessionOptions = device === "cpu" ? { intraOpNumThreads: this.threads } : undefined;
      return transformers.pipeline("feature-extraction", this.model, { device, dtype: this.dtype, session_options: sessionOptions });
    });
  }

  // The in-process session for bulk embedding on the current device.
  private load(): Promise<FeatureExtractionPipeline> {
    if (this.runtimeDevice === "cpu") return this.loadCpu();
    if (!this.pipeline) {
      const loading = this.createPipeline(this.runtimeDevice);
      this.pipeline = loading;
      void loading.catch(() => {
        if (this.pipeline === loading) this.pipeline = undefined;
      });
    }
    return this.pipeline;
  }

  // Queries always use this session, so a search never loads, waits for, or depends on a GPU.
  private loadCpu(): Promise<FeatureExtractionPipeline> {
    if (!this.cpuPipeline) {
      const loading = this.createPipeline("cpu");
      this.cpuPipeline = loading;
      void loading.catch(() => {
        if (this.cpuPipeline === loading) this.cpuPipeline = undefined;
      });
    }
    return this.cpuPipeline;
  }

  private loadTokenizer(): Promise<PreTrainedTokenizer> {
    if (!this.tokenizer) {
      const loading = import("@huggingface/transformers").then((transformers) => {
        if (this.options.cacheDirectory) transformers.env.cacheDir = this.options.cacheDirectory;
        return transformers.AutoTokenizer.from_pretrained(this.model);
      });
      this.tokenizer = loading;
      void loading.catch(() => {
        if (this.tokenizer === loading) this.tokenizer = undefined;
      });
    }
    return this.tokenizer;
  }

  private async countTokens(texts: string[]): Promise<number[]> {
    const tokenizer = await this.loadTokenizer();
    return texts.map((text) => tokenizer(text, { truncation: true }).input_ids.dims.at(-1) ?? 1);
  }

  /** The worker pool for bulk CPU embedding, or undefined to embed in-process. */
  private async bulkPool(): Promise<EmbeddingWorkerPool | undefined> {
    if (this.workers <= 1 || this.runtimeDevice !== "cpu" || this.poolUnavailable) return undefined;
    if (!this.pool) {
      const workerData: EmbeddingWorkerData = { ...this.inferenceSettings, dtype: this.dtype, cacheDirectory: this.options.cacheDirectory, threads: this.threadsPerWorker };
      this.pool = new EmbeddingWorkerPool({ size: this.workers, workerUrl: this.options.workerUrl ?? new URL("./embedding-worker.js", import.meta.url), workerData });
    }
    try {
      if (!this.pool.available) throw new Error("the embedding worker script is missing; run the built package");
      await this.pool.start();
      return this.pool;
    } catch (error) {
      // Embedding still works in-process, just without the parallel speed-up.
      this.poolUnavailable = true;
      this.metrics.workerFallbacks += 1;
      console.error(`Embedding workers unavailable (${error instanceof Error ? error.message : String(error)}); embedding in-process.`);
      return undefined;
    }
  }

  private async inferInProcess(texts: string[], query = false): Promise<Float32Array[]> {
    const extractor = await (query ? this.loadCpu() : this.load());
    return splitVectors(await extractEmbeddings(extractor, this.layerNormOperation!, texts, this.inferenceSettings), texts.length, this.dimensions);
  }

  private useCpuFallback(error: unknown): void {
    if (this.runtimeDevice === "cpu") return;
    const device = this.runtimeDevice;
    const gpuPipeline = this.pipeline;
    this.pipeline = undefined;
    this.runtimeDevice = "cpu";
    this.metrics.deviceTransitions += 1;
    console.error(`Embedding on ${device} failed (${error instanceof Error ? error.message : String(error)}); continuing on the CPU with the same model and dtype.`);
    // Released without waiting: a wedged GPU can hang its own teardown, and that must not stall the CPU continuation.
    void gpuPipeline?.then((pipeline) => pipeline.dispose()).catch(() => undefined);
  }

  // A batch that runs out of memory is halved. Any other failure on a GPU (a lost device, a driver fault, an operator
  // the provider can't run) moves embedding to the CPU with the same model and dtype, so the vectors stay consistent.
  private async inferWithRecovery(texts: string[], infer: (texts: string[]) => Promise<Float32Array[]>): Promise<Float32Array[]> {
    try {
      return await infer(texts);
    } catch (error) {
      if (isOutOfMemoryError(error) && texts.length > 1) {
        this.metrics.batchSplits += 1;
        const midpoint = Math.ceil(texts.length / 2);
        return [...await this.inferWithRecovery(texts.slice(0, midpoint), infer), ...await this.inferWithRecovery(texts.slice(midpoint), infer)];
      }
      if (this.runtimeDevice === "cpu") throw error;
      this.useCpuFallback(error);
      return this.inferWithRecovery(texts, infer);
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
    const deliver = async (batch: IndexedText[], batchVectors: Float32Array[]): Promise<void> => {
      if (vectors) batch.forEach((item, row) => { vectors[item.index] = batchVectors[row]!; });
      completed += batch.length;
      await onBatch?.({ indexes: batch.map((item) => item.index), vectors: batchVectors, completed, total: texts.length });
      onProgress?.(completed, texts.length);
    };
    const sortByLength = this.options.sortByLength ?? true;
    const tokenCounts = sortByLength && texts.length > 1 ? await this.countTokens(prefixedTexts) : undefined;
    const batches = createEmbeddingBatches(prefixedTexts, this.batchSize, sortByLength, this.maxBatchTokens, tokenCounts);
    const pool = priority ? undefined : await this.bulkPool();
    if (!pool) {
      for (const batch of batches) {
        const inputs = batch.map((item) => item.text);
        await this.schedule(priority, async () => deliver(batch, priority ? await this.inferInProcess(inputs, true) : await this.inferWithRecovery(inputs, (texts) => this.inferInProcess(texts))));
      }
      return vectors ?? [];
    }
    const inferPooled = async (inputs: string[]) => splitVectors(await pool.run(inputs), inputs.length, this.dimensions);
    // Longest batches first, so the pass ends on short batches instead of leaving most workers idle behind one long one.
    const queue = sortByLength ? [...batches].reverse() : batches;
    // One runner per worker keeps each worker busy. After a failure, runners stop taking batches but let in-flight ones finish and deliver, so no result arrives after this call has rejected.
    let next = 0;
    let failure: unknown;
    const runner = async (): Promise<void> => {
      while (failure === undefined && next < queue.length) {
        const batch = queue[next++]!;
        try {
          await deliver(batch, await this.inferWithRecovery(batch.map((item) => item.text), inferPooled));
        } catch (error) {
          failure ??= error;
        }
      }
    };
    await Promise.all(Array.from({ length: Math.min(pool.size, batches.length) }, runner));
    if (failure !== undefined) throw failure;
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

  async dispose(): Promise<void> {
    const pool = this.pool;
    this.pool = undefined;
    await pool?.dispose();
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
  readonly metrics: EmbeddingMetrics = { requests: 0, retries: 0, truncations: 0, batchSplits: 0, deviceTransitions: 0, workerFallbacks: 0 };
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

  async dispose(): Promise<void> {
    await this.queryProvider.dispose?.();
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