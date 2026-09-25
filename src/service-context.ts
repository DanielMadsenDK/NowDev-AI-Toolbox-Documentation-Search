import { chunkDocument, contentHash } from "./chunker.js";
import { DEFAULT_FAMILY, DEFAULT_MAX_EMBEDDING_CHARACTERS, resolvePaths, SEARCH_SCHEMA_VERSION, type DocumentationSearchPaths } from "./config.js";
import { DocumentationSearchDatabase, readStoredDimensions, readStoredEmbeddingDtype, readStoredEmbeddingProfile } from "./database.js";
import { EmbeddingCache, embeddingCacheSize, embeddingTextHash } from "./embedding-cache.js";
import { DEFAULT_EMBEDDING_PROFILE, embeddingProfile, isEmbeddingProfileName, type EmbeddingProfileName } from "./embedding-profiles.js";
import { EndpointDocumentEmbeddingProvider, isEmbeddingDtype, TransformersEmbeddingProvider, type EmbeddingBatch, type EmbeddingDevice, type EmbeddingProvider, type TransformersEmbeddingOptions } from "./embedder.js";
import { GitHubDocumentationSource, type DocumentationArea, type SourceEntry } from "./github.js";
import { rerankPassage, TransformersReranker, type Reranker } from "./reranker.js";
import type { DocumentChunk, IndexFailure, IndexManifest, SearchOptions, SearchResult, UpdateResult } from "./types.js";

export interface DocumentationSearchOptions {
  dataDirectory?: string;
  embeddingDevice?: EmbeddingDevice;
  embeddingDtype?: TransformersEmbeddingOptions["dtype"];
  embeddingBatchSize?: number;
  embeddingMaxCharacters?: number;
  embeddingThreads?: number;
  /** Worker threads for bulk CPU document embedding (default: min(embeddingThreads, 8)); 1 embeds in-process. */
  embeddingWorkers?: number;
  embeddingProfile?: EmbeddingProfileName;
  embeddingDimensions?: number;
  embeddingEndpoint?: string;
  embeddingEndpointApiKey?: string;
  embeddingEndpointModel?: string;
  embeddingEndpointBatchSize?: number;
  embeddingEndpointConcurrency?: number;
  embeddingEndpointTimeoutMilliseconds?: number;
  vectorOversample?: number;
  embeddingProvider?: EmbeddingProvider;
  /** Reuse document vectors from the persistent embedding cache. Defaults to true, except with a custom embeddingProvider whose identity may not be stable. */
  embeddingCache?: boolean;
  /** Cross-encoder used to rerank the top fused candidates. Defaults to a local Transformers.js reranker, except with a custom embeddingProvider; pass false to disable. */
  reranker?: Reranker | false;
  rerankerModel?: string;
  source?: GitHubDocumentationSource;
}

export interface UpdateOptions {
  family?: string;
  branch?: string;
  area?: DocumentationArea;
  refresh?: boolean;
  concurrency?: number;
  limit?: number;
  onProgress?: (message: string) => void;
}

async function mapConcurrent<T, R>(items: T[], concurrency: number, action: (item: T, index: number) => Promise<R>): Promise<R[]> {
  const output = new Array<R>(items.length);
  let cursor = 0;
  let failure: unknown;
  async function worker(): Promise<void> {
    while (cursor < items.length && failure === undefined) {
      const index = cursor++;
      try {
        output[index] = await action(items[index]!, index);
      } catch (error) {
        failure ??= error;
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, Math.max(items.length, 1)) }, worker));
  if (failure !== undefined) throw failure;
  return output;
}

// Chunks per shared-batch group. Batching sorts texts by length within a group to cut wasted padding, which scrambles document order — a document only commits once every one of its chunks is embedded. Keeping groups well below the size of a full large corpus means a document waits on its own group (a few hundred documents), not the entire run, before its embeddings land in the database.
const EMBEDDING_GROUP_CHUNK_TARGET = 4000;
const PREPARATION_SOURCE_BATCH_SIZE = 128;

function groupSources<T extends { chunks: unknown[] }>(sources: T[], targetChunks: number): T[][] {
  const groups: T[][] = [];
  let current: T[] = [];
  let currentChunks = 0;
  for (const source of sources) {
    if (current.length && currentChunks + source.chunks.length > targetChunks) {
      groups.push(current);
      current = [];
      currentChunks = 0;
    }
    current.push(source);
    currentChunks += source.chunks.length;
  }
  if (current.length) groups.push(current);
  return groups;
}

interface ParsedSource {
  path: string;
  blobSha: string;
  contentHash: string;
  content: string;
  chunks: DocumentChunk[];
}

interface PreparedSource extends ParsedSource {
  embeddings: Float32Array[];
  /** Every vector came from the embedding cache, so there is nothing new to store in it. */
  fromCache?: boolean;
}

interface EmbeddingState {
  source: ParsedSource;
  embeddings: Array<Float32Array | undefined>;
  completed: number;
}

type ParseResult = { parsed: ParsedSource; failure?: never } | { parsed?: never; failure: IndexFailure };
type ParseBatchOutcome = { ok: true; results: ParseResult[] } | { ok: false; error: unknown };

// Fused candidates scored by the cross-encoder. Deeper pools cost roughly linear reranking time for little recall gain.
const RERANK_DEPTH = 30;
// Share of the final score taken from the cross-encoder; the rest keeps the fused rank's exact-identifier and keyword evidence.
const RERANK_WEIGHT = 0.7;

const QUERY_EMBEDDING_CACHE_SIZE = 256;

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

function failureMessage(error: unknown): string {
  if (!(error instanceof Error)) return String(error);
  const cause = error.cause instanceof Error ? `: ${error.cause.message}` : "";
  return `${error.message}${cause}`;
}

export class DocumentationSearch {
  readonly paths: DocumentationSearchPaths;
  readonly embeddings: EmbeddingProvider;
  readonly embeddingProfileName: EmbeddingProfileName | null;
  readonly source: GitHubDocumentationSource;
  readonly database: DocumentationSearchDatabase;
  readonly reranker: Reranker | null;
  readonly embeddingCacheEnabled: boolean;
  private readonly queryEmbeddings = new Map<string, Promise<Float32Array>>();

  constructor(options: DocumentationSearchOptions = {}) {
    this.paths = resolvePaths(options.dataDirectory);
    if (options.embeddingProvider && options.embeddingEndpoint) throw new Error("embeddingProvider and embeddingEndpoint cannot be used together");
    if (Boolean(options.embeddingEndpoint) !== Boolean(options.embeddingEndpointModel)) throw new Error("embeddingEndpoint and embeddingEndpointModel must be specified together");
    if (options.embeddingEndpoint && !options.embeddingEndpointApiKey) throw new Error("embeddingEndpointApiKey is required when embeddingEndpoint is configured");
    const storedValue = options.embeddingProvider ? null : readStoredEmbeddingProfile(this.paths.database);
    if (storedValue && !isEmbeddingProfileName(storedValue)) {
      throw new Error(`Index uses unknown embedding profile ${storedValue}. Upgrade DocumentationSearch or use a supported index.`);
    }
    const storedProfile: EmbeddingProfileName | null = storedValue && isEmbeddingProfileName(storedValue) ? storedValue : null;
    const selectedProfile = options.embeddingProfile ?? storedProfile ?? DEFAULT_EMBEDDING_PROFILE;
    this.embeddingProfileName = options.embeddingProvider ? null : selectedProfile;
    const profile = embeddingProfile(selectedProfile);
    const storedDimensions = options.embeddingProvider ? null : readStoredDimensions(this.paths.database);
    // A stored dimension count only applies to the profile that wrote it; reopening under a different (or unknown) profile must not silently reuse it.
    const selectedDimensions = options.embeddingDimensions ?? (storedProfile === selectedProfile ? storedDimensions : null) ?? profile.dimensions;
    if (!options.embeddingProvider && selectedDimensions !== profile.dimensions) {
      if (profile.minDimensions === undefined) {
        throw new Error(`Embedding profile ${selectedProfile} does not support custom dimensions (fixed at ${profile.dimensions}). Omit --embedding-dimensions or choose a profile that supports it.`);
      }
      if (selectedDimensions < profile.minDimensions || selectedDimensions > profile.dimensions) {
        throw new Error(`--embedding-dimensions must be between ${profile.minDimensions} and ${profile.dimensions} for ${selectedProfile}.`);
      }
    }
    const storedDtype = options.embeddingProvider ? null : readStoredEmbeddingDtype(this.paths.database);
    if (storedDtype && !isEmbeddingDtype(storedDtype)) {
      throw new Error(`Index uses unknown embedding dtype ${storedDtype}. Upgrade DocumentationSearch or use a supported index.`);
    }
    // Like the dimension count, a stored dtype only applies to the profile that wrote it. A new index built on a GPU
    // defaults to fp16: the q8 model's dynamically quantized operators run slower on WebGPU than on the CPU.
    const selectedDtype = options.embeddingDtype
      ?? (storedProfile === selectedProfile && storedDtype && isEmbeddingDtype(storedDtype) ? storedDtype : undefined)
      ?? (options.embeddingDevice === "webgpu" ? "fp16" : undefined);
    const localEmbeddings = options.embeddingProvider ?? new TransformersEmbeddingProvider({
      ...profile,
      cacheDirectory: this.paths.models,
      device: options.embeddingDevice,
      dtype: selectedDtype,
      batchSize: options.embeddingBatchSize,
      ...(options.embeddingMaxCharacters === undefined ? {} : { maxEmbeddingCharacters: options.embeddingMaxCharacters }),
      threads: options.embeddingThreads,
      workers: options.embeddingWorkers,
      dimensions: selectedDimensions,
    });
    this.embeddings = options.embeddingEndpoint ? new EndpointDocumentEmbeddingProvider(localEmbeddings, {
      endpoint: options.embeddingEndpoint,
      apiKey: options.embeddingEndpointApiKey!,
      model: options.embeddingEndpointModel!,
      batchSize: options.embeddingEndpointBatchSize,
      concurrency: options.embeddingEndpointConcurrency,
      timeoutMilliseconds: options.embeddingEndpointTimeoutMilliseconds,
    }) : localEmbeddings;
    this.reranker = options.reranker === false
      ? null
      : options.reranker ?? (options.embeddingProvider ? null : new TransformersReranker({ model: options.rerankerModel, cacheDirectory: this.paths.models, threads: options.embeddingThreads }));
    this.embeddingCacheEnabled = options.embeddingCache ?? !options.embeddingProvider;
    this.source = options.source ?? new GitHubDocumentationSource({ repositoryDirectory: this.paths.repository });
    this.database = new DocumentationSearchDatabase(this.paths.database, this.embeddings.dimensions, options.vectorOversample);
    if (!options.embeddingProvider) {
      const activeProfile = this.database.embeddingProfile();
      if (activeProfile && activeProfile !== selectedProfile) {
        this.database.close();
        throw new Error(`Index uses embedding profile ${activeProfile}, but ${selectedProfile} was selected. Use --embedding-profile ${activeProfile}, a separate data directory, or reset the index.`);
      }
      this.database.setEmbeddingProfile(selectedProfile);
      // Stored immediately, like the profile, so an interrupted first build cannot resume with other vectors.
      const activeDtype = this.database.embeddingDtype();
      if (activeDtype && activeDtype !== localEmbeddings.dtype) {
        this.database.close();
        throw new Error(`Index uses embedding dtype ${activeDtype}, but ${localEmbeddings.dtype} was selected. Omit --embedding-dtype to use the index's dtype, or use a separate data directory or reset the index.`);
      }
      if (localEmbeddings.dtype) this.database.setEmbeddingDtype(localEmbeddings.dtype);
    }
    const manifest = this.database.manifest();
    if (manifest && manifest.schemaVersion !== SEARCH_SCHEMA_VERSION) {
      this.database.close();
      throw new Error(`Index uses search schema ${manifest.schemaVersion}, but the active schema is ${SEARCH_SCHEMA_VERSION}. Run nowdev-ai-toolbox-documentationsearch reset-index --yes before rebuilding.`);
    }
    if (manifest && (
      manifest.embeddingProvider !== this.embeddings.name
      || manifest.embeddingModel !== this.embeddings.model
      || (this.embeddings.endpointModel !== undefined && manifest.documentEmbeddingModel !== undefined && manifest.documentEmbeddingModel !== this.embeddings.endpointModel)
      || manifest.dimensions !== this.embeddings.dimensions
      || manifest.pooling !== (this.embeddings.pooling ?? "mean")
      || manifest.layerNorm !== this.embeddings.layerNorm
      || (manifest.documentPrefix ?? "") !== (this.embeddings.documentPrefix ?? "")
      || (manifest.queryPrefix ?? "") !== (this.embeddings.queryPrefix ?? "")
      || (manifest.maxEmbeddingCharacters ?? Number.POSITIVE_INFINITY) !== (this.embeddings.maxEmbeddingCharacters ?? Number.POSITIVE_INFINITY)
      || (manifest.dtype ?? "") !== (this.embeddings.dtype ?? "")
    )) {
      this.database.close();
      throw new Error(`Index was built with ${manifest.embeddingProvider}/${manifest.embeddingModel} (${manifest.dimensions} dimensions, ${manifest.pooling} pooling, ${manifest.dtype ?? "default"} dtype), but the active provider is ${this.embeddings.name}/${this.embeddings.model} (${this.embeddings.dimensions} dimensions, ${this.embeddings.pooling ?? "mean"} pooling, ${this.embeddings.dtype ?? "default"} dtype). Use a separate data directory or run nowdev-ai-toolbox-documentationsearch reset-index --yes before rebuilding.`);
    }
  }

  close(): void {
    this.database.close();
    void this.embeddings.dispose?.();
  }

  /**
   * Loads the query embedding model and the in-memory vector index ahead of the first search. Long-running hosts such
   * as the MCP server call this at startup; one-shot CLI searches skip it because loading costs more than it saves.
   */
  async warmUp(options: { rerank?: boolean } = {}): Promise<void> {
    await Promise.all([
      this.embedQuery("warm up"),
      this.database.warmUp(),
      options.rerank && this.reranker?.warmUp ? this.reranker.warmUp() : Promise.resolve(),
    ]);
  }

  private embedQuery(query: string): Promise<Float32Array> {
    // Agents often repeat or retry the same query; a small LRU avoids re-running the model for it.
    const cached = this.queryEmbeddings.get(query);
    if (cached) {
      this.queryEmbeddings.delete(query);
      this.queryEmbeddings.set(query, cached);
      return cached;
    }
    const embedding = this.embeddings.embedQuery
      ? this.embeddings.embedQuery(query)
      : this.embeddings.embed([query]).then((vectors) => vectors[0]!);
    this.queryEmbeddings.set(query, embedding);
    void embedding.catch(() => this.queryEmbeddings.delete(query));
    if (this.queryEmbeddings.size > QUERY_EMBEDDING_CACHE_SIZE) this.queryEmbeddings.delete(this.queryEmbeddings.keys().next().value!);
    return embedding;
  }

  async update(options: UpdateOptions = {}): Promise<UpdateResult> {
    if (options.concurrency !== undefined && (!Number.isInteger(options.concurrency) || options.concurrency < 1 || options.concurrency > 64)) {
      throw new Error("concurrency must be an integer between 1 and 64");
    }
    if (options.limit !== undefined && (!Number.isInteger(options.limit) || options.limit < 1)) {
      throw new Error("limit must be a positive integer");
    }
    const family = (options.family ?? DEFAULT_FAMILY).toLowerCase();
    if (!/^[a-z0-9][a-z0-9._-]*$/.test(family)) throw new Error(`Invalid family name: ${family}`);
    const branch = options.branch ?? family;
    const area = options.area ?? "all-docs";
    const progress = options.onProgress ?? (() => undefined);
    if (this.embeddings.device === "webgpu" && this.embeddings.dtype === "q8") {
      progress("This index uses q8 embeddings, which run slower on WebGPU than on the CPU. For GPU indexing, build a separate fp16 index (a new data directory with --device webgpu).");
    }
    progress(`Discovering ${family} documentation...`);
    const tree = await this.source.discover(branch, area);
    const entries = options.limit ? tree.entries.slice(0, options.limit) : tree.entries;
    const vectorIndexWasLoaded = this.database.vectorIndexLoaded;
    const current = this.database.sourceMap(family);
    const remotePaths = new Set(entries.map((entry) => entry.path));
    const deleted = options.limit ? [] : [...current.keys()].filter((sourcePath) => !remotePaths.has(sourcePath));
    const changed = entries.filter((entry) => options.refresh || current.get(entry.path)?.blobSha !== entry.blobSha);
    const manifest: IndexManifest = {
      schemaVersion: SEARCH_SCHEMA_VERSION,
      family,
      branch,
      embeddingProfile: this.embeddingProfileName ?? undefined,
      embeddingProvider: this.embeddings.name,
      embeddingModel: this.embeddings.model,
      documentEmbeddingModel: this.embeddings.endpointModel,
      dimensions: this.embeddings.dimensions,
      pooling: this.embeddings.pooling ?? "mean",
      dtype: this.embeddings.dtype,
      layerNorm: this.embeddings.layerNorm,
      normalized: true,
      documentPrefix: this.embeddings.documentPrefix,
      queryPrefix: this.embeddings.queryPrefix,
      maxEmbeddingCharacters: this.embeddings.maxEmbeddingCharacters,
      updatedAt: new Date().toISOString(),
      sourceCommit: tree.commit,
    };
    progress(`${changed.length} changed, ${deleted.length} deleted; preparing chunks...`);
    let processed = 0;
    const parseEntries = (sourceEntries: SourceEntry[]) => mapConcurrent(sourceEntries, options.concurrency ?? 8, async (entry: SourceEntry): Promise<ParseResult> => {
      const failed = (stage: IndexFailure["stage"], error: unknown): ParseResult => {
        processed += 1;
        const failure = { sourcePath: entry.path, stage, error: failureMessage(error) };
        progress(`Skipped ${processed}/${changed.length}: ${entry.path} (${stage}: ${failure.error})`);
        return { failure };
      };
      let markdown: string;
      try {
        markdown = await this.source.download(branch, entry);
      } catch (error) {
        return failed("download", error);
      }
      let chunks: DocumentChunk[];
      try {
        const maxChunkCharacters = Math.max(256, (this.embeddings.maxEmbeddingCharacters ?? DEFAULT_MAX_EMBEDDING_CHARACTERS) - 256);
        chunks = chunkDocument(entry.path, markdown, branch, family, maxChunkCharacters);
      } catch (error) {
        return failed("chunk", error);
      }
      processed += 1;
      progress(`Prepared ${processed}/${changed.length}: ${entry.path}`);
      const sourceContent = chunks.find((chunk) => typeof chunk.metadata.full_content === "string")?.metadata.full_content;
      return { parsed: { path: entry.path, blobSha: entry.blobSha, contentHash: contentHash(markdown), content: typeof sourceContent === "string" ? sourceContent : markdown, chunks } };
    });
    // Settles (never rejects) so the next batch can be parsed in the background, concurrently with the current
    // batch's embedding, without an unawaited rejection tripping Node's unhandled-rejection handling before the
    // loop gets back around to consuming it.
    const startParseBatch = (sourceEntries: SourceEntry[]): Promise<ParseBatchOutcome> => parseEntries(sourceEntries).then(
      (results): ParseBatchOutcome => ({ ok: true, results }),
      (error): ParseBatchOutcome => ({ ok: false, error }),
    );
    const failures: IndexFailure[] = [];
    const prepared: Array<{ path: string; chunkCount: number }> = [];
    const committedPaths = new Set<string>();
    // Opened per update rather than per context, so search-only processes never create the file or hold it open.
    const cache = changed.length && this.embeddingCacheEnabled ? EmbeddingCache.open(this.paths.embeddingCache, manifest, progress) : undefined;
    let cachedChunks = 0;
    const commit = (sources: PreparedSource[]): void => {
      if (!sources.length) return;
      this.database.replaceSources(family, sources, [], manifest);
      // After the index write: the cache is only an optimization, so a failure here must not cost the document.
      cache?.store(sources.filter((source) => !source.fromCache).flatMap((source) => source.chunks.map((chunk, index) => ({ hash: embeddingTextHash(chunk.content), vector: source.embeddings[index]! }))));
      for (const source of sources) {
        prepared.push({ path: source.path, chunkCount: source.chunks.length });
        committedPaths.add(source.path);
        // The database now durably holds this source's content, metadata, and vectors; drop the in-memory copies (shared with a group's `states`/`locations` via object identity) so embedding a large corpus doesn't hold every document's text and vectors in memory for the whole run.
        for (const chunk of source.chunks) {
          chunk.content = "";
          chunk.metadata = {};
        }
        source.embeddings.length = 0;
      }
    };
    const embedSource = async (source: ParsedSource): Promise<Float32Array[]> => {
      const embeddings = new Array<Float32Array>(source.chunks.length);
      if (this.embeddings.embedBatched) {
        await this.embeddings.embedBatched(source.chunks.map((chunk) => chunk.content), (batch) => {
          batch.indexes.forEach((index, batchIndex) => {
            const vector = batch.vectors[batchIndex];
            if (!vector) throw new Error(`Embedding count mismatch for ${source.path}`);
            embeddings[index] = vector;
          });
        });
      } else {
        const sourceEmbeddings = await this.embeddings.embed(source.chunks.map((chunk) => chunk.content));
        sourceEmbeddings.forEach((embedding, index) => { embeddings[index] = embedding; });
      }
      if (embeddings.some((embedding) => !embedding)) throw new Error(`Embedding count mismatch for ${source.path}`);
      return embeddings as Float32Array[];
    };
    // Time-based rather than percentage-based: source preparation is streamed, so the final chunk count is not retained in memory up front.
    let overallCompleted = 0;
    let lastEmbeddingProgressTime = Date.now();
    const progressIntervalMilliseconds = 10_000;
    const reportOverallProgress = (): void => {
      const now = Date.now();
      if (now - lastEmbeddingProgressTime >= progressIntervalMilliseconds) {
        progress(`Embedded ${overallCompleted} chunks...`);
        lastEmbeddingProgressTime = now;
      }
    };
    const embedGroup = async (group: ParsedSource[]): Promise<void> => {
      if (!group.some((source) => source.chunks.length)) return;
      // Scoped per group (not the whole corpus): once a group finishes, its states/targets become garbage, and every document in it has committed, however the length sort inside embedBatched reordered this group's chunks.
      const states = group.map<EmbeddingState>((source) => ({ source, embeddings: new Array(source.chunks.length), completed: 0 }));
      const cached = cache?.lookup(states.flatMap((state) => state.source.chunks.map((chunk) => embeddingTextHash(chunk.content)))) ?? [];
      // The model only sees uncached passages, each distinct text once; `targets` fans every vector out to all chunks sharing its text.
      const texts: string[] = [];
      const targets: Array<Array<{ state: EmbeddingState; index: number }>> = [];
      const textPositions = new Map<string, number>();
      let position = 0;
      for (const state of states) {
        state.source.chunks.forEach((chunk, index) => {
          const vector = cached[position++];
          if (vector) {
            state.embeddings[index] = vector;
            state.completed += 1;
            cachedChunks += 1;
            return;
          }
          const existing = textPositions.get(chunk.content);
          if (existing !== undefined) {
            targets[existing]!.push({ state, index });
            return;
          }
          textPositions.set(chunk.content, texts.length);
          texts.push(chunk.content);
          targets.push([{ state, index }]);
        });
      }
      commit(states
        .filter((state) => state.source.chunks.length && state.completed === state.source.chunks.length)
        .map((state) => ({ ...state.source, embeddings: state.embeddings as Float32Array[], fromCache: true })));
      if (!texts.length) return;
      const acceptBatch = async (batch: EmbeddingBatch): Promise<void> => {
        const completedSources: PreparedSource[] = [];
        batch.indexes.forEach((textIndex, batchIndex) => {
          const locations = targets[textIndex];
          const vector = batch.vectors[batchIndex];
          if (!locations || !vector) throw new Error(`Invalid embedding batch index ${textIndex}`);
          for (const { state, index } of locations) {
            if (state.embeddings[index]) throw new Error(`Invalid embedding batch index ${textIndex}`);
            state.embeddings[index] = vector;
            state.completed += 1;
            overallCompleted += 1;
            if (state.completed === state.source.chunks.length) completedSources.push({ ...state.source, embeddings: state.embeddings as Float32Array[] });
          }
        });
        commit(completedSources);
        reportOverallProgress();
      };
      try {
        if (this.embeddings.embedBatched) {
          await this.embeddings.embedBatched(texts, acceptBatch);
        } else {
          const vectors = await this.embeddings.embed(texts);
          if (vectors.length !== texts.length) throw new Error(`Embedding count mismatch: ${vectors.length} vectors for ${texts.length} passages`);
          await acceptBatch({ indexes: vectors.map((_, index) => index), vectors, completed: vectors.length, total: vectors.length });
        }
        // A provider that resolves without delivering every vector would otherwise leave its documents silently uncommitted.
        if (states.some((state) => state.completed < state.source.chunks.length)) throw new Error("Embedding provider returned fewer vectors than passages");
      } catch (batchError) {
        progress(`Shared embedding pass failed for a group (${failureMessage(batchError)}); retrying its documents individually...`);
        for (const source of group) {
          if (committedPaths.has(source.path)) continue;
          try {
            const embeddings = await embedSource(source);
            commit([{ ...source, embeddings }]);
            overallCompleted += source.chunks.length;
          } catch (error) {
            const failure = { sourcePath: source.path, stage: "embed" as const, error: failureMessage(error) };
            failures.push(failure);
            progress(`Skipped: ${source.path} (embed: ${failure.error})`);
            continue;
          }
        }
        reportOverallProgress();
      }
    };
    // One batch of parsing runs ahead of the batch currently being embedded: parsing (network-bound) and embedding
    // (CPU-bound) use disjoint resources, so overlapping them hides one phase's latency behind the other's instead
    // of paying for both in sequence. Only ever one batch ahead, so memory stays bounded for large corpora.
    let nextBatch = changed.length ? startParseBatch(changed.slice(0, PREPARATION_SOURCE_BATCH_SIZE)) : undefined;
    try {
      for (let offset = 0; offset < changed.length; offset += PREPARATION_SOURCE_BATCH_SIZE) {
        const outcome = await nextBatch!;
        if (!outcome.ok) throw outcome.error;
        const nextOffset = offset + PREPARATION_SOURCE_BATCH_SIZE;
        nextBatch = nextOffset < changed.length ? startParseBatch(changed.slice(nextOffset, nextOffset + PREPARATION_SOURCE_BATCH_SIZE)) : undefined;
        const parsed = outcome.results.flatMap((result) => result.parsed ? [result.parsed] : []);
        failures.push(...outcome.results.flatMap((result) => result.failure ? [result.failure] : []));
        if (parsed.some((source) => source.chunks.length)) progress(`Embedding prepared chunks through ${processed}/${changed.length} sources...`);
        for (const group of groupSources(parsed, EMBEDDING_GROUP_CHUNK_TARGET)) await embedGroup(group);
      }
    } finally {
      cache?.close();
    }
    if (overallCompleted) progress(`Embedded ${overallCompleted} chunks.`);
    if (cachedChunks) progress(`Reused ${cachedChunks} cached embeddings.`);
    if (this.embeddings.metrics && (this.embeddings.metrics.retries || this.embeddings.metrics.truncations || this.embeddings.metrics.batchSplits || this.embeddings.metrics.deviceTransitions || this.embeddings.metrics.workerFallbacks)) {
      const metrics = this.embeddings.metrics;
      progress(`Embedding adjustments: ${metrics.retries} retries, ${metrics.truncations} truncations, ${metrics.batchSplits} batch splits, ${metrics.deviceTransitions} device transitions, ${metrics.workerFallbacks} worker fallbacks.`);
    }
    if (failures.length) progress(`Completed with ${failures.length} failed document${failures.length === 1 ? "" : "s"}; see the failures list in the result.`);
    if (deleted.length) this.database.replaceSources(family, [], deleted, manifest);
    if (prepared.length || deleted.length) {
      progress("Optimizing the full-text search index...");
      this.database.optimizeSearchIndex();
      if (vectorIndexWasLoaded) await this.database.warmUp();
    }
    this.database.setManifest(manifest);
    const addedCount = prepared.filter((source) => !current.has(source.path)).length;
    const changedCount = prepared.length - addedCount;
    return {
      family,
      discovered: entries.length,
      added: addedCount,
      changed: changedCount,
      deleted: deleted.length,
      chunks: prepared.reduce((total, item) => total + item.chunkCount, 0),
      cachedChunks,
      failures,
    };
  }

  async search(query: string, options: SearchOptions = {}): Promise<SearchResult[]> {
    if (!query.trim()) throw new Error("Search query cannot be empty");
    const limit = options.limit ?? 10;
    const threshold = options.threshold ?? 0.3;
    const maxResultsPerSource = options.maxResultsPerSource ?? 3;
    if (!Number.isInteger(limit) || limit < 1 || limit > 50) throw new Error("limit must be an integer between 1 and 50");
    if (!Number.isFinite(threshold) || threshold < -1 || threshold > 1) throw new Error("threshold must be a finite number between -1 and 1");
    if (!Number.isInteger(maxResultsPerSource) || maxResultsPerSource < 1 || maxResultsPerSource > 10) throw new Error("maxResultsPerSource must be an integer between 1 and 10");
    const embedding = await this.embedQuery(query);
    const rerank = Boolean(this.reranker) && (options.rerank ?? false);
    const candidates = this.database.search(query, embedding, options, rerank ? Math.max(limit, RERANK_DEPTH) : limit, threshold, options.deduplicateReleases, maxResultsPerSource);
    if (!rerank || candidates.length < 2) return candidates.slice(0, limit);
    const logits = await this.reranker!.score(query, candidates.map(rerankPassage));
    const topFusedScore = candidates[0]!.score || 1;
    return candidates
      .map((result, index) => ({ result, score: RERANK_WEIGHT * sigmoid(logits[index]!) + (1 - RERANK_WEIGHT) * (result.score / topFusedScore) }))
      .sort((left, right) => right.score - left.score)
      .slice(0, limit)
      .map(({ result, score }) => ({ ...result, score }));
  }

  getDocument(sourcePath: string, release?: string) {
    const sourceContent = this.database.getSourceContent(sourcePath, release);
    return this.database.getDocument(sourcePath, release).map((chunk, index) => ({
      ...chunk,
      content: index === 0 && sourceContent ? sourceContent : chunk.content,
      metadata: Object.fromEntries(Object.entries(chunk.metadata).filter(([key]) => key !== "full_content" && key !== "section_content")),
    }));
  }

  getDocumentOutline(sourcePath: string, release?: string) {
    return this.database.getDocument(sourcePath, release).map(({ chunkIndex, chunkType, heading, content }) => ({ chunkIndex, chunkType, heading, contentPreview: content.slice(0, 240) }));
  }

  listPublications(release?: string) {
    return this.database.listPublications(release);
  }

  status() {
    return {
      ...this.database.stats(),
      manifest: this.database.manifest(),
      embedding: {
        profile: this.database.embeddingProfile(),
        name: this.embeddings.name,
        model: this.embeddings.model,
        dimensions: this.embeddings.dimensions,
        device: this.embeddings.activeDevice ?? this.embeddings.device ?? "unknown",
        pooling: this.embeddings.pooling ?? "mean",
        dtype: this.embeddings.dtype,
        documentProvider: this.embeddings.endpointModel ? "endpoint" : "local",
        endpointModel: this.embeddings.endpointModel,
        endpointConcurrency: this.embeddings.endpointConcurrency,
        workers: this.embeddings.workers,
        metrics: this.embeddings.metrics,
      },
      embeddingCache: {
        path: this.paths.embeddingCache,
        enabled: this.embeddingCacheEnabled,
        sizeBytes: embeddingCacheSize(this.paths.embeddingCache),
      },
      dataDirectory: this.paths.root,
    };
  }
}