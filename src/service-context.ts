import { chunkDocument, contentHash } from "./chunker.js";
import { DEFAULT_FAMILY, resolvePaths, type DocumentationSearchPaths } from "./config.js";
import { DocumentationSearchDatabase } from "./database.js";
import { TransformersEmbeddingProvider, type EmbeddingProvider } from "./embedder.js";
import { GitHubDocumentationSource, type DocumentationArea, type SourceEntry } from "./github.js";
import type { DocumentChunk, IndexFailure, IndexManifest, SearchOptions, SearchResult, UpdateResult } from "./types.js";

export interface DocumentationSearchOptions {
  dataDirectory?: string;
  embeddingProvider?: EmbeddingProvider;
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

interface ParsedSource {
  path: string;
  blobSha: string;
  contentHash: string;
  chunks: DocumentChunk[];
}

interface PreparedSource extends ParsedSource {
  embeddings: Float32Array[];
}

type ParseResult = { parsed: ParsedSource; failure?: never } | { parsed?: never; failure: IndexFailure };

function failureMessage(error: unknown): string {
  if (!(error instanceof Error)) return String(error);
  const cause = error.cause instanceof Error ? `: ${error.cause.message}` : "";
  return `${error.message}${cause}`;
}

export class DocumentationSearch {
  readonly paths: DocumentationSearchPaths;
  readonly embeddings: EmbeddingProvider;
  readonly source: GitHubDocumentationSource;
  readonly database: DocumentationSearchDatabase;

  constructor(options: DocumentationSearchOptions = {}) {
    this.paths = resolvePaths(options.dataDirectory);
    this.embeddings = options.embeddingProvider ?? new TransformersEmbeddingProvider({ cacheDirectory: this.paths.models });
    this.source = options.source ?? new GitHubDocumentationSource({ repositoryDirectory: this.paths.repository });
    this.database = new DocumentationSearchDatabase(this.paths.database, this.embeddings.dimensions);
    const manifest = this.database.manifest();
    if (manifest && (
      manifest.embeddingProvider !== this.embeddings.name
      || manifest.embeddingModel !== this.embeddings.model
      || manifest.dimensions !== this.embeddings.dimensions
      || manifest.pooling !== (this.embeddings.pooling ?? "mean")
      || (manifest.queryPrefix ?? "") !== (this.embeddings.queryPrefix ?? "")
    )) {
      this.database.close();
      throw new Error(`Index was built with ${manifest.embeddingProvider}/${manifest.embeddingModel} (${manifest.dimensions} dimensions, ${manifest.pooling} pooling), but the active provider is ${this.embeddings.name}/${this.embeddings.model} (${this.embeddings.dimensions} dimensions, ${this.embeddings.pooling ?? "mean"} pooling). Use a separate data directory or run documentationsearch reset-index --yes before rebuilding.`);
    }
  }

  close(): void {
    this.database.close();
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
    progress(`Discovering ${family} documentation...`);
    const tree = await this.source.discover(branch, area);
    const entries = options.limit ? tree.entries.slice(0, options.limit) : tree.entries;
    const current = this.database.sourceMap(family);
    const remotePaths = new Set(entries.map((entry) => entry.path));
    const deleted = options.limit ? [] : [...current.keys()].filter((sourcePath) => !remotePaths.has(sourcePath));
    const changed = entries.filter((entry) => options.refresh || current.get(entry.path)?.blobSha !== entry.blobSha);
    progress(`${changed.length} changed, ${deleted.length} deleted; preparing chunks...`);
    let processed = 0;
    const parseResults = await mapConcurrent(changed, options.concurrency ?? 8, async (entry: SourceEntry): Promise<ParseResult> => {
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
        chunks = chunkDocument(entry.path, markdown, branch, family);
      } catch (error) {
        return failed("chunk", error);
      }
      processed += 1;
      progress(`Prepared ${processed}/${changed.length}: ${entry.path}`);
      return { parsed: { path: entry.path, blobSha: entry.blobSha, contentHash: contentHash(markdown), chunks } };
    });
    const parsed = parseResults.flatMap((result) => result.parsed ? [result.parsed] : []);
    const failures = parseResults.flatMap((result) => result.failure ? [result.failure] : []);
    const prepared: PreparedSource[] = [];
    const allChunks = parsed.flatMap((source) => source.chunks);
    if (allChunks.length) progress(`Embedding ${allChunks.length} chunks in shared batches...`);
    try {
      let lastEmbeddingProgress = 0;
      const progressInterval = Math.max(32, Math.ceil(allChunks.length / 20));
      const allEmbeddings = await this.embeddings.embed(allChunks.map((chunk) => chunk.content), (completed, total) => {
        if (completed === total || completed - lastEmbeddingProgress >= progressInterval) {
          progress(`Embedded ${completed}/${total} chunks...`);
          lastEmbeddingProgress = completed;
        }
      });
      let offset = 0;
      for (const source of parsed) {
        const embeddings = allEmbeddings.slice(offset, offset + source.chunks.length);
        prepared.push({ ...source, embeddings });
        offset += source.chunks.length;
      }
    } catch (batchError) {
      progress(`Shared embedding pass failed (${failureMessage(batchError)}); retrying per document to isolate failures...`);
      for (const source of parsed) {
        let embeddings: Float32Array[];
        try {
          embeddings = await this.embeddings.embed(source.chunks.map((chunk) => chunk.content));
        } catch (error) {
          const failure = { sourcePath: source.path, stage: "embed" as const, error: failureMessage(error) };
          failures.push(failure);
          progress(`Skipped: ${source.path} (embed: ${failure.error})`);
          continue;
        }
        prepared.push({ ...source, embeddings });
      }
    }
    if (failures.length) progress(`Completed with ${failures.length} failed document${failures.length === 1 ? "" : "s"}; see the failures list in the result.`);
    this.database.replaceSources(family, prepared, deleted);
    const manifest: IndexManifest = {
      schemaVersion: 1,
      family,
      branch,
      embeddingProvider: this.embeddings.name,
      embeddingModel: this.embeddings.model,
      dimensions: this.embeddings.dimensions,
      pooling: this.embeddings.pooling ?? "mean",
      normalized: true,
      queryPrefix: this.embeddings.queryPrefix,
      updatedAt: new Date().toISOString(),
      sourceCommit: tree.commit,
    };
    this.database.setManifest(manifest);
    const addedCount = prepared.filter((source) => !current.has(source.path)).length;
    const changedCount = prepared.length - addedCount;
    return {
      family,
      discovered: entries.length,
      added: addedCount,
      changed: changedCount,
      deleted: deleted.length,
      chunks: prepared.reduce((total, item) => total + item.chunks.length, 0),
      failures,
    };
  }

  async search(query: string, options: SearchOptions = {}): Promise<SearchResult[]> {
    if (!query.trim()) throw new Error("Search query cannot be empty");
    const limit = options.limit ?? 10;
    const threshold = options.threshold ?? 0.3;
    if (!Number.isInteger(limit) || limit < 1 || limit > 50) throw new Error("limit must be an integer between 1 and 50");
    if (!Number.isFinite(threshold) || threshold < -1 || threshold > 1) throw new Error("threshold must be a finite number between -1 and 1");
    const embedding = this.embeddings.embedQuery
      ? await this.embeddings.embedQuery(query)
      : (await this.embeddings.embed([query]))[0]!;
    return this.database.search(query, embedding, options, limit, threshold, options.deduplicateReleases);
  }

  getDocument(sourcePath: string, release?: string) {
    return this.database.getDocument(sourcePath, release).map((chunk) => ({
      ...chunk,
      content: String(chunk.metadata.full_content ?? chunk.metadata.section_content ?? chunk.content),
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
    return { ...this.database.stats(), manifest: this.database.manifest(), dataDirectory: this.paths.root };
  }
}