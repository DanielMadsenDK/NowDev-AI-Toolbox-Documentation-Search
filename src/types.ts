export const DOC_TYPES = ["scripting-api", "rest-api", "scripting-guide", "developer-guide", "product-doc", "release-notes", "glossary"] as const;
export type DocType = typeof DOC_TYPES[number];

export const CHUNK_TYPES = ["overview", "section", "method", "endpoint", "parameter", "returns", "example", "definition"] as const;
export type ChunkType = typeof CHUNK_TYPES[number];

/** Subsets of ServiceNowDocs an update can index. */
export const DOCUMENTATION_AREAS = ["all-docs", "scripting", "server", "client", "scripts"] as const;
export type DocumentationArea = typeof DOCUMENTATION_AREAS[number];

export interface DocumentChunk {
  id?: number;
  docType: DocType;
  publication: string;
  sourcePath: string;
  release: string;
  chunkType: ChunkType;
  chunkIndex: number;
  title: string;
  heading: string | null;
  content: string;
  topicType: string | null;
  product: string | null;
  classification: string | null;
  lastUpdated: string | null;
  objectName: string | null;
  methodName: string | null;
  /** Breadcrumb and heading path indexed as a weighted full-text column; not part of the returned content. */
  context?: string | null;
  metadata: Record<string, unknown>;
  contentHash: string;
}

export interface SearchFilters {
  release?: string;
  docType?: DocType;
  publication?: string;
  chunkType?: ChunkType;
  topicType?: string;
}

export interface SearchOptions extends SearchFilters {
  limit?: number;
  threshold?: number;
  deduplicateReleases?: boolean;
  maxResultsPerSource?: number;
  /** Rerank the top fused candidates with the configured cross-encoder (default: false). Improves some how-to and glossary queries at several hundred milliseconds of CPU time per search. */
  rerank?: boolean;
}

export interface SearchResult extends DocumentChunk {
  similarity: number;
  score: number;
}

export interface IndexManifest {
  schemaVersion: number;
  family: string;
  branch: string;
  embeddingProfile?: string;
  embeddingProvider: string;
  embeddingModel: string;
  documentEmbeddingModel?: string;
  dimensions: number;
  pooling: string;
  dtype?: string;
  layerNorm?: boolean;
  normalized: boolean;
  documentPrefix?: string;
  queryPrefix?: string;
  maxEmbeddingCharacters?: number;
  updatedAt: string;
  sourceCommit?: string;
}

export interface UpdateResult {
  family: string;
  discovered: number;
  added: number;
  changed: number;
  deleted: number;
  chunks: number;
  /** Chunks whose vectors were reused from the embedding cache instead of being embedded. */
  cachedChunks: number;
  failures: IndexFailure[];
}

export interface IndexFailure {
  sourcePath: string;
  stage: "download" | "chunk" | "embed";
  error: string;
}