export type DocType =
  | "scripting-api"
  | "rest-api"
  | "scripting-guide"
  | "developer-guide"
  | "product-doc"
  | "release-notes"
  | "glossary";

export type ChunkType =
  | "overview"
  | "section"
  | "method"
  | "endpoint"
  | "parameter"
  | "returns"
  | "example"
  | "definition";

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
}

export interface SearchResult extends DocumentChunk {
  similarity: number;
  score: number;
}

export interface IndexManifest {
  schemaVersion: number;
  family: string;
  branch: string;
  embeddingProvider: string;
  embeddingModel: string;
  dimensions: number;
  pooling: string;
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
  failures: IndexFailure[];
}

export interface IndexFailure {
  sourcePath: string;
  stage: "download" | "chunk" | "embed";
  error: string;
}