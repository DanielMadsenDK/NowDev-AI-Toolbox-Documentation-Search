import type { TransformersEmbeddingOptions } from "./embedder.js";

export const DEFAULT_EMBEDDING_PROFILE = "all-minilm-l6-v2";

export const EMBEDDING_PROFILES = {
  "bge-base-en-v1.5": {
    model: "Xenova/bge-base-en-v1.5",
    dimensions: 768,
    pooling: "cls",
    layerNorm: false,
    documentPrefix: "",
    queryPrefix: "Represent this sentence for searching relevant passages: ",
    maxEmbeddingCharacters: 2048,
  },
  "nomic-embed-text-v1.5": {
    model: "nomic-ai/nomic-embed-text-v1.5",
    dimensions: 768,
    pooling: "mean",
    layerNorm: true,
    documentPrefix: "search_document: ",
    queryPrefix: "search_query: ",
    maxEmbeddingCharacters: 2048,
  },
  "nomic-embed-text-v1": {
    model: "nomic-ai/nomic-embed-text-v1",
    dimensions: 768,
    pooling: "mean",
    layerNorm: false,
    documentPrefix: "search_document: ",
    queryPrefix: "search_query: ",
    maxEmbeddingCharacters: 2048,
  },
  "multilingual-e5-small": {
    model: "Xenova/multilingual-e5-small",
    dimensions: 384,
    pooling: "mean",
    layerNorm: false,
    documentPrefix: "passage: ",
    queryPrefix: "query: ",
    maxEmbeddingCharacters: 2048,
  },
  "all-minilm-l6-v2": {
    model: "Xenova/all-MiniLM-L6-v2",
    dimensions: 384,
    pooling: "mean",
    layerNorm: false,
    documentPrefix: "",
    queryPrefix: "",
    maxEmbeddingCharacters: 1024,
  },
} as const satisfies Record<string, Pick<TransformersEmbeddingOptions, "model" | "dimensions" | "pooling" | "layerNorm" | "documentPrefix" | "queryPrefix" | "maxEmbeddingCharacters">>;

export type EmbeddingProfileName = keyof typeof EMBEDDING_PROFILES;

export function embeddingProfile(name: EmbeddingProfileName) {
  return EMBEDDING_PROFILES[name];
}

export function isEmbeddingProfileName(value: string): value is EmbeddingProfileName {
  return Object.hasOwn(EMBEDDING_PROFILES, value);
}