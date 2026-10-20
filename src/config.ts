import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";

const packageJson = createRequire(import.meta.url)("../package.json") as { name: string; version: string };

export const PACKAGE_NAME = packageJson.name;
export const PACKAGE_VERSION = packageJson.version;
export const DEFAULT_FAMILY = "australia";
export const DEFAULT_DEVICE = "cpu";
export const DEFAULT_MAX_EMBEDDING_CHARACTERS = 2048;
export const SEARCH_SCHEMA_VERSION = 7;
export const DEFAULT_SEARCH_LIMIT = 10;
export const MAX_SEARCH_LIMIT = 50;
export const DEFAULT_SIMILARITY_THRESHOLD = 0.3;
export const DEFAULT_MAX_RESULTS_PER_SOURCE = 3;
export const MAX_RESULTS_PER_SOURCE = 10;
export const DEFAULT_DOWNLOAD_CONCURRENCY = 8;
export const MAX_DOWNLOAD_CONCURRENCY = 64;

/** ServiceNowDocs folders that hold API reference pages. "cllent" is the upstream folder's spelling. */
export const API_REFERENCE_PREFIXES = {
  server: "markdown/api-reference/server-api-reference/",
  clientMobile: "markdown/api-reference/cllent-mobile-api-reference/",
  scripts: "markdown/api-reference/scripts/",
  uiBuilder: "markdown/api-reference/ui-builder-api-reference/",
  rest: "markdown/api-reference/rest-apis/",
} as const;
/** API reference pages directly under markdown/api-reference/, outside any sub-folder. */
export const TOP_LEVEL_API_PATH = /^markdown\/api-reference\/[^/]+\.md$/;

export interface DocumentationSearchPaths {
  root: string;
  database: string;
  models: string;
  repository: string;
  /** Content-addressed document embeddings; kept by reset-index so a rebuild reuses unchanged passages. */
  embeddingCache: string;
}

export function defaultDataDirectory(): string {
  if (process.env.NOWDEV_AI_TOOLBOX_DOCUMENTATIONSEARCH_HOME) return path.resolve(process.env.NOWDEV_AI_TOOLBOX_DOCUMENTATIONSEARCH_HOME);
  const cacheRoot = process.platform === "win32"
    ? process.env.LOCALAPPDATA ?? os.homedir()
    : process.platform === "darwin"
      ? path.join(os.homedir(), "Library", "Caches")
      : process.env.XDG_CACHE_HOME ?? path.join(os.homedir(), ".cache");
  return path.join(cacheRoot, "nowdev-ai-toolbox-documentationsearch");
}

export function resolvePaths(root = defaultDataDirectory()): DocumentationSearchPaths {
  const absoluteRoot = path.resolve(root);
  return {
    root: absoluteRoot,
    database: path.join(absoluteRoot, "index.sqlite"),
    models: path.join(absoluteRoot, "models"),
    repository: path.join(absoluteRoot, "ServiceNowDocs"),
    embeddingCache: path.join(absoluteRoot, "embedding-cache.sqlite"),
  };
}