import os from "node:os";
import path from "node:path";

export const PACKAGE_NAME = "@nowdevaitoolbox/documentationsearch";
export const PACKAGE_VERSION = "0.1.0";
export const DEFAULT_FAMILY = "australia";
export const DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5";
export const DEFAULT_DIMENSIONS = 768;
export const DEFAULT_POOLING = "mean";
export const DEFAULT_DOCUMENT_PREFIX = "search_document: ";
export const DEFAULT_QUERY_PREFIX = "search_query: ";

export interface DocumentationSearchPaths {
  root: string;
  database: string;
  models: string;
  repository: string;
}

export function defaultDataDirectory(): string {
  if (process.env.DOCUMENTATIONSEARCH_HOME) return path.resolve(process.env.DOCUMENTATIONSEARCH_HOME);
  if (process.env.SERVICECONTEXT_HOME) return path.resolve(process.env.SERVICECONTEXT_HOME);
  const cacheRoot = process.platform === "win32"
    ? process.env.LOCALAPPDATA ?? os.homedir()
    : process.platform === "darwin"
      ? path.join(os.homedir(), "Library", "Caches")
      : process.env.XDG_CACHE_HOME ?? path.join(os.homedir(), ".cache");
  return path.join(cacheRoot, "documentationsearch");
}

export function resolvePaths(root = defaultDataDirectory()): DocumentationSearchPaths {
  const absoluteRoot = path.resolve(root);
  return {
    root: absoluteRoot,
    database: path.join(absoluteRoot, "index.sqlite"),
    models: path.join(absoluteRoot, "models"),
    repository: path.join(absoluteRoot, "ServiceNowDocs"),
  };
}