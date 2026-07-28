import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";

const packageJson = createRequire(import.meta.url)("../package.json") as { name: string; version: string };

export const PACKAGE_NAME = packageJson.name;
export const PACKAGE_VERSION = packageJson.version;
export const DEFAULT_FAMILY = "australia";
export const DEFAULT_MODEL = "Xenova/bge-base-en-v1.5";
export const DEFAULT_DIMENSIONS = 768;
export const DEFAULT_POOLING = "cls";
export const DEFAULT_DEVICE = "cpu";
export const DEFAULT_DOCUMENT_PREFIX = "";
export const DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: ";
export const DEFAULT_MAX_EMBEDDING_CHARACTERS = 2048;
export const SEARCH_SCHEMA_VERSION = 6;

export interface DocumentationSearchPaths {
  root: string;
  database: string;
  models: string;
  repository: string;
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
  };
}