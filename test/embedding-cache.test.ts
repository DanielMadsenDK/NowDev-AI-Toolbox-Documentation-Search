import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { chunkDocument } from "../src/chunker.js";
import { SEARCH_SCHEMA_VERSION } from "../src/config.js";
import { DocumentationSearchDatabase } from "../src/database.js";
import { EmbeddingCache, embeddingFingerprint, embeddingTextHash, harvestIndexEmbeddings, type EmbeddingIdentity } from "../src/embedding-cache.js";
import { HashEmbeddingProvider } from "../src/embedder.js";
import type { IndexManifest } from "../src/types.js";
import { removeDirectory } from "./helpers.js";

const identity: EmbeddingIdentity = {
  embeddingProvider: "transformers.js",
  embeddingModel: "Xenova/all-MiniLM-L6-v2",
  dimensions: 4,
  pooling: "mean",
  dtype: "q8",
  layerNorm: false,
  documentPrefix: "",
  maxEmbeddingCharacters: 1024,
};

const temporaryDirectories: string[] = [];
afterEach(async () => {
  for (const directory of temporaryDirectories.splice(0)) await removeDirectory(directory);
});

function directory(): string {
  const created = fs.mkdtempSync(path.join(os.tmpdir(), "embedding-cache-"));
  temporaryDirectories.push(created);
  return created;
}

function open(filename: string, cacheIdentity = identity, errors: string[] = []): EmbeddingCache {
  const cache = EmbeddingCache.open(filename, cacheIdentity, (message) => errors.push(message));
  if (!cache) throw new Error(errors.join("; "));
  return cache;
}

describe("EmbeddingCache", () => {
  it("returns stored vectors bit for bit, across reopening", () => {
    const filename = path.join(directory(), "cache.sqlite");
    const vector = new Float32Array([0.1, -0.2, 0.30000001, Math.fround(1 / 3)]);
    const cache = open(filename);
    cache.store([{ hash: embeddingTextHash("passage"), vector }]);
    cache.close();

    const reopened = open(filename);
    const [hit, miss] = reopened.lookup([embeddingTextHash("passage"), embeddingTextHash("other passage")]);
    expect(Array.from(hit!)).toEqual(Array.from(vector));
    expect(miss).toBeUndefined();
    reopened.close();
  });

  it("keeps embedding configurations apart", () => {
    const filename = path.join(directory(), "cache.sqlite");
    const cache = open(filename);
    cache.store([{ hash: embeddingTextHash("passage"), vector: new Float32Array([1, 0, 0, 0]) }]);
    cache.close();
    for (const change of [{ dtype: "fp32" }, { documentPrefix: "passage: " }, { embeddingModel: "other/model" }, { maxEmbeddingCharacters: 2048 }, { documentEmbeddingModel: "remote/model" }]) {
      const other = open(filename, { ...identity, ...change });
      expect(other.lookup([embeddingTextHash("passage")])[0]).toBeUndefined();
      other.close();
    }
  });

  it("treats manifest round-trip differences as the same configuration", () => {
    expect(embeddingFingerprint({ ...identity, maxEmbeddingCharacters: Number.POSITIVE_INFINITY })).toBe(embeddingFingerprint({ ...identity, maxEmbeddingCharacters: undefined }));
    expect(embeddingFingerprint({ ...identity, documentPrefix: undefined, layerNorm: undefined })).toBe(embeddingFingerprint(identity));
  });

  it("reports an unusable file instead of throwing", () => {
    const filename = path.join(directory(), "cache.sqlite");
    fs.writeFileSync(filename, "not a sqlite database, just text that is long enough to be read as a header");
    const errors: string[] = [];
    expect(EmbeddingCache.open(filename, identity, (message) => errors.push(message))).toBeUndefined();
    expect(errors[0]).toContain("Embedding cache unavailable");
  });

  it("disables itself after a failure instead of throwing", () => {
    const errors: string[] = [];
    const cache = open(path.join(directory(), "cache.sqlite"), identity, errors);
    cache.close();
    expect(() => cache.store([{ hash: embeddingTextHash("passage"), vector: new Float32Array(4) }])).not.toThrow();
    expect(cache.enabled).toBe(false);
    expect(cache.lookup([embeddingTextHash("passage")])).toEqual([]);
    expect(errors).toHaveLength(1);
  });
});

describe("harvestIndexEmbeddings", () => {
  it("copies an index's vectors into the cache under the index's embedding configuration", async () => {
    const root = directory();
    const indexFilename = path.join(root, "index.sqlite");
    const database = new DocumentationSearchDatabase(indexFilename, 8);
    const provider = new HashEmbeddingProvider(8);
    const sourcePath = "markdown/itsm/incidents.md";
    const chunks = chunkDocument(sourcePath, "---\ntitle: Incidents\n---\n# Incidents\nHandle interruptions.\n## Resolve\nRestore service.", "australia", "australia");
    const embeddings = await provider.embed(chunks.map((chunk) => chunk.content));
    const manifest: IndexManifest = {
      schemaVersion: SEARCH_SCHEMA_VERSION,
      family: "australia",
      branch: "australia",
      embeddingProvider: provider.name,
      embeddingModel: provider.model,
      dimensions: provider.dimensions,
      pooling: provider.pooling,
      normalized: true,
      queryPrefix: provider.queryPrefix,
      updatedAt: new Date().toISOString(),
    };
    database.replaceSources("australia", [{ path: sourcePath, blobSha: "sha", contentHash: "hash", chunks, embeddings }], [], manifest);
    database.close();

    const cacheFilename = path.join(root, "embedding-cache.sqlite");
    expect(harvestIndexEmbeddings(indexFilename, cacheFilename)).toEqual({ harvested: chunks.length, vectors: chunks.length });
    // A second harvest finds every vector already cached.
    expect(harvestIndexEmbeddings(indexFilename, cacheFilename)).toEqual({ harvested: 0, vectors: chunks.length });
    const cache = open(cacheFilename, manifest);
    const cached = cache.lookup(chunks.map((chunk) => embeddingTextHash(chunk.content)));
    expect(cached.map((vector) => Array.from(vector!))).toEqual(embeddings.map((vector) => Array.from(vector)));
    cache.close();
  });

  it("skips indexes it cannot reuse", () => {
    const root = directory();
    expect(harvestIndexEmbeddings(path.join(root, "missing.sqlite"), path.join(root, "cache.sqlite"))).toEqual({ skipped: "there is no index" });
    const indexFilename = path.join(root, "index.sqlite");
    new DocumentationSearchDatabase(indexFilename, 8).close();
    expect(harvestIndexEmbeddings(indexFilename, path.join(root, "cache.sqlite"))).toEqual({ skipped: "the index has no manifest" });
  });
});
