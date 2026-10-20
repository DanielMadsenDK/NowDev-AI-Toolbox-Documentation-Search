import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { SEARCH_SCHEMA_VERSION } from "../src/config.js";
import { HashEmbeddingProvider, type EmbeddingBatch } from "../src/embedder.js";
import { GitHubDocumentationSource, type DocumentationArea, type SourceEntry, type SourceTree } from "../src/github.js";
import { DocumentationSearch } from "../src/service-context.js";
import { removeDirectory } from "./helpers.js";

class FakeSource extends GitHubDocumentationSource {
  downloads = 0;
  private readonly entry = { path: "markdown/itsm/incidents.md", blobSha: "sha-1" };

  override async discover(_branch: string, _area: DocumentationArea): Promise<SourceTree> {
    return { commit: "commit-1", entries: [this.entry] };
  }

  override async download(_branch: string, _entry: SourceEntry): Promise<string> {
    this.downloads += 1;
    return "---\ntitle: Incident management\nproduct: ITSM\n---\n# Incident management\nHandle interruptions.\n## Resolve incidents\nRestore service quickly.";
  }
}

class PartiallyFailingSource extends GitHubDocumentationSource {
  attempts = new Map<string, number>();

  override async discover(): Promise<SourceTree> {
    return {
      commit: "commit-partial",
      entries: [
        { path: "markdown/itsm/good.md", blobSha: "good-sha" },
        { path: "markdown/itsm/bad.md", blobSha: "bad-sha" },
      ],
    };
  }

  override async download(_branch: string, entry: SourceEntry): Promise<string> {
    this.attempts.set(entry.path, (this.attempts.get(entry.path) ?? 0) + 1);
    if (entry.path.endsWith("bad.md")) throw new Error("simulated download failure");
    return "---\ntitle: Good document\n---\n# Good document\nSearchable content.";
  }
}

class TwoDocumentSource extends GitHubDocumentationSource {
  override async discover(): Promise<SourceTree> {
    return {
      commit: "two-documents",
      entries: [
        { path: "markdown/itsm/first.md", blobSha: "first" },
        { path: "markdown/itsm/second.md", blobSha: "second" },
      ],
    };
  }

  override async download(_branch: string, entry: SourceEntry): Promise<string> {
    return `---\ntitle: ${entry.path}\n---\n# ${entry.path}\nContent for ${entry.path}.`;
  }
}

class ManyDocumentSource extends GitHubDocumentationSource {
  // Spans three preparation batches (128 + 128 + 1) so the test can distinguish "batch 2 was
  // prefetched while batch 1 embedded" from "everything was parsed up front".
  readonly total = 2 * 128 + 1;
  downloads = 0;

  override async discover(): Promise<SourceTree> {
    return {
      commit: "many-documents",
      entries: Array.from({ length: this.total }, (_, index) => ({ path: `markdown/guides/${index}.md`, blobSha: `sha-${index}` })),
    };
  }

  override async download(_branch: string, entry: SourceEntry): Promise<string> {
    this.downloads += 1;
    return `---\ntitle: ${entry.path}\n---\n# ${entry.path}\nContent.`;
  }
}

class MutableSource extends GitHubDocumentationSource {
  constructor(readonly documents: Map<string, string>) { super(); }

  override async discover(): Promise<SourceTree> {
    return { commit: "mutable", entries: [...this.documents.keys()].map((sourcePath) => ({ path: sourcePath, blobSha: String(this.documents.get(sourcePath)!.length) })) };
  }

  override async download(_branch: string, entry: SourceEntry): Promise<string> {
    return this.documents.get(entry.path)!;
  }
}

const INCIDENTS = "---\ntitle: Incident management\nproduct: ITSM\n---\n# Incident management\nHandle interruptions.\n## Resolve incidents\nRestore service quickly.";

class CountingEmbeddingProvider extends HashEmbeddingProvider {
  embeddedTexts = 0;

  constructor() { super(32); }

  override async embed(texts: string[], onProgress?: (completed: number, total: number) => void): Promise<Float32Array[]> {
    this.embeddedTexts += texts.length;
    return super.embed(texts, onProgress);
  }

  // Counts document passages only; search queries go through here.
  override async embedQuery(query: string): Promise<Float32Array> {
    const [vector] = await super.embed([query]);
    return vector!;
  }
}

class DtypeEmbeddingProvider extends HashEmbeddingProvider {
  constructor(readonly dtype: string) { super(32); }
}

class StreamingEmbeddingProvider extends HashEmbeddingProvider {
  calls = 0;
  afterBatch?: (batch: EmbeddingBatch) => void;

  override async embedBatched(texts: string[], onBatch: (batch: EmbeddingBatch) => void | Promise<void>, onProgress?: (completed: number, total: number) => void): Promise<void> {
    this.calls += 1;
    for (let index = 0; index < texts.length; index += 1) {
      const [vector] = await super.embed([texts[index]!]);
      const batch = { indexes: [index], vectors: [vector!], completed: index + 1, total: texts.length };
      await onBatch(batch);
      this.afterBatch?.(batch);
      onProgress?.(batch.completed, batch.total);
    }
  }
}

const temporaryDirectories: string[] = [];
afterEach(async () => {
  for (const directory of temporaryDirectories.splice(0)) await removeDirectory(directory);
});

describe("ServiceContext", () => {
  it("uses MiniLM for a new index when no profile is selected", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-default-profile-"));
    temporaryDirectories.push(directory);
    const context = new DocumentationSearch({ dataDirectory: directory });
    expect(context.status().embedding).toMatchObject({ profile: "all-minilm-l6-v2", model: "Xenova/all-MiniLM-L6-v2", dimensions: 384, pooling: "mean" });
    context.close();
  });

  it("persists and automatically reloads a curated embedding profile", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-profile-"));
    temporaryDirectories.push(directory);
    const selected = new DocumentationSearch({ dataDirectory: directory, embeddingProfile: "all-minilm-l6-v2" });
    expect(selected.status().embedding).toMatchObject({ profile: "all-minilm-l6-v2", model: "Xenova/all-MiniLM-L6-v2", dimensions: 384, pooling: "mean" });
    selected.close();

    const reopened = new DocumentationSearch({ dataDirectory: directory });
    expect(reopened.status().embedding).toMatchObject({ profile: "all-minilm-l6-v2", model: "Xenova/all-MiniLM-L6-v2", dimensions: 384 });
    reopened.close();
  });

  it("rejects a different same-dimension profile for an existing index", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-profile-mismatch-"));
    temporaryDirectories.push(directory);
    const selected = new DocumentationSearch({ dataDirectory: directory, embeddingProfile: "nomic-embed-text-v1" });
    selected.close();
    expect(() => new DocumentationSearch({ dataDirectory: directory, embeddingProfile: "nomic-embed-text-v1.5" })).toThrow("Index uses embedding profile nomic-embed-text-v1");
  });

  it("truncates a Matryoshka-capable profile to a custom dimension count and persists it across reopen", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-dimensions-"));
    temporaryDirectories.push(directory);
    const selected = new DocumentationSearch({ dataDirectory: directory, embeddingProfile: "nomic-embed-text-v1.5", embeddingDimensions: 256 });
    expect(selected.status().embedding).toMatchObject({ profile: "nomic-embed-text-v1.5", dimensions: 256 });
    selected.close();

    const reopened = new DocumentationSearch({ dataDirectory: directory });
    expect(reopened.status().embedding).toMatchObject({ profile: "nomic-embed-text-v1.5", dimensions: 256 });
    reopened.close();
  });

  it("rejects custom dimensions for a profile that doesn't support them", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-dimensions-unsupported-"));
    temporaryDirectories.push(directory);
    expect(() => new DocumentationSearch({ dataDirectory: directory, embeddingProfile: "all-minilm-l6-v2", embeddingDimensions: 200 })).toThrow("does not support custom dimensions");
  });

  it("rejects a custom dimension count below the profile's minimum", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-dimensions-too-small-"));
    temporaryDirectories.push(directory);
    expect(() => new DocumentationSearch({ dataDirectory: directory, embeddingProfile: "nomic-embed-text-v1.5", embeddingDimensions: 32 })).toThrow("--embedding-dimensions must be between 64 and 768");
  });

  it("remembers the embedding dtype of an index", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-dtype-stored-"));
    temporaryDirectories.push(directory);
    new DocumentationSearch({ dataDirectory: directory, embeddingDtype: "fp16" }).close();
    const reopened = new DocumentationSearch({ dataDirectory: directory });
    expect(reopened.status().embedding).toMatchObject({ dtype: "fp16", device: "cpu" });
    reopened.close();
    expect(() => new DocumentationSearch({ dataDirectory: directory, embeddingDtype: "q8" })).toThrow("Index uses embedding dtype fp16, but q8 was selected");
  });

  it("builds a new GPU index in fp16 and searches it on the CPU", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-dtype-gpu-"));
    temporaryDirectories.push(directory);
    const gpu = new DocumentationSearch({ dataDirectory: directory, embeddingDevice: "webgpu" });
    expect(gpu.status().embedding).toMatchObject({ dtype: "fp16", device: "webgpu" });
    gpu.close();
    const search = new DocumentationSearch({ dataDirectory: directory });
    expect(search.status().embedding).toMatchObject({ dtype: "fp16", device: "cpu" });
    search.close();
  });

  it("reads the dtype of an index built before dtypes were stored separately", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-dtype-legacy-"));
    temporaryDirectories.push(directory);
    const built = new DocumentationSearch({ dataDirectory: directory, embeddingDtype: "fp32" });
    built.database.setManifest({
      schemaVersion: SEARCH_SCHEMA_VERSION,
      family: "australia",
      branch: "australia",
      embeddingProfile: "all-minilm-l6-v2",
      embeddingProvider: built.embeddings.name,
      embeddingModel: built.embeddings.model,
      dimensions: built.embeddings.dimensions,
      pooling: built.embeddings.pooling ?? "mean",
      dtype: built.embeddings.dtype,
      layerNorm: built.embeddings.layerNorm,
      normalized: true,
      documentPrefix: built.embeddings.documentPrefix,
      queryPrefix: built.embeddings.queryPrefix,
      maxEmbeddingCharacters: built.embeddings.maxEmbeddingCharacters,
      updatedAt: new Date().toISOString(),
    });
    built.database.db.exec("DELETE FROM settings WHERE key = 'embedding_dtype'");
    built.close();
    const reopened = new DocumentationSearch({ dataDirectory: directory });
    expect(reopened.status().embedding.dtype).toBe("fp32");
    reopened.close();
  });

  it("rejects reopening an index with a different embedding dtype", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-dtype-mismatch-"));
    temporaryDirectories.push(directory);
    const built = new DocumentationSearch({ dataDirectory: directory, embeddingProvider: new DtypeEmbeddingProvider("q8") });
    built.database.setManifest({
      schemaVersion: SEARCH_SCHEMA_VERSION,
      family: "australia",
      branch: "australia",
      embeddingProvider: built.embeddings.name,
      embeddingModel: built.embeddings.model,
      dimensions: built.embeddings.dimensions,
      pooling: built.embeddings.pooling ?? "mean",
      dtype: built.embeddings.dtype,
      layerNorm: built.embeddings.layerNorm,
      normalized: true,
      documentPrefix: built.embeddings.documentPrefix,
      queryPrefix: built.embeddings.queryPrefix,
      maxEmbeddingCharacters: built.embeddings.maxEmbeddingCharacters,
      updatedAt: new Date().toISOString(),
    });
    built.close();

    expect(() => new DocumentationSearch({ dataDirectory: directory, embeddingProvider: new DtypeEmbeddingProvider("q4") })).toThrow(/dtype/);
  });

  it("indexes changed sources and skips unchanged sources", async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-e2e-"));
    temporaryDirectories.push(directory);
    const source = new FakeSource();
    const context = new DocumentationSearch({ dataDirectory: directory, embeddingProvider: new HashEmbeddingProvider(32), source });
    const first = await context.update({ family: "australia" });
    const second = await context.update({ family: "australia" });
    expect(first).toMatchObject({ added: 1, changed: 0, chunks: 2 });
    expect(second).toMatchObject({ added: 0, changed: 0, chunks: 0 });
    expect(source.downloads).toBe(1);
    expect((await context.search("restore service", { threshold: 0 }))[0]?.sourcePath).toBe("markdown/itsm/incidents.md");
    const [document] = context.getDocument("markdown/itsm/incidents.md", "australia");
    expect(document?.content).toContain("## Resolve incidents\nRestore service quickly.");
    expect(document?.content.length).toBeGreaterThan(document?.metadata.summary?.toString().length ?? 0);
    context.close();
  });

  it("stops an update when its signal is aborted, keeping what was already committed", async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-abort-"));
    temporaryDirectories.push(directory);
    const source = new FakeSource();
    const context = new DocumentationSearch({ dataDirectory: directory, embeddingProvider: new HashEmbeddingProvider(32), source });
    const controller = new AbortController();
    controller.abort(new Error("cancelled by test"));
    await expect(context.update({ family: "australia", signal: controller.signal })).rejects.toThrow("cancelled by test");
    expect(source.downloads).toBe(0);
    expect(context.status()).toMatchObject({ documents: 0 });
    await context.close();
  });

  it("closes once, releasing the embedding provider and reranker", async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-close-"));
    temporaryDirectories.push(directory);
    const disposed: string[] = [];
    const embeddings = Object.assign(new HashEmbeddingProvider(32), { dispose: async () => { disposed.push("embeddings"); } });
    const reranker = { model: "test-reranker", score: async () => [], dispose: async () => { disposed.push("reranker"); } };
    const context = new DocumentationSearch({ dataDirectory: directory, embeddingProvider: embeddings, reranker });
    await Promise.all([context.close(), context.close()]);
    expect(disposed.sort()).toEqual(["embeddings", "reranker"]);
  });

  it("commits successful documents and retries failed documents later", async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-partial-"));
    temporaryDirectories.push(directory);
    const source = new PartiallyFailingSource();
    const context = new DocumentationSearch({ dataDirectory: directory, embeddingProvider: new HashEmbeddingProvider(32), source });
    const first = await context.update({ family: "australia" });
    expect(first).toMatchObject({ added: 1, changed: 0, chunks: 1 });
    expect(first.failures).toEqual([{ sourcePath: "markdown/itsm/bad.md", stage: "download", error: "simulated download failure" }]);
    expect(context.status()).toMatchObject({ documents: 1, chunks: 1 });

    const second = await context.update({ family: "australia" });
    expect(second).toMatchObject({ added: 0, changed: 0, chunks: 0 });
    expect(second.failures).toHaveLength(1);
    expect(source.attempts.get("markdown/itsm/good.md")).toBe(1);
    expect(source.attempts.get("markdown/itsm/bad.md")).toBe(2);
    context.close();
  });

  it("commits each completed document while shared embedding is still running", async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-batch-"));
    temporaryDirectories.push(directory);
    const embeddings = new StreamingEmbeddingProvider(32);
    const context = new DocumentationSearch({ dataDirectory: directory, embeddingProvider: embeddings, source: new TwoDocumentSource() });
    const observedDocuments: number[] = [];
    const observedManifests: unknown[] = [];
    embeddings.afterBatch = () => {
      observedDocuments.push(context.status().documents);
      observedManifests.push(context.status().manifest);
    };
    const result = await context.update({ family: "australia" });
    expect(result).toMatchObject({ added: 2, chunks: 2, failures: [] });
    expect(embeddings.calls).toBe(1);
    expect(observedDocuments).toEqual([1, 2]);
    expect(observedManifests.every((manifest) => manifest !== null)).toBe(true);
    context.close();
  });

  it("reuses cached embeddings on refresh and after the index is removed", async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-cache-"));
    temporaryDirectories.push(directory);
    const embeddings = new CountingEmbeddingProvider();
    const context = new DocumentationSearch({ dataDirectory: directory, embeddingProvider: embeddings, embeddingCache: true, source: new FakeSource() });
    const first = await context.update({ family: "australia" });
    expect(first).toMatchObject({ added: 1, chunks: 2, cachedChunks: 0 });
    expect(embeddings.embeddedTexts).toBe(2);
    const before = await context.search("restore service", { threshold: 0 });

    const refreshed = await context.update({ family: "australia", refresh: true });
    expect(refreshed).toMatchObject({ changed: 1, chunks: 2, cachedChunks: 2 });
    expect(embeddings.embeddedTexts).toBe(2);
    expect(await context.search("restore service", { threshold: 0 })).toEqual(before);
    context.close();

    for (const suffix of ["", "-shm", "-wal"]) fs.rmSync(path.join(directory, `index.sqlite${suffix}`), { force: true });
    const rebuilt = new DocumentationSearch({ dataDirectory: directory, embeddingProvider: embeddings, embeddingCache: true, source: new FakeSource() });
    expect(await rebuilt.update({ family: "australia" })).toMatchObject({ added: 1, chunks: 2, cachedChunks: 2 });
    expect(embeddings.embeddedTexts).toBe(2);
    expect(await rebuilt.search("restore service", { threshold: 0 })).toEqual(before);
    rebuilt.close();
  });

  it("embeds only the changed passages of a partly cached document", async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-cache-partial-"));
    temporaryDirectories.push(directory);
    const embeddings = new CountingEmbeddingProvider();
    const documents = new Map([["markdown/itsm/incidents.md", INCIDENTS]]);
    const context = new DocumentationSearch({ dataDirectory: directory, embeddingProvider: embeddings, embeddingCache: true, source: new MutableSource(documents) });
    await context.update({ family: "australia" });
    documents.set("markdown/itsm/incidents.md", INCIDENTS.replace("Restore service quickly.", "Restore service quickly and record the cause."));
    const result = await context.update({ family: "australia" });
    expect(result).toMatchObject({ changed: 1, chunks: 2, cachedChunks: 1, failures: [] });
    expect(embeddings.embeddedTexts).toBe(3);
    expect(context.getDocumentOutline("markdown/itsm/incidents.md", "australia")).toHaveLength(2);
    context.close();
  });

  it("embeds a passage shared by several documents once", async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-dedupe-"));
    temporaryDirectories.push(directory);
    const embeddings = new CountingEmbeddingProvider();
    const documents = new Map([["markdown/itsm/incidents.md", INCIDENTS], ["markdown/itsm-copy/incidents.md", INCIDENTS]]);
    const context = new DocumentationSearch({ dataDirectory: directory, embeddingProvider: embeddings, source: new MutableSource(documents) });
    expect(await context.update({ family: "australia" })).toMatchObject({ added: 2, chunks: 4, cachedChunks: 0 });
    expect(embeddings.embeddedTexts).toBe(2);
    context.close();
  });

  it("leaves custom providers uncached unless asked", async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-cache-default-"));
    temporaryDirectories.push(directory);
    const context = new DocumentationSearch({ dataDirectory: directory, embeddingProvider: new HashEmbeddingProvider(32), source: new FakeSource() });
    await context.update({ family: "australia" });
    expect(context.status().embeddingCache).toMatchObject({ enabled: false, sizeBytes: 0 });
    expect(fs.existsSync(path.join(directory, "embedding-cache.sqlite"))).toBe(false);
    context.close();
  });

  it("indexes without the cache when its file cannot be used", async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-cache-broken-"));
    temporaryDirectories.push(directory);
    fs.mkdirSync(path.join(directory, "embedding-cache.sqlite"));
    const messages: string[] = [];
    const context = new DocumentationSearch({ dataDirectory: directory, embeddingProvider: new HashEmbeddingProvider(32), embeddingCache: true, source: new FakeSource() });
    const result = await context.update({ family: "australia", onProgress: (message) => messages.push(message) });
    expect(result).toMatchObject({ added: 1, chunks: 2, cachedChunks: 0, failures: [] });
    expect(messages.some((message) => message.startsWith("Embedding cache unavailable"))).toBe(true);
    context.close();
  });

  it("prefetches only the next batch while embedding the current one", async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-streaming-"));
    temporaryDirectories.push(directory);
    const source = new ManyDocumentSource();
    const embeddings = new StreamingEmbeddingProvider(32);
    const context = new DocumentationSearch({ dataDirectory: directory, embeddingProvider: embeddings, source });
    const downloadsAtFirstEmbedding: number[] = [];
    embeddings.afterBatch = () => {
      if (!downloadsAtFirstEmbedding.length) downloadsAtFirstEmbedding.push(source.downloads);
    };

    const result = await context.update({ family: "australia", concurrency: 8 });
    // Batch 1 (128 docs) must be fully parsed before its own embedding starts, and batch 2 (128 docs)
    // may already be prefetched in the background by then -- but batch 3 (the last document) must not be.
    expect(downloadsAtFirstEmbedding[0]).toBeGreaterThanOrEqual(128);
    expect(downloadsAtFirstEmbedding[0]).toBeLessThan(source.total);
    expect(result).toMatchObject({ added: source.total, chunks: source.total, failures: [] });
    context.close();
  });
});