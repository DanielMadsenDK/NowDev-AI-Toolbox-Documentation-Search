import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
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
  readonly total = 129;
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
    context.close();
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
    embeddings.afterBatch = () => observedDocuments.push(context.status().documents);
    const result = await context.update({ family: "australia" });
    expect(result).toMatchObject({ added: 2, chunks: 2, failures: [] });
    expect(embeddings.calls).toBe(1);
    expect(observedDocuments).toEqual([1, 2]);
    context.close();
  });

  it("starts embedding before every changed source is prepared", async () => {
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
    expect(downloadsAtFirstEmbedding).toEqual([128]);
    expect(result).toMatchObject({ added: 129, chunks: 129, failures: [] });
    context.close();
  });
});