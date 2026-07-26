import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { HashEmbeddingProvider } from "../src/embedder.js";
import { GitHubDocumentationSource, type DocumentationArea, type SourceEntry, type SourceTree } from "../src/github.js";
import { DocumentationSearch } from "../src/service-context.js";

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

class CountingEmbeddingProvider extends HashEmbeddingProvider {
  calls = 0;

  override async embed(texts: string[]): Promise<Float32Array[]> {
    this.calls += 1;
    return super.embed(texts);
  }
}

const temporaryDirectories: string[] = [];
afterEach(() => temporaryDirectories.splice(0).forEach((directory) => fs.rmSync(directory, { recursive: true, force: true })));

describe("ServiceContext", () => {
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

  it("embeds chunks from multiple documents in one shared call", async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-batch-"));
    temporaryDirectories.push(directory);
    const embeddings = new CountingEmbeddingProvider(32);
    const context = new DocumentationSearch({ dataDirectory: directory, embeddingProvider: embeddings, source: new TwoDocumentSource() });
    const result = await context.update({ family: "australia" });
    expect(result).toMatchObject({ added: 2, chunks: 2, failures: [] });
    expect(embeddings.calls).toBe(1);
    context.close();
  });
});