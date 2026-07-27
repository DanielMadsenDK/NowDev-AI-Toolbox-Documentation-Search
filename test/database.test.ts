import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { chunkDocument } from "../src/chunker.js";
import { DocumentationSearchDatabase } from "../src/database.js";
import { HashEmbeddingProvider } from "../src/embedder.js";
import { removeDirectory } from "./helpers.js";

const temporaryDirectories: string[] = [];
afterEach(async () => {
  for (const directory of temporaryDirectories.splice(0)) await removeDirectory(directory);
});

describe("DocumentationSearchDatabase", () => {
  it("stores and searches chunks with hybrid ranking", async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-"));
    temporaryDirectories.push(directory);
    const embedder = new HashEmbeddingProvider(32);
    const database = new DocumentationSearchDatabase(path.join(directory, "index.sqlite"), embedder.dimensions);
    const markdown = "---\ntitle: Incident management\nproduct: ITSM\n---\n# Incident management\nResolve service interruptions.\n## Assign an incident\nAssign incidents to support groups.";
    const chunks = chunkDocument("markdown/itsm/incidents.md", markdown, "australia", "australia");
    database.replaceSources("australia", [{ path: "markdown/itsm/incidents.md", blobSha: "abc", contentHash: chunks[0]!.contentHash, chunks, embeddings: await embedder.embed(chunks.map((chunk) => chunk.content)) }], []);
    const [query] = await embedder.embed(["assign incident support group"]);
    const results = database.search("assign incident support group", query!, { release: "australia" }, 5, 0);
    expect(results[0]?.sourcePath).toBe("markdown/itsm/incidents.md");
    expect(results[0]?.metadata).not.toHaveProperty("full_content");
    expect(results[0]?.metadata).not.toHaveProperty("section_content");
    const [exactVector] = await embedder.embed([chunks[1]!.content]);
    const semanticOnly = database.search("keyword-that-does-not-exist", exactVector!, {}, 1, 0.99);
    expect(semanticOnly[0]?.similarity).toBeCloseTo(1, 5);
    expect(database.stats()).toEqual({ documents: 1, chunks: 2, releases: ["australia"] });
    database.close();
  });

  it("applies the cosine threshold to keyword candidates", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-threshold-"));
    temporaryDirectories.push(directory);
    const database = new DocumentationSearchDatabase(path.join(directory, "index.sqlite"), 2);
    const [chunk] = chunkDocument("markdown/itsm/incidents.md", "---\ntitle: Incidents\n---\n# Incidents\nIncident response.", "australia", "australia");
    database.replaceSources("australia", [{ path: chunk!.sourcePath, blobSha: "one", contentHash: chunk!.contentHash, chunks: [chunk!], embeddings: [Float32Array.from([1, 0])] }], []);
    expect(database.search("incident", Float32Array.from([0, 1]), {}, 10, 0.99)).toEqual([]);
    database.close();
  });

  it("preserves exact split API identifier matches below the cosine threshold", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-identifier-threshold-"));
    temporaryDirectories.push(directory);
    const database = new DocumentationSearchDatabase(path.join(directory, "index.sqlite"), 2);
    const method = chunkDocument("markdown/api-reference/server-api-reference/gliderecord.md", "---\ntitle: GlideRecord\n---\n# GlideRecord\n## GlideRecord - setWorkflow(Boolean enable)\nControls processing.", "australia", "australia").find((chunk) => chunk.chunkType === "method")!;
    database.replaceSources("australia", [{ path: method.sourcePath, blobSha: "method", contentHash: method.contentHash, chunks: [method], embeddings: [Float32Array.from([1, 0])] }], []);

    const results = database.search("set workflow", Float32Array.from([0, 1]), {}, 1, 0.99);
    expect(results[0]?.methodName).toBe("setWorkflow");
    expect(results[0]?.similarity).toBeCloseTo(0, 5);
    database.close();
  });

  it("deduplicates matching chunks across releases after ranking", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-dedup-"));
    temporaryDirectories.push(directory);
    const database = new DocumentationSearchDatabase(path.join(directory, "index.sqlite"), 2);
    const sourcePath = "markdown/itsm/incidents.md";
    for (const release of ["australia", "zurich"]) {
      const [chunk] = chunkDocument(sourcePath, "---\ntitle: Incidents\n---\n# Incidents\nIncident response.", release, release);
      database.replaceSources(release, [{ path: sourcePath, blobSha: release, contentHash: chunk!.contentHash, chunks: [chunk!], embeddings: [Float32Array.from([1, 0])] }], []);
    }
    const query = Float32Array.from([1, 0]);
    expect(database.search("incident", query, {}, 10, 0.9, false)).toHaveLength(2);
    expect(database.search("incident", query, {}, 10, 0.9, true)).toHaveLength(1);
    database.close();
  });

  it("prioritizes exact API identifiers and diversifies sources", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-ranking-"));
    temporaryDirectories.push(directory);
    const database = new DocumentationSearchDatabase(path.join(directory, "index.sqlite"), 2);
    const apiMarkdown = "---\ntitle: GlideRecord\n---\n# GlideRecord\n## GlideRecord - getValue(fieldName)\nReturns a field value.\n## GlideRecord - setValue(fieldName, value)\nSets a field value.";
    const firstChunks = chunkDocument("markdown/api-reference/server-api-reference/gliderecord.md", apiMarkdown, "australia", "australia");
    const secondChunks = chunkDocument("markdown/api-reference/server-api-reference/other.md", apiMarkdown.replaceAll("GlideRecord", "OtherRecord").replaceAll("getValue", "setValue"), "australia", "australia");
    database.replaceSources("australia", [
      { path: firstChunks[0]!.sourcePath, blobSha: "first", contentHash: firstChunks[0]!.contentHash, chunks: firstChunks, embeddings: firstChunks.map(() => Float32Array.from([1, 0])) },
      { path: secondChunks[0]!.sourcePath, blobSha: "second", contentHash: secondChunks[0]!.contentHash, chunks: secondChunks, embeddings: secondChunks.map(() => Float32Array.from([1, 0])) },
    ], []);
    const results = database.search("getValue", Float32Array.from([1, 0]), { release: "australia" }, 4, 0, false, 1);
    expect(results).toHaveLength(2);
    expect(results[0]?.sourcePath).toBe(firstChunks[0]!.sourcePath);
    expect(results[0]?.methodName).toBe("getValue");
    expect(new Set(results.map((result) => result.sourcePath))).toHaveLength(2);
    database.close();
  });

  it("applies metadata filters before selecting vector candidates", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-prefilter-"));
    temporaryDirectories.push(directory);
    const database = new DocumentationSearchDatabase(path.join(directory, "index.sqlite"), 2);
    const irrelevantSources = Array.from({ length: 101 }, (_, index) => {
      const [chunk] = chunkDocument(`markdown/guides/irrelevant-${index}.md`, `---\ntitle: Irrelevant ${index}\n---\n# Irrelevant ${index}\nUnrelated content.`, "australia", "australia");
      return { path: chunk!.sourcePath, blobSha: `irrelevant-${index}`, contentHash: chunk!.contentHash, chunks: [chunk!], embeddings: [Float32Array.from([1, 0])] };
    });
    const [target] = chunkDocument("markdown/api-reference/server-api-reference/target.md", "---\ntitle: TargetApi\n---\n# TargetApi\n## TargetApi - targetMethod(String value)\nPerforms an operation.\n\n|Name|Type|Description|\n|---|---|---|\n|value|String|Input.|", "australia", "australia").filter((chunk) => chunk.chunkType === "parameter");
    database.replaceSources("australia", [
      ...irrelevantSources,
      { path: target!.sourcePath, blobSha: "target", contentHash: target!.contentHash, chunks: [target!], embeddings: [Float32Array.from([0, 1])] },
    ], []);

    const results = database.search("absent-keyword", Float32Array.from([1, 0]), { release: "australia", chunkType: "parameter" }, 1, -1);
    expect(results[0]?.sourcePath).toBe(target!.sourcePath);
    expect(results[0]?.chunkType).toBe("parameter");
    database.close();
  });

  it("uses configured BM25 column weights and preserves results after optimization", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-bm25-"));
    temporaryDirectories.push(directory);
    const database = new DocumentationSearchDatabase(path.join(directory, "index.sqlite"), 2);
    const [bodyMatch] = chunkDocument("markdown/guides/body.md", "---\ntitle: General guide\n---\n# General guide\nNeedle needle needle needle needle.", "australia", "australia");
    const [titleMatch] = chunkDocument("markdown/guides/title.md", "---\ntitle: Needle\n---\n# Needle\nUnrelated material.", "australia", "australia");
    database.replaceSources("australia", [
      { path: bodyMatch!.sourcePath, blobSha: "body", contentHash: bodyMatch!.contentHash, chunks: [bodyMatch!], embeddings: [Float32Array.from([1, 0])] },
      { path: titleMatch!.sourcePath, blobSha: "title", contentHash: titleMatch!.contentHash, chunks: [titleMatch!], embeddings: [Float32Array.from([1, 0])] },
    ], []);

    const search = () => database.search("needle", Float32Array.from([1, 0]), {}, 2, -1, false, 1);
    expect(search()[0]?.sourcePath).toBe(titleMatch!.sourcePath);
    database.optimizeSearchIndex();
    expect(search()[0]?.sourcePath).toBe(titleMatch!.sourcePath);
    database.close();
  });
});