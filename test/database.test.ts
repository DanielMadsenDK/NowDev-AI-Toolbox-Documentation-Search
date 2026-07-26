import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { chunkDocument } from "../src/chunker.js";
import { DocumentationSearchDatabase } from "../src/database.js";
import { HashEmbeddingProvider } from "../src/embedder.js";

const temporaryDirectories: string[] = [];
afterEach(() => temporaryDirectories.splice(0).forEach((directory) => fs.rmSync(directory, { recursive: true, force: true })));

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
});