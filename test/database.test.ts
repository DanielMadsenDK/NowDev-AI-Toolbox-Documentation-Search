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
    expect(database.getSourceContent("markdown/itsm/incidents.md", "australia")).toContain("Resolve service interruptions");
    const storedMetadata = database.db.prepare("SELECT metadata FROM documents ORDER BY id").all() as Array<{ metadata: string }>;
    expect(storedMetadata.every((row) => !row.metadata.includes("full_content") && !row.metadata.includes("section_content"))).toBe(true);
    expect(database.db.prepare("SELECT value FROM settings WHERE key = 'database_schema_version'").get()).toEqual({ value: "3" });
    database.close();
  });

  it("allows keyword candidates below the semantic threshold", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-threshold-"));
    temporaryDirectories.push(directory);
    const database = new DocumentationSearchDatabase(path.join(directory, "index.sqlite"), 2);
    const [chunk] = chunkDocument("markdown/itsm/incidents.md", "---\ntitle: Incidents\n---\n# Incidents\nIncident response.", "australia", "australia");
    database.replaceSources("australia", [{ path: chunk!.sourcePath, blobSha: "one", contentHash: chunk!.contentHash, chunks: [chunk!], embeddings: [Float32Array.from([1, 0])] }], []);
    expect(database.search("incident", Float32Array.from([0, 1]), {}, 10, 0.99)[0]?.sourcePath).toBe(chunk!.sourcePath);
    database.close();
  });

  it("uses binary coarse vectors and float re-scoring when dimensions support quantization", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-binary-vectors-"));
    temporaryDirectories.push(directory);
    const database = new DocumentationSearchDatabase(path.join(directory, "index.sqlite"), 8);
    const [first] = chunkDocument("markdown/guides/first.md", "# First\nNearest content.", "australia", "australia");
    const [second] = chunkDocument("markdown/guides/second.md", "# Second\nOther content.", "australia", "australia");
    database.replaceSources("australia", [
      { path: first!.sourcePath, blobSha: "first", contentHash: first!.contentHash, chunks: [first!], embeddings: [Float32Array.from([1, 0, 0, 0, 0, 0, 0, 0])] },
      { path: second!.sourcePath, blobSha: "second", contentHash: second!.contentHash, chunks: [second!], embeddings: [Float32Array.from([0, 1, 0, 0, 0, 0, 0, 0])] },
    ], []);

    const results = database.search("absent-keyword", Float32Array.from([1, 0, 0, 0, 0, 0, 0, 0]), { release: "australia" }, 1, -1);
    expect(database.db.prepare("SELECT sql FROM sqlite_master WHERE name = 'document_vectors'").get()).toEqual(expect.objectContaining({ sql: expect.stringContaining("coarse_embedding bit[8]") }));
    expect(results[0]?.sourcePath).toBe(first!.sourcePath);
    expect(results[0]?.similarity).toBeCloseTo(1, 5);
    database.close();
  });

  it("rejects unsupported binary vector oversampling values", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-vector-oversample-"));
    temporaryDirectories.push(directory);
    expect(() => new DocumentationSearchDatabase(path.join(directory, "index.sqlite"), 8, 0)).toThrow("coarseVectorOversample must be an integer between 1 and 64");
  });

  it("preserves structured API metadata while omitting large body fields", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-metadata-"));
    temporaryDirectories.push(directory);
    const database = new DocumentationSearchDatabase(path.join(directory, "index.sqlite"), 2);
    const method = chunkDocument("markdown/api-reference/server-api-reference/c_ExampleAPI.md", `# Example
## Example - run(String value)
Runs an example.
|Name|Type|Description|
|----|----|-----------|
|value|String|Input value.|
\`\`\`
run("value");
\`\`\``, "australia", "australia").find((chunk) => chunk.chunkType === "method")!;
    database.replaceSources("australia", [{ path: method.sourcePath, blobSha: "one", contentHash: method.contentHash, chunks: [method], embeddings: [Float32Array.from([1, 0])] }], []);
    const [stored] = database.getDocument(method.sourcePath, "australia");
    expect(stored?.metadata.parameters).toEqual(method.metadata.parameters);
    expect(stored?.metadata.examples).toEqual(method.metadata.examples);
    expect(stored?.metadata).not.toHaveProperty("full_content");
    database.close();
  });

  it("replaces all source vectors and documents as a set", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-replacement-"));
    temporaryDirectories.push(directory);
    const database = new DocumentationSearchDatabase(path.join(directory, "index.sqlite"), 2);
    const sourcePath = "markdown/guides/replaced.md";
    const first = chunkDocument(sourcePath, "---\ntitle: First\n---\n# First\nOld content.\n## Old section\nOld detail.", "australia", "australia");
    const second = chunkDocument(sourcePath, "---\ntitle: Second\n---\n# Second\nNew content.", "australia", "australia");
    database.replaceSources("australia", [{ path: sourcePath, blobSha: "one", contentHash: first[0]!.contentHash, chunks: first, embeddings: first.map(() => Float32Array.from([1, 0])) }], []);
    database.replaceSources("australia", [{ path: sourcePath, blobSha: "two", contentHash: second[0]!.contentHash, chunks: second, embeddings: second.map(() => Float32Array.from([0, 1])) }], []);
    expect(database.getDocument(sourcePath, "australia").map((chunk) => chunk.title)).toEqual(["Second"]);
    expect(database.stats()).toMatchObject({ documents: 1, chunks: 1 });
    database.close();
  });

  it("rejects chunks whose release or source path differs from their source", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-identity-"));
    temporaryDirectories.push(directory);
    const database = new DocumentationSearchDatabase(path.join(directory, "index.sqlite"), 2);
    const [chunk] = chunkDocument("markdown/guides/example.md", "# Example\nContent.", "australia", "australia");
    expect(() => database.replaceSources("zurich", [{ path: chunk!.sourcePath, blobSha: "one", contentHash: chunk!.contentHash, chunks: [chunk!], embeddings: [Float32Array.from([1, 0])] }], [])).toThrow("identity mismatch");
    expect(database.stats()).toMatchObject({ documents: 0, chunks: 0 });
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

  it("supports every metadata prefilter combination", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-filter-combinations-"));
    temporaryDirectories.push(directory);
    const database = new DocumentationSearchDatabase(path.join(directory, "index.sqlite"), 2);
    const [chunk] = chunkDocument("markdown/guides/task.md", "---\ntitle: Task guide\ntopic_type: task\n---\n# Task guide\nPerform a task.", "australia", "australia");
    database.replaceSources("australia", [{ path: chunk!.sourcePath, blobSha: "task", contentHash: chunk!.contentHash, chunks: [chunk!], embeddings: [Float32Array.from([1, 0])] }], []);
    const entries = [
      ["docType", chunk!.docType],
      ["publication", chunk!.publication],
      ["chunkType", chunk!.chunkType],
      ["topicType", chunk!.topicType!],
    ] as const;

    for (let mask = 0; mask < 16; mask += 1) {
      const filters = Object.fromEntries(entries.filter((_, index) => mask & (1 << index)));
      expect(database.search("absent-keyword", Float32Array.from([1, 0]), { release: "australia", ...filters }, 1, -1)[0]?.sourcePath).toBe(chunk!.sourcePath);
    }
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

  it("stores shared frontmatter once per source and merges it back into results", async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-metadata-"));
    temporaryDirectories.push(directory);
    const embedder = new HashEmbeddingProvider(16);
    const database = new DocumentationSearchDatabase(path.join(directory, "index.sqlite"), embedder.dimensions);
    const markdown = "---\ntitle: Data policies\nbreadcrumb: [Data policy, Administer]\ncanonical_url: https://example.test/data-policy.html\n---\n# Data policies\nEnforce data rules.\n## Create a data policy\nSteps to create one.\n### Add rules\nRule steps.";
    const chunks = chunkDocument("markdown/platform-administration/data-policy.md", markdown, "australia", "australia");
    database.replaceSources("australia", [{ path: chunks[0]!.sourcePath, blobSha: "one", contentHash: chunks[0]!.contentHash, chunks, embeddings: await embedder.embed(chunks.map((chunk) => chunk.content)) }], []);
    const stored = database.db.prepare("SELECT metadata FROM documents ORDER BY chunk_index").all() as Array<{ metadata: string }>;
    expect(stored.every((row) => !row.metadata.includes("canonical_url"))).toBe(true);
    expect(JSON.parse(stored.at(-1)!.metadata)).toEqual({ heading_path: "Create a data policy > Add rules" });
    expect(JSON.parse((database.db.prepare("SELECT metadata FROM sources").get() as { metadata: string }).metadata)).toMatchObject({ canonical_url: "https://example.test/data-policy.html", breadcrumb: ["Data policy", "Administer"] });
    const [queryVector] = await embedder.embed(["add rules"]);
    const [result] = database.search("add rules", queryVector!, {}, 1, -1);
    expect(result?.metadata).toMatchObject({ canonical_url: "https://example.test/data-policy.html", heading_path: "Create a data policy > Add rules" });
    expect(database.getDocument("markdown/platform-administration/data-policy.md").every((chunk) => chunk.metadata.canonical_url === "https://example.test/data-policy.html")).toBe(true);
    database.close();
  });

  it("collapses identical passages published under several publications", async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-fingerprint-"));
    temporaryDirectories.push(directory);
    const embedder = new HashEmbeddingProvider(16);
    const database = new DocumentationSearchDatabase(path.join(directory, "index.sqlite"), embedder.dimensions);
    const page = (breadcrumb: string) => `---\ntitle: Classic business rules\nbreadcrumb: [${breadcrumb}]\n---\n# Classic business rules\nBusiness rules run server-side scripts when records change.`;
    const sources = ["markdown/api-reference/business-rules/c_BusinessRules.md", "markdown/build-workflows/business-rules/c_BusinessRules.md"].map((sourcePath, index) => {
      const chunks = chunkDocument(sourcePath, page(index ? "Build workflows" : "API reference"), "australia", "australia");
      return { path: sourcePath, blobSha: String(index), contentHash: chunks[0]!.contentHash, chunks };
    });
    database.replaceSources("australia", await Promise.all(sources.map(async (source) => ({ ...source, embeddings: await embedder.embed(source.chunks.map((chunk) => chunk.content)) }))), []);
    const [queryVector] = await embedder.embed(["business rules"]);
    expect(database.search("business rules", queryVector!, {}, 10, -1)).toHaveLength(1);
    database.close();
  });

  it("matches inflected forms through porter stemming", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-porter-"));
    temporaryDirectories.push(directory);
    const database = new DocumentationSearchDatabase(path.join(directory, "index.sqlite"), 2);
    const [chunk] = chunkDocument("markdown/guides/schedules.md", "---\ntitle: Scheduling\n---\n# Scheduling\nThe job runs nightly.", "australia", "australia");
    database.replaceSources("australia", [{ path: chunk!.sourcePath, blobSha: "one", contentHash: chunk!.contentHash, chunks: [chunk!], embeddings: [Float32Array.from([1, 0])] }], []);
    expect(database.search("run nightly jobs", Float32Array.from([0, 1]), {}, 5, 0.99)[0]?.sourcePath).toBe(chunk!.sourcePath);
    database.close();
  });

  it("returns the same semantic candidates from the in-memory binary index as from vec0", async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-memory-index-"));
    temporaryDirectories.push(directory);
    const embedder = new HashEmbeddingProvider(32);
    const database = new DocumentationSearchDatabase(path.join(directory, "index.sqlite"), embedder.dimensions);
    const topics = ["incident assignment groups", "change approval policies", "knowledge article feedback", "service catalog variables", "discovery schedules and probes", "update set merging"];
    const sources = await Promise.all(topics.map(async (topic, index) => {
      const chunks = chunkDocument(`markdown/${index % 2 ? "itsm" : "platform"}/topic-${index}.md`, `---\ntitle: ${topic}\n---\n# ${topic}\nDetails about ${topic}.`, "australia", "australia");
      return { path: chunks[0]!.sourcePath, blobSha: String(index), contentHash: chunks[0]!.contentHash, chunks, embeddings: await embedder.embed(chunks.map((chunk) => chunk.content)) };
    }));
    database.replaceSources("australia", sources, []);
    const [queryVector] = await embedder.embed(["knowledge article feedback"]);
    const search = (filters = {}) => database.search("absent", queryVector!, filters, 5, -1).map((result) => result.sourcePath);
    const fromVec0 = search();
    const filteredFromVec0 = search({ publication: "itsm" });
    await database.warmUp();
    expect(database.vectorIndexLoaded).toBe(true);
    expect(search()).toEqual(fromVec0);
    expect(search({ publication: "itsm" })).toEqual(filteredFromVec0);
    expect(search({ publication: "missing" })).toEqual([]);
    database.replaceSources("australia", [], [sources[0]!.path]);
    expect(database.vectorIndexLoaded).toBe(false);
    database.close();
  });
});

