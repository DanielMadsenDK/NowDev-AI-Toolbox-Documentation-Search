import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { chunkDocument } from "../src/chunker.js";
import { SEARCH_SCHEMA_VERSION } from "../src/config.js";
import { DocumentationSearchDatabase } from "../src/database.js";
import { HashEmbeddingProvider } from "../src/embedder.js";
import { removeDirectory } from "./helpers.js";

let dataDirectory: string;

function runCli(args: string[]) {
  return spawnSync(process.execPath, [path.resolve("dist/cli-entry.js"), ...args], { encoding: "utf8" });
}

beforeAll(async () => {
  execFileSync("npm", ["run", "build"], { stdio: "ignore", shell: process.platform === "win32" });
  dataDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-cli-"));
  const database = new DocumentationSearchDatabase(path.join(dataDirectory, "index.sqlite"), 64);
  const embedder = new HashEmbeddingProvider(64);
  const sourcePath = "markdown/api-reference/server-api-reference/GlideQuery.md";
  const [queryEmbedding] = await embedder.embed(["GlideQuery"]);
  for (const release of ["australia", "zurich"]) {
    const chunks = chunkDocument(sourcePath, "---\ntitle: GlideQuery\nclassification: server-api-reference\n---\n# GlideQuery\nQuery records with GlideQuery.", release, release);
    database.replaceSources(release, [{ path: sourcePath, blobSha: release, contentHash: chunks[0]!.contentHash, chunks, embeddings: chunks.map(() => queryEmbedding!) }], []);
  }
  database.close();
}, 60_000);

afterAll(() => removeDirectory(dataDirectory));

describe("nowdev-ai-toolbox-documentationsearch CLI", () => {
  it.each([
    ["search limit", ["search", "GlideQuery", "--limit", "nope"], "limit must be an integer"],
    ["search threshold", ["search", "GlideQuery", "--threshold", "nope"], "threshold must be a finite number"],
    ["embedding profile", ["--embedding-profile", "unsupported", "status"], "Allowed choices"],
    ["embedding dimensions", ["--embedding-dimensions", "nope", "status"], "must be an integer"],
    ["embedding endpoint concurrency", ["--embedding-endpoint-concurrency", "0", "status"], "must be between 1 and 32"],
    ["embedding endpoint timeout", ["--embedding-endpoint-timeout", "0", "status"], "must be between 1 and 600"],
    ["vector oversample", ["--vector-oversample", "0", "status"], "must be between 1 and 64"],
    ["embedding workers", ["--embedding-workers", "0", "status"], "must be between 1 and 64"],
    ["update concurrency", ["update", "--concurrency", "nope"], "concurrency must be an integer"],
    ["update limit", ["update", "--limit", "0"], "limit must be between 1 and 1000000"],
  ])("rejects malformed %s values as JSON", (_name, commandArguments, expected) => {
    const result = runCli(["--json", "--data-dir", dataDirectory, "--deterministic-embeddings", ...commandArguments]);
    expect(result.status).toBe(1);
    expect(JSON.parse(result.stderr)).toMatchObject({ error: expect.stringContaining(expected) });
  });

  it("applies threshold semantics through the CLI", () => {
    const result = runCli(["--json", "--data-dir", dataDirectory, "--deterministic-embeddings", "search", "GlideQuery", "--threshold", "0.99", "--limit", "10"]);
    expect(result.status).toBe(0);
    const rows = JSON.parse(result.stdout) as Array<{ similarity: number }>;
    expect(rows).toHaveLength(2);
    expect(rows.every((row) => row.similarity >= 0.99)).toBe(true);
  });

  it("accepts a precision-first binary vector oversampling setting", () => {
    const result = runCli(["--json", "--data-dir", dataDirectory, "--deterministic-embeddings", "--vector-oversample", "16", "search", "GlideQuery"]);
    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).not.toHaveLength(0);
  });

  it("supports combined filters and release deduplication", () => {
    const result = runCli([
      "--json", "--data-dir", dataDirectory, "--deterministic-embeddings", "search", "GlideQuery",
      "--doc-type", "scripting-api", "--publication", "api-reference", "--chunk-type", "overview",
      "--threshold", "0", "--deduplicate-releases", "--limit", "10",
    ]);
    expect(result.status).toBe(0);
    const rows = JSON.parse(result.stdout) as Array<{ sourcePath: string; docType: string; publication: string; chunkType: string }>;
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ sourcePath: "markdown/api-reference/server-api-reference/GlideQuery.md", docType: "scripting-api", publication: "api-reference", chunkType: "overview" });
  });

  it("returns runtime failures as JSON when requested", () => {
    const result = runCli(["--json", "--data-dir", dataDirectory, "status"]);
    expect(result.status).toBe(1);
    expect(JSON.parse(result.stderr)).toMatchObject({ error: expect.stringContaining("provider uses 384") });
  });

  it("saves index vectors to the embedding cache on reset-index, and clears the cache on request", async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-cli-reset-"));
    try {
      const database = new DocumentationSearchDatabase(path.join(directory, "index.sqlite"), 64);
      const embedder = new HashEmbeddingProvider(64);
      const sourcePath = "markdown/itsm/incidents.md";
      const chunks = chunkDocument(sourcePath, "---\ntitle: Incidents\n---\n# Incidents\nHandle interruptions.", "australia", "australia");
      database.replaceSources("australia", [{ path: sourcePath, blobSha: "sha", contentHash: "hash", chunks, embeddings: await embedder.embed(chunks.map((chunk) => chunk.content)) }], [], {
        schemaVersion: SEARCH_SCHEMA_VERSION, family: "australia", branch: "australia", embeddingProvider: embedder.name, embeddingModel: embedder.model,
        dimensions: embedder.dimensions, pooling: embedder.pooling, normalized: true, updatedAt: new Date().toISOString(),
      });
      database.close();
      const cachePath = path.join(directory, "embedding-cache.sqlite");

      const reset = runCli(["--json", "--data-dir", directory, "reset-index", "--yes"]);
      expect(reset.status).toBe(0);
      expect(JSON.parse(reset.stdout)).toMatchObject({ preserved: expect.arrayContaining([cachePath]), embeddingCache: { harvested: chunks.length, vectors: chunks.length } });
      expect(fs.existsSync(path.join(directory, "index.sqlite"))).toBe(false);
      expect(fs.existsSync(cachePath)).toBe(true);

      const cleared = runCli(["--json", "--data-dir", directory, "reset-index", "--yes", "--clear-embedding-cache"]);
      expect(cleared.status).toBe(0);
      expect(JSON.parse(cleared.stdout)).toMatchObject({ embeddingCache: { removed: cachePath } });
      expect(JSON.parse(cleared.stdout).preserved).not.toContain(cachePath);
      expect(fs.existsSync(cachePath)).toBe(false);
    } finally {
      await removeDirectory(directory);
    }
  });

  it("starts the MCP server from its dedicated entry point", async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-mcp-entry-"));
    const client = new Client({ name: "cli-test", version: "1.0.0" });
    try {
      await client.connect(new StdioClientTransport({
        command: process.execPath,
        args: [path.resolve("dist/mcp-entry.js")],
        env: { ...process.env, NOWDEV_AI_TOOLBOX_DOCUMENTATIONSEARCH_HOME: directory } as Record<string, string>,
        stderr: "ignore",
      }));
      expect((await client.listTools()).tools.map((tool) => tool.name)).toContain("search_servicenow_docs");
    } finally {
      await client.close();
      await removeDirectory(directory);
    }
  });

  it("requires endpoint URL and model together", () => {
    const result = runCli(["--json", "--data-dir", dataDirectory, "--embedding-endpoint", "https://openrouter.ai/api/v1/embeddings", "status"]);
    expect(result.status).toBe(1);
    expect(JSON.parse(result.stderr)).toMatchObject({ error: expect.stringContaining("must be specified together") });
  });

  it("reads endpoint credentials only from the configured environment variable", () => {
    const result = runCli([
      "--json", "--data-dir", dataDirectory,
      "--embedding-endpoint", "https://openrouter.ai/api/v1/embeddings",
      "--embedding-endpoint-model", "baai/bge-base-en-v1.5",
      "--embedding-api-key-env", "MISSING_EMBEDDING_TEST_KEY",
      "status",
    ]);
    expect(result.status).toBe(1);
    expect(JSON.parse(result.stderr)).toMatchObject({ error: expect.stringContaining("MISSING_EMBEDDING_TEST_KEY is not set") });
  });
});