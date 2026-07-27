import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { chunkDocument } from "../src/chunker.js";
import { DocumentationSearchDatabase } from "../src/database.js";
import { HashEmbeddingProvider } from "../src/embedder.js";

let dataDirectory: string;

function runCli(args: string[]) {
  return spawnSync(process.execPath, [path.resolve("dist/cli.js"), ...args], { encoding: "utf8" });
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

afterAll(() => fs.rmSync(dataDirectory, { recursive: true, force: true }));

describe("nowdev-ai-toolbox-documentationsearch CLI", () => {
  it.each([
    ["search limit", ["search", "GlideQuery", "--limit", "nope"], "limit must be an integer"],
    ["search threshold", ["search", "GlideQuery", "--threshold", "nope"], "threshold must be a finite number"],
    ["embedding profile", ["--embedding-profile", "unsupported", "status"], "Allowed choices"],
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
});