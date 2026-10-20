import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { afterEach, describe, expect, it } from "vitest";
import { HashEmbeddingProvider } from "../src/embedder.js";
import { GitHubDocumentationSource, type SourceEntry, type SourceTree } from "../src/github.js";
import { createMcpServer, documentUri, formatSearchResults, MCP_SERVER_INSTRUCTIONS, MCP_TOOL_DESCRIPTIONS } from "../src/mcp.js";
import { DocumentationSearch } from "../src/service-context.js";
import type { SearchResult } from "../src/types.js";
import { removeDirectory } from "./helpers.js";

const searchResult: SearchResult = {
  id: 1,
  docType: "scripting-api",
  publication: "api-reference",
  sourcePath: "markdown/api-reference/GlideRecord.md",
  release: "australia",
  chunkType: "method",
  chunkIndex: 4,
  title: "GlideRecord",
  heading: "query()",
  content: "Runs a query.",
  topicType: null,
  product: null,
  classification: null,
  lastUpdated: null,
  objectName: "GlideRecord",
  methodName: "query",
  metadata: {
    url: "https://example.test/GlideRecord.md",
    parameters: [{ name: "field" }],
    examples: ["var record = new GlideRecord('incident');"],
  },
  contentHash: "hash",
  similarity: 0.9,
  score: 0.02,
};

describe("MCP search results", () => {
  it("returns compact agent-oriented hits by default", () => {
    const [result] = formatSearchResults([searchResult], false);

    expect(result).toMatchObject({ sourceUrl: searchResult.metadata.url, sourcePath: searchResult.sourcePath });
    expect(result).not.toHaveProperty("metadata");
    expect(result).not.toHaveProperty("contentHash");
  });

  it("preserves detailed metadata when explicitly requested", () => {
    expect(formatSearchResults([searchResult], true)).toEqual([searchResult]);
  });

  it("describes the recommended search, retrieval, and update workflow", () => {
    expect(MCP_TOOL_DESCRIPTIONS.search).toContain("original wording");
    expect(MCP_TOOL_DESCRIPTIONS.search).toContain("before applying");
    // Everyday wording misses API methods and features described in the documentation's own vocabulary.
    expect(MCP_TOOL_DESCRIPTIONS.search).toContain("GlideRecord setWorkflow");
    expect(MCP_TOOL_DESCRIPTIONS.search).toContain("canonical ServiceNow name");
    expect(MCP_TOOL_DESCRIPTIONS.outline).toContain("Prefer this");
    expect(MCP_TOOL_DESCRIPTIONS.document).toContain("full documents can be large");
    expect(MCP_TOOL_DESCRIPTIONS.status).toContain("Call before");
    expect(MCP_TOOL_DESCRIPTIONS.update).toContain("mutates the local index");
  });

  it("prefers the public canonical URL and adds a readable breadcrumb", () => {
    const [result] = formatSearchResults([{
      ...searchResult,
      context: "Scripting > GlideRecord",
      metadata: { ...searchResult.metadata, canonical_url: "https://www.servicenow.com/docs/r/api-reference/c\\_GlideRecordAPI.html", breadcrumb: ["Server API", "API reference"] },
    }], false);
    expect(result).toMatchObject({ sourceUrl: "https://www.servicenow.com/docs/r/api-reference/c_GlideRecordAPI.html", breadcrumb: "API reference > Server API" });
    expect(result).not.toHaveProperty("context");
  });

  it("builds document resource URIs that round-trip nested source paths", () => {
    expect(documentUri("australia", "markdown/api-reference/c_GlideRecord API.md")).toBe("servicenow-doc://australia/markdown/api-reference/c_GlideRecord%20API.md");
  });
});

class SlowSource extends GitHubDocumentationSource {
  release?: () => void;
  private readonly gate = new Promise<void>((resolve) => { this.release = resolve; });

  override async discover(): Promise<SourceTree> {
    return { commit: "commit-1", entries: [{ path: "markdown/itsm/incidents.md", blobSha: "sha-1" }] };
  }

  override async download(_branch: string, _entry: SourceEntry): Promise<string> {
    await this.gate;
    return "---\ntitle: Incident management\n---\n# Incident management\nHandle interruptions.";
  }
}

describe("MCP server", () => {
  const cleanup: Array<() => Promise<void>> = [];
  afterEach(async () => {
    for (const step of cleanup.splice(0).reverse()) await step();
  });

  async function connect() {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "documentationsearch-mcp-"));
    const source = new SlowSource();
    const context = new DocumentationSearch({ dataDirectory: directory, embeddingProvider: new HashEmbeddingProvider(64), source });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const server = createMcpServer(context);
    const client = new Client({ name: "mcp-test", version: "1.0.0" });
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
    cleanup.push(async () => {
      await client.close();
      await server.close();
      await context.close();
      await removeDirectory(directory);
    });
    return { client, source };
  }

  it("serves instructions, searches with structured results, and reports missing documents as tool errors", async () => {
    const { client, source } = await connect();
    expect(client.getInstructions()).toBe(MCP_SERVER_INSTRUCTIONS);
    source.release!();
    await client.callTool({ name: "update_servicenow_docs", arguments: {} });

    const search = await client.callTool({ name: "search_servicenow_docs", arguments: { query: "incident interruptions", threshold: 0 } });
    expect((search.structuredContent as { results: SearchResult[] }).results[0]).toMatchObject({ sourcePath: "markdown/itsm/incidents.md", release: "australia" });
    expect(search.content).toContainEqual(expect.objectContaining({ type: "resource_link", uri: documentUri("australia", "markdown/itsm/incidents.md") }));

    const missing = await client.callTool({ name: "get_servicenow_document", arguments: { sourcePath: "markdown/missing.md" } });
    expect(missing.isError).toBe(true);
    expect(missing.content).toEqual([expect.objectContaining({ text: expect.stringContaining("No indexed document markdown/missing.md") })]);
    expect((await client.callTool({ name: "get_servicenow_document_outline", arguments: { sourcePath: "markdown/missing.md" } })).isError).toBe(true);
  });

  it("rejects a second update while one is running", async () => {
    const { client, source } = await connect();
    const first = client.callTool({ name: "update_servicenow_docs", arguments: {} });
    // Lets the first call reach the server and start downloading before the second arrives.
    await new Promise((resolve) => setTimeout(resolve, 20));
    const second = await client.callTool({ name: "update_servicenow_docs", arguments: {} });
    expect(second.isError).toBe(true);
    expect(second.content).toEqual([expect.objectContaining({ text: expect.stringContaining("already running") })]);
    source.release!();
    expect((await first).isError).toBeFalsy();
  });

  it("annotates the update tool as destructive and not idempotent", async () => {
    const { client } = await connect();
    const update = (await client.listTools()).tools.find((tool) => tool.name === "update_servicenow_docs");
    expect(update?.annotations).toMatchObject({ readOnlyHint: false, destructiveHint: true, idempotentHint: false });
  });
});
