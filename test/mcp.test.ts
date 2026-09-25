import { describe, expect, it } from "vitest";
import { documentUri, formatSearchResults, MCP_TOOL_DESCRIPTIONS } from "../src/mcp.js";
import type { SearchResult } from "../src/types.js";

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

