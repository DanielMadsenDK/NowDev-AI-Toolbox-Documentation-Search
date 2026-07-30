import { describe, expect, it } from "vitest";
import { formatSearchResults } from "../src/mcp.js";
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
});