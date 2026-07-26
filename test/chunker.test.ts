import { describe, expect, it } from "vitest";
import { chunkDocument } from "../src/chunker.js";

describe("chunkDocument", () => {
  it("creates focused method lookup chunks", () => {
    const markdown = `---
title: GlideRecord - Global
release: australia
classification: server-api-reference
---
# GlideRecord - Global

## GlideRecord - chooseWindow(Number firstRow, Number lastRow)

Sets a range of rows to be returned by subsequent queries.

|Name|Type|Description|
|----|----|-----------|
|firstRow|Number|Zero-based start row.|
|lastRow|Number|Zero-based end row.|

|Type|Description|
|----|-----------|
|void||

\`\`\`
gr.chooseWindow(0, 10);
\`\`\`
`;
    const chunks = chunkDocument("markdown/api-reference/server-api-reference/c_GlideRecordAPI.md", markdown, "australia", "australia");
    const methodChunks = chunks.filter((chunk) => chunk.methodName === "chooseWindow");
    expect(methodChunks.filter((chunk) => chunk.chunkType === "method")).toHaveLength(1);
    expect(methodChunks.filter((chunk) => chunk.chunkType === "parameter")).toHaveLength(2);
    expect(methodChunks.filter((chunk) => chunk.chunkType === "returns")).toHaveLength(1);
    expect(methodChunks.filter((chunk) => chunk.chunkType === "example")).toHaveLength(1);
    expect(methodChunks.find((chunk) => chunk.chunkType === "method")?.content).toContain("Summary: Sets a range");
  });

  it("recognizes deprecated API titles", () => {
    const markdown = `---
title: GlideEncrypter - Global \\(deprecated\\)
release: australia
---
# GlideEncrypter - Global \\(deprecated\\)

## GlideEncrypter - GlideEncrypter()

Creates an instance.
`;
    const method = chunkDocument("markdown/api-reference/server-api-reference/GlideEncrypterAPI.md", markdown, "australia", "australia").find((chunk) => chunk.chunkType === "method");
    expect(method?.metadata.deprecated).toBe(true);
  });

  it("splits topic documents at H2 headings", () => {
    const chunks = chunkDocument("markdown/it-service-management/incidents.md", "---\ntitle: Incidents\nproduct: ITSM\n---\n# Incidents\nIntro.\n## Create an incident\nSteps.", "australia", "australia");
    expect(chunks.map((chunk) => chunk.chunkType)).toEqual(["overview", "section"]);
    expect(chunks[1]?.heading).toBe("Create an incident");
  });

  it("tolerates unquoted colons in ServiceNow frontmatter", () => {
    const markdown = `---
title: Extensions to Jelly syntax
description: Apache's Jelly syntax explains three tags: <g:insert>, <g:inline>, and <g:call>.
breadcrumb: [Jelly tags, Scripting, API implementation]
---
# Extensions to Jelly syntax

Apache Jelly documentation.
`;
    const [chunk] = chunkDocument("markdown/api-reference/scripts/c_ExtensionsToJellySyntax.md", markdown, "australia", "australia");
    expect(chunk?.metadata.description).toContain("three tags: <g:insert>");
    expect(chunk?.metadata.breadcrumb).toEqual(["Jelly tags", "Scripting", "API implementation"]);
  });
});