import { describe, expect, it } from "vitest";
import { chunkDocument, MAX_CHUNK_CHARACTERS } from "../src/chunker.js";

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

  it("extracts parameters and returns from ServiceNow's HTML table fallback", () => {
    // ServiceNow renders parameter tables as raw HTML instead of GFM pipe tables whenever a
    // cell holds a list or multiple paragraphs (e.g. GlideRecord.addQuery's operator list).
    const markdown = `---
title: GlideRecord - Global
release: australia
classification: server-api-reference
---
# GlideRecord - Global

## GlideRecord - addQuery\\(String name, Object operator, Object value\\)

Provides the ability to build a request.

<table id="table_ix2_hvp_dt" class="parameters"><thead><tr><th>

Name

</th><th>

Type

</th><th>

Description

</th></tr></thead><tbody><tr><td>

name

</td><td>

String

</td><td>

Table field name.

</td></tr><tr><td>

operator

</td><td>

Object

</td><td>

Query operator. Numbers:

-   =
-   &gt;
-   &lt;

</td></tr></tbody>
</table>|Type|Description|
|----|-----------|
|GlideQueryCondition|Reference to the added condition.|

\`\`\`
gr.addQuery('active', true);
\`\`\`
`;
    const chunks = chunkDocument("markdown/api-reference/server-api-reference/c_GlideRecordAPI.md", markdown, "australia", "australia");
    const addQuery = chunks.filter((chunk) => chunk.methodName === "addQuery");
    const parameters = addQuery.filter((chunk) => chunk.chunkType === "parameter");
    expect(parameters).toHaveLength(2);
    expect(parameters[0]?.content).toContain("name: String. Table field name.");
    expect(parameters[1]?.content).toContain(">");
    expect(parameters[1]?.content).toContain("<");
    expect(addQuery.find((chunk) => chunk.chunkType === "returns")?.content).toContain("GlideQueryCondition");
  });

  it("adds split identifier keywords to every API chunk", () => {
    const markdown = `---
title: GlideEmailOutbound - Global
release: australia
---
# GlideEmailOutbound - Global

## GlideEmailOutbound - addAddress(String recipient)

Adds a recipient.

|Name|Type|Description|
|----|----|-----------|
|recipient|String|Email address.|

|Type|Description|
|----|-----------|
|void||

\`\`\`
email.addAddress("user@example.com");
\`\`\`
`;
    const chunks = chunkDocument("markdown/api-reference/server-api-reference/c_GlideEmailOutboundAPI.md", markdown, "australia", "australia");
    const apiChunks = chunks.filter((chunk) => chunk.methodName === "addAddress");
    expect(apiChunks).toHaveLength(4);
    expect(apiChunks.every((chunk) => chunk.content.includes("Keywords: glide email outbound add address"))).toBe(true);
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

  it("splits long sections at paragraph boundaries without dropping the tail", () => {
    const paragraphs = Array.from({ length: 80 }, (_, index) => `Paragraph ${index}: ${"retrieval guidance and configuration details ".repeat(8)}`).join("\n\n");
    const chunks = chunkDocument("markdown/it-service-management/long-guide.md", `---\ntitle: Long guide\n---\n# Long guide\n## Configuration\n${paragraphs}`, "australia", "australia");
    const sectionChunks = chunks.filter((chunk) => chunk.chunkType === "section");
    expect(sectionChunks.length).toBeGreaterThan(1);
    expect(sectionChunks.every((chunk) => chunk.content.length <= MAX_CHUNK_CHARACTERS)).toBe(true);
    expect(sectionChunks.at(-1)?.content).toContain("Paragraph 79:");
  });

  it("respects a smaller model-specific chunk budget", () => {
    const paragraphs = Array.from({ length: 20 }, (_, index) => `Paragraph ${index}: ${"compact retrieval guidance ".repeat(12)}`).join("\n\n");
    const chunks = chunkDocument("markdown/guides/minilm.md", `---\ntitle: MiniLM guide\n---\n# MiniLM guide\n## Configuration\n${paragraphs}`, "australia", "australia", 768);
    const sections = chunks.filter((chunk) => chunk.chunkType === "section");
    expect(sections.length).toBeGreaterThan(1);
    expect(sections.every((chunk) => chunk.content.length <= 768)).toBe(true);
    expect(sections.at(-1)?.content).toContain("Paragraph 19:");
  });

  it("summarizes method prose after fenced code", () => {
    const markdown = `# Example
## Example - run()

\`\`\`javascript
const heading = "not a summary";
\`\`\`

Runs the example after initialization.

Later details.`;
    const method = chunkDocument("markdown/api-reference/server-api-reference/c_ExampleAPI.md", markdown, "australia", "australia").find((chunk) => chunk.chunkType === "method");
    expect(method?.metadata.summary).toBe("Runs the example after initialization.");
  });

  it("skips lists, tables, and HTML blocks before method prose", () => {
    const markdown = `# Example
## Example - run()
- preliminary item

|Name|Type|Description|
|----|----|-----------|
|value|String|Input value.|

<table><tr><td>Not prose</td></tr></table>

Runs with the supplied value.`;
    const method = chunkDocument("markdown/api-reference/server-api-reference/c_ExampleAPI.md", markdown, "australia", "australia").find((chunk) => chunk.chunkType === "method");
    expect(method?.metadata.summary).toBe("Runs with the supplied value.");
  });

  it.each([
    ["c_GlideRecordAPI.md", "GlideRecord"],
    ["r_GlideQueryAPI.md", "GlideQuery"],
    ["p_GlideSystemScoped.md", "GlideSystem"],
  ])("derives API object names from prefixed filenames", (filename, expected) => {
    const [overview] = chunkDocument(`markdown/api-reference/server-api-reference/${filename}`, "No title is available.", "australia", "australia");
    expect(overview?.objectName).toBe(expected);
    expect(overview?.content).toContain(`Object: ${expected}`);
  });

  it("preserves API suffixes in REST API filenames", () => {
    const [overview] = chunkDocument("markdown/api-reference/rest-apis/c_CustomAPI.md", "REST endpoint documentation.", "australia", "australia");
    expect(overview?.objectName).toBe("CustomAPI");
  });

  it.each([
    ["Global API", "Global behavior.", ["Global"]],
    ["Scoped API", "Scoped behavior.", ["Scoped"]],
    ["Mixed API", "Available to global and scoped applications.", ["Global", "Scoped"]],
  ])("extracts API scopes", (title, prose, expected) => {
    const [overview] = chunkDocument("markdown/api-reference/server-api-reference/c_ScopeExampleAPI.md", `---\ntitle: ${title}\n---\n# ${title}\n${prose}`, "australia", "australia");
    expect(overview?.metadata.scopes).toEqual(expected);
  });

  it("prefers frontmatter descriptions for API overview summaries", () => {
    const [overview] = chunkDocument("markdown/api-reference/server-api-reference/c_ExampleAPI.md", "---\ntitle: Example - Global\ndescription: Preferred concise API description.\n---\n# Example - Global\nFallback body paragraph.", "australia", "australia");
    expect(overview?.content).toContain("Summary: Preferred concise API description.");
    expect(overview?.content).not.toContain("Fallback body paragraph");
  });

  it("classifies Markdown tables by complete normalized header shape", () => {
    const markdown = `# Example
## Example - inspect()
Inspects a value.

|Name|Type|Value|
|----|----|-----|
|unrelated|String|Ignored|

|Name|Type|Description|Optional|
|----|----|-----------|--------|
|value|String|Input value.|No|

|Type|Description|
|----|-----------|
|Boolean|Whether it matched.|

|Property|Description|
|--------|-----------|
|count|Number matched.|

|Properties|Description|
|----------|-----------|
|items|Matched values.|`;
    const chunks = chunkDocument("markdown/api-reference/server-api-reference/c_ExampleAPI.md", markdown, "australia", "australia");
    expect(chunks.filter((chunk) => chunk.chunkType === "parameter")).toHaveLength(1);
    const returns = chunks.filter((chunk) => chunk.chunkType === "returns");
    expect(returns).toHaveLength(3);
    expect(returns.map((chunk) => chunk.content)).toEqual(expect.arrayContaining([
      expect.stringContaining("Boolean: Whether it matched."),
      expect.stringContaining("count: Number matched."),
      expect.stringContaining("items: Matched values."),
    ]));
  });

  it("extracts classified HTML tables with entities and line breaks", () => {
    const markdown = `# Example
## Example - decode()
Decodes a value.
<table><tr><th>Name</th><th>Type</th><th>Description</th></tr><tr><td>value</td><td>String</td><td>One&lt;Two<br>Three&amp;Four</td></tr></table>`;
    const parameter = chunkDocument("markdown/api-reference/server-api-reference/c_ExampleAPI.md", markdown, "australia", "australia").find((chunk) => chunk.chunkType === "parameter");
    expect(parameter?.content).toContain("One<Two Three&Four");
  });

  it("adds complete shared and focused method metadata", () => {
    const markdown = `# Example
## Example - run(String value)
Runs an example.
|Name|Type|Description|
|----|----|-----------|
|value|String|Input value.|
\`\`\`
run("value");
\`\`\``;
    const chunks = chunkDocument("markdown/api-reference/server-api-reference/c_ExampleAPI.md", markdown, "australia", "australia").filter((chunk) => chunk.methodName === "run");
    const method = chunks.find((chunk) => chunk.chunkType === "method");
    expect(method?.metadata).toMatchObject({ object_name: "Example", method_name: "run", section: "method", examples_count: 1 });
    expect(method?.metadata.full_content).toContain("Runs an example");
    for (const chunk of chunks.filter((item) => item.chunkType !== "method")) {
      expect(chunk.metadata).toMatchObject({ object_name: "Example", method_name: "run", section: chunk.chunkType, examples_count: 1 });
      expect(chunk.metadata).not.toHaveProperty("full_content");
    }
  });

  it("ignores headings inside fenced code across document types", () => {
    const api = chunkDocument("markdown/api-reference/server-api-reference/c_ExampleAPI.md", "# Example\n\`\`\`markdown\n## Example - fake()\n\`\`\`\n## Example - real()\nReal method.", "australia", "australia");
    expect(api.filter((chunk) => chunk.chunkType === "method").map((chunk) => chunk.methodName)).toEqual(["real"]);

    const topic = chunkDocument("markdown/guides/example.md", "# Example\n\`\`\`markdown\n## Fake section\n\`\`\`\n## Real section\nReal content.", "australia", "australia");
    expect(topic.filter((chunk) => chunk.chunkType === "section").map((chunk) => chunk.heading)).toEqual(["Real section"]);

    const glossary = chunkDocument("markdown/glossary/example.md", "# Glossary\n\`\`\`markdown\n### Fake term\n\`\`\`\n### Real term\nReal definition.", "australia", "australia");
    expect(glossary.filter((chunk) => chunk.chunkType === "definition").map((chunk) => chunk.heading)).toEqual(["Real term"]);
  });

  it("keeps split API chunks within a configured maximum", () => {
    const detail = `Detailed method guidance ${"with configuration and retrieval context ".repeat(20)}`;
    const chunks = chunkDocument("markdown/api-reference/server-api-reference/c_ExampleAPI.md", `# Example\n## Example - run()\n${detail}`, "australia", "australia", 400);
    const methods = chunks.filter((chunk) => chunk.chunkType === "method");
    expect(methods.length).toBeGreaterThan(1);
    expect(methods.every((chunk) => chunk.content.length <= 400)).toBe(true);
  });
});