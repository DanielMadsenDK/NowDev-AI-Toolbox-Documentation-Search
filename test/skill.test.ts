import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import YAML from "yaml";

describe("bundled DocumentationSearch skill", () => {
  it("has valid metadata and core CLI guidance", () => {
    const skillPath = path.resolve("skills/nowdev-ai-toolbox-documentationsearch/SKILL.md");
    const content = fs.readFileSync(skillPath, "utf8");
    const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n/);
    expect(match).not.toBeNull();
    const metadata = YAML.parse(match![1]!) as { name?: string; description?: string };
    expect(metadata.name).toBe("nowdev-ai-toolbox-documentationsearch");
    expect(metadata.description).toContain("ServiceNow documentation");
    expect(content).toContain("nowdev-ai-toolbox-documentationsearch --json status");
    expect(content).toContain("search_servicenow_docs");
    expect(content).toContain("failures");
    expect(content).toContain("bge-base-en-v1.5");
    expect(content).toContain("Treat search results as the default evidence");
    expect(content).toContain("Do not retrieve a full document merely because a source was selected");
    expect(content).toContain("Retrieve full content only when the search result and outline cannot safely answer the question");
  });
});