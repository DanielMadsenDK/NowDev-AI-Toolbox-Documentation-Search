import { createHash } from "node:crypto";
import path from "node:path";
import YAML from "yaml";
import type { ChunkType, DocType, DocumentChunk } from "./types.js";

const API_PREFIXES = [
  "markdown/api-reference/server-api-reference/",
  "markdown/api-reference/cllent-mobile-api-reference/",
  "markdown/api-reference/ui-builder-api-reference/",
  "markdown/api-reference/rest-apis/",
];
const GLOSSARY_PREFIX = "markdown/glossary/";
const TOP_LEVEL_API = /^markdown\/api-reference\/[^/]+\.md$/;
const METHOD_HEADING = /^##\s+(.+?)(?:\s+[-–—]\s*|\s*[-–—]\s+)(.+?)\s*$/;
const CODE_FENCE = /```[\w+-]*\n([\s\S]*?)\n```/g;

interface Frontmatter {
  [key: string]: unknown;
}

function text(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function nullable(value: unknown): string | null {
  const result = text(value);
  return result || null;
}

export function contentHash(markdown: string): string {
  return createHash("sha1").update(markdown).digest("hex");
}

export function parseFrontmatter(markdown: string): [Frontmatter, string] {
  const match = markdown.match(/^---\s*\n([\s\S]*?)\n---\s*\n/);
  if (!match) return [{}, markdown];
  const source = match[1] ?? "";
  let metadata: Frontmatter | null;
  try {
    metadata = YAML.parse(source) as Frontmatter | null;
  } catch {
    metadata = {};
    for (const line of source.split("\n")) {
      const field = line.match(/^([A-Za-z0-9_]+):\s*(.*)$/);
      if (!field) continue;
      const key = field[1]!;
      const rawValue = field[2]!.trim();
      if (!rawValue) {
        metadata[key] = "";
        continue;
      }
      try {
        metadata[key] = /^[\[{"']/.test(rawValue) || /^(?:true|false|null|-?\d+(?:\.\d+)?)$/i.test(rawValue)
          ? YAML.parse(rawValue)
          : rawValue;
      } catch {
        metadata[key] = rawValue.replace(/^(?:"(.*)"|'(.*)')$/, "$1$2");
      }
    }
  }
  return [metadata ?? {}, markdown.slice(match[0].length)];
}

function normalizeMarkdown(markdown: string): string {
  return markdown
    .replaceAll("\\(", "(")
    .replaceAll("\\)", ")")
    .replace(/<\/table>(?=\|)/g, "</table>\n")
    .replace(/<\/table>(?=\S)/g, "</table>\n\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function publicationFromPath(sourcePath: string): string {
  return sourcePath.split("/")[1] ?? "";
}

export function classifyDocType(sourcePath: string): DocType {
  if (sourcePath.startsWith("markdown/api-reference/rest-apis/")) return "rest-api";
  if (API_PREFIXES.some((prefix) => sourcePath.startsWith(prefix)) || TOP_LEVEL_API.test(sourcePath)) return "scripting-api";
  if (sourcePath.startsWith(GLOSSARY_PREFIX)) return "glossary";
  if (sourcePath.startsWith("markdown/release-notes/")) return "release-notes";
  if (sourcePath.startsWith("markdown/api-reference/scripts/")) return "scripting-guide";
  if (sourcePath.startsWith("markdown/api-reference/")) return "developer-guide";
  return "product-doc";
}

function rawUrl(branch: string, sourcePath: string): string {
  return `https://raw.githubusercontent.com/ServiceNow/ServiceNowDocs/${encodeURIComponent(branch)}/${sourcePath.split("/").map(encodeURIComponent).join("/")}`;
}

function extractTitle(body: string, frontmatter: Frontmatter): string {
  if (text(frontmatter.title)) return text(frontmatter.title);
  return body.split("\n").find((line) => line.startsWith("# "))?.slice(2).trim() || "Untitled ServiceNow document";
}

function cleanCell(value: string): string {
  return value.replace(/&nbsp;/g, " ").replace(/<[^>]+>/g, " ").replace(/[ \t]+/g, " ").trim();
}

function parseMarkdownTables(section: string): Array<Array<Record<string, string>>> {
  const lines = section.split("\n");
  const tables: Array<Array<Record<string, string>>> = [];
  for (let index = 0; index < lines.length - 2; index += 1) {
    const header = lines[index];
    const separator = lines[index + 1];
    if (!header?.includes("|") || !separator || !/^\s*\|?[\s:|-]+\|?\s*$/.test(separator)) continue;
    const headers = header.split("|").map(cleanCell).filter(Boolean).map((item) => item.toLowerCase().replace(/[^a-z0-9]+/g, "_"));
    const rows: Array<Record<string, string>> = [];
    let cursor = index + 2;
    while (cursor < lines.length && lines[cursor]?.includes("|")) {
      const values = lines[cursor]!.split("|").map(cleanCell).filter((_, cellIndex, all) => !(cellIndex === 0 && all[cellIndex] === "") && !(cellIndex === all.length - 1 && all[cellIndex] === ""));
      if (values.length === headers.length) rows.push(Object.fromEntries(headers.map((key, i) => [key, values[i] ?? ""])));
      cursor += 1;
    }
    if (rows.length) tables.push(rows);
    index = cursor - 1;
  }
  return tables;
}

function firstParagraph(body: string): string {
  const result: string[] = [];
  for (const line of body.split("\n")) {
    const value = line.trim();
    if (!value || value.startsWith("#") || value.startsWith("-") || value.startsWith("|") || value.startsWith("```")) {
      if (result.length) break;
      continue;
    }
    result.push(value);
  }
  return result.join(" ").slice(0, 500);
}

function extractExamples(body: string): string[] {
  return [...body.matchAll(CODE_FENCE)].map((match) => match[1]?.trim() ?? "").filter(Boolean);
}

function apiContent(label: string, objectName: string, methodName: string | null, signature: string | null, release: string, detail?: string): string {
  return [
    `Object: ${objectName}`,
    methodName ? `Method: ${methodName}` : null,
    signature ? `Signature: ${signature}` : null,
    `Release: ${release}`,
    detail ? `${label}: ${detail}` : null,
  ].filter(Boolean).join("\n");
}

function chunkApiDoc(sourcePath: string, markdown: string, branch: string, family: string): DocumentChunk[] {
  const [frontmatter, rawBody] = parseFrontmatter(markdown);
  const body = normalizeMarkdown(rawBody);
  const title = extractTitle(body, frontmatter);
  const release = text(frontmatter.release) || family;
  const docType = classifyDocType(sourcePath);
  const publication = publicationFromPath(sourcePath);
  const hash = contentHash(markdown);
  const titleObject = title.split(" - ")[0]?.replace(/\s*\(deprecated\)\s*/i, "").trim() || path.basename(sourcePath, ".md");
  const metadataBase = {
    ...frontmatter,
    title,
    release,
    family,
    branch,
    source_path: sourcePath,
    url: rawUrl(branch, sourcePath),
    deprecated: /\(\s*deprecated\s*\)/i.test(normalizeMarkdown(title)),
  };
  const lines = body.split("\n");
  const positions = lines.flatMap((line, index) => {
    const match = line.trim().match(METHOD_HEADING);
    return match ? [{ objectName: match[1]!.trim(), signature: match[2]!.trim(), index }] : [];
  });
  const overviewEnd = positions[0]?.index ?? lines.length;
  const overview = lines.slice(0, overviewEnd).join("\n").trim();
  const chunks: DocumentChunk[] = [{
    docType,
    publication,
    sourcePath,
    release,
    chunkType: "overview",
    chunkIndex: 0,
    title,
    heading: null,
    content: apiContent("Summary", titleObject, null, null, release, firstParagraph(overview)),
    topicType: null,
    product: nullable(frontmatter.product),
    classification: nullable(frontmatter.classification),
    lastUpdated: nullable(frontmatter.last_updated),
    objectName: titleObject,
    methodName: null,
    metadata: { ...metadataBase, full_content: body },
    contentHash: hash,
  }];

  for (let methodIndex = 0; methodIndex < positions.length; methodIndex += 1) {
    const position = positions[methodIndex]!;
    const end = positions[methodIndex + 1]?.index ?? lines.length;
    const section = lines.slice(position.index, end).join("\n").trim();
    const methodName = position.signature.replaceAll("\\(", "(").split("(")[0]!.trim();
    const summary = firstParagraph(lines.slice(position.index + 1, end).join("\n"));
    const tables = parseMarkdownTables(section);
    const parameters = tables.flat().filter((row) => "name" in row && "type" in row);
    const returns = tables.flat().filter((row) => !("name" in row) && ("type" in row || "property" in row || "properties" in row));
    const examples = extractExamples(section);
    const methodMetadata = { ...metadataBase, method_signature: position.signature, summary, parameters, returns, examples, full_content: section };
    const append = (chunkType: ChunkType, content: string, metadata: Record<string, unknown>) => chunks.push({
      docType,
      publication,
      sourcePath,
      release,
      chunkType,
      chunkIndex: chunks.length,
      title,
      heading: position.signature,
      content,
      topicType: null,
      product: nullable(frontmatter.product),
      classification: nullable(frontmatter.classification),
      lastUpdated: nullable(frontmatter.last_updated),
      objectName: position.objectName,
      methodName,
      metadata,
      contentHash: hash,
    });
    append(docType === "rest-api" ? "endpoint" : "method", apiContent("Summary", position.objectName, methodName, position.signature, release, summary), methodMetadata);
    parameters.forEach((parameter) => append("parameter", apiContent("Parameter", position.objectName, methodName, position.signature, release, `${parameter.name}: ${parameter.type}. ${parameter.description ?? ""}`.trim()), { ...methodMetadata, full_content: undefined, parameter }));
    returns.forEach((result) => append("returns", apiContent("Returns", position.objectName, methodName, position.signature, release, `${result.type ?? ""}: ${result.description ?? ""}`.trim()), { ...methodMetadata, full_content: undefined, return: result }));
    examples.forEach((example, exampleIndex) => append("example", apiContent("Example", position.objectName, methodName, position.signature, release, example), { ...methodMetadata, full_content: undefined, example, example_index: exampleIndex + 1 }));
  }
  return chunks;
}

function baseTopicMetadata(sourcePath: string, frontmatter: Frontmatter, branch: string, family: string, title: string, release: string) {
  return {
    ...frontmatter,
    title,
    release,
    family,
    branch,
    publication: publicationFromPath(sourcePath),
    source_path: sourcePath,
    url: rawUrl(branch, sourcePath),
  };
}

function chunkTopicDoc(sourcePath: string, markdown: string, branch: string, family: string): DocumentChunk[] {
  const [frontmatter, rawBody] = parseFrontmatter(markdown);
  if (frontmatter.topic_type === "toc") return [];
  const body = normalizeMarkdown(rawBody);
  const title = extractTitle(body, frontmatter);
  const release = text(frontmatter.release) || family;
  const metadata = baseTopicMetadata(sourcePath, frontmatter, branch, family, title, release);
  const lines = body.split("\n");
  const headings = lines.flatMap((line, index) => line.trim().startsWith("## ") ? [{ heading: line.trim().slice(3).trim(), index }] : []);
  const firstHeading = headings[0]?.index ?? lines.length;
  const common = {
    docType: classifyDocType(sourcePath), publication: publicationFromPath(sourcePath), sourcePath, release,
    title, topicType: nullable(frontmatter.topic_type), product: nullable(frontmatter.product),
    classification: nullable(frontmatter.classification), lastUpdated: nullable(frontmatter.last_updated),
    objectName: null, methodName: null, contentHash: contentHash(markdown),
  } as const;
  const contentPrefix = (heading?: string) => [
    `Title: ${title}`,
    heading ? `Section: ${heading}` : null,
    text(frontmatter.product) ? `Product: ${text(frontmatter.product)}` : null,
    `Release: ${release}`,
  ].filter(Boolean).join("\n");
  const overviewBody = lines.slice(0, firstHeading).join("\n").trim();
  const chunks: DocumentChunk[] = [{ ...common, chunkType: "overview", chunkIndex: 0, heading: null, content: `${contentPrefix()}${overviewBody ? `\n\n${overviewBody.slice(0, 2500)}` : ""}`, metadata: { ...metadata, full_content: body } }];
  headings.forEach((item, index) => {
    const end = headings[index + 1]?.index ?? lines.length;
    const sectionBody = lines.slice(item.index + 1, end).join("\n").trim();
    chunks.push({ ...common, chunkType: "section", chunkIndex: index + 1, heading: item.heading, content: `${contentPrefix(item.heading)}${sectionBody ? `\n\n${sectionBody.slice(0, 2500)}` : ""}`, metadata: { ...metadata, section_content: lines.slice(item.index, end).join("\n").trim() } });
  });
  return chunks;
}

function chunkGlossary(sourcePath: string, markdown: string, branch: string, family: string): DocumentChunk[] {
  const [frontmatter, rawBody] = parseFrontmatter(markdown);
  const body = normalizeMarkdown(rawBody);
  const title = extractTitle(body, frontmatter);
  const release = text(frontmatter.release) || family;
  const lines = body.split("\n");
  const terms = lines.flatMap((line, index) => line.trim().startsWith("### ") ? [{ term: line.trim().slice(4).trim(), index }] : []);
  const common = {
    docType: "glossary" as const, publication: publicationFromPath(sourcePath), sourcePath, release, title,
    topicType: "reference", product: "ServiceNow AI Platform", classification: "glossary",
    lastUpdated: nullable(frontmatter.last_updated), objectName: null, methodName: null, contentHash: contentHash(markdown),
  };
  if (!terms.length) return [{ ...common, chunkType: "overview", chunkIndex: 0, heading: null, content: `Title: ${title}\nProduct: ServiceNow AI Platform\nRelease: ${release}\n\n${body}`, metadata: { source_path: sourcePath, url: rawUrl(branch, sourcePath), release, family, full_content: body } }];
  return terms.map((item, index) => {
    const end = terms[index + 1]?.index ?? lines.length;
    const definition = lines.slice(item.index + 1, end).map((line) => line.trim()).filter((line) => line && !line.startsWith("## ")).join(" ");
    return { ...common, chunkType: "definition", chunkIndex: index, heading: item.term, content: `Term: ${item.term}\nDefinition: ${definition}\nProduct: ServiceNow AI Platform\nRelease: ${release}`, metadata: { term: item.term, definition, source_path: sourcePath, url: rawUrl(branch, sourcePath), release, family } };
  });
}

export function chunkDocument(sourcePath: string, markdown: string, branch: string, family: string): DocumentChunk[] {
  if (API_PREFIXES.some((prefix) => sourcePath.startsWith(prefix)) || TOP_LEVEL_API.test(sourcePath)) return chunkApiDoc(sourcePath, markdown, branch, family);
  if (sourcePath.startsWith(GLOSSARY_PREFIX)) return chunkGlossary(sourcePath, markdown, branch, family);
  return chunkTopicDoc(sourcePath, markdown, branch, family);
}