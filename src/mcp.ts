#!/usr/bin/env node
import { fileURLToPath } from "node:url";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { PACKAGE_NAME, PACKAGE_VERSION } from "./config.js";
import { DocumentationSearch } from "./service-context.js";
import type { SearchResult } from "./types.js";

function result(value: unknown) {
	return {
		content: [{ type: "text" as const, text: JSON.stringify(value, null, 2) }],
	};
}

type CompactSearchResult = Omit<SearchResult, "metadata" | "contentHash"> & {
	sourceUrl?: string;
};

export const MCP_TOOL_DESCRIPTIONS = {
	search: "Hybrid semantic and keyword search over the local ServiceNow documentation index. Start with the user's original wording and a release filter when known; inspect that baseline before applying docType, publication, chunkType, or topicType filters. docType is the indexed source classification, not the ServiceNow topic; leave it unset for procedural questions. Use maxResultsPerSource: 1 for broad topic discovery.",
	document: "Fetch complete indexed chunks for a selected ServiceNowDocs source path. Use only when the matching search result and its outline cannot establish the answer; full documents can be large.",
	outline: "Fetch headings and concise previews for a selected ServiceNowDocs source path. Prefer this after a relevant but incomplete search result or when document scope is unclear; use the same sourcePath selected from search.",
	publications: "List locally indexed publications and document counts. Use this to discover publication filters after inspecting an unfiltered search result.",
	status: "Inspect local index size, indexed releases, location, and embedding configuration. Call before release-specific searches or before initializing/updating an index.",
	update: "Download and incrementally index public ServiceNowDocs content. This mutates the local index; use only when requested. Use limit for a smoke test, avoid refresh for routine updates, and inspect failures before relying on complete coverage.",
} as const;

export function formatSearchResults(results: SearchResult[], includeMetadata: boolean): Array<SearchResult | CompactSearchResult> {
	if (includeMetadata) return results;
	return results.map(({ metadata, contentHash: _contentHash, ...searchResult }) => {
		const sourceUrl = typeof metadata.url === "string" ? metadata.url : undefined;
		return sourceUrl ? { ...searchResult, sourceUrl } : searchResult;
	});
}

export async function startMcpServer(dataDirectory?: string): Promise<void> {
	const context = new DocumentationSearch({ dataDirectory: dataDirectory ?? process.env.NOWDEV_AI_TOOLBOX_DOCUMENTATIONSEARCH_HOME });
	const server = new McpServer({ name: PACKAGE_NAME, version: PACKAGE_VERSION });

	server.registerTool("search_servicenow_docs", {
		title: "Search ServiceNow documentation",
		description: MCP_TOOL_DESCRIPTIONS.search,
		inputSchema: z.object({
			query: z.string().min(1),
			limit: z.number().int().min(1).max(50).default(10),
			threshold: z.number().min(-1).max(1).default(0.3).describe("Minimum cosine similarity; exact API object and method keyword matches are retained below this value"),
			deduplicateReleases: z.boolean().default(false),
			maxResultsPerSource: z.number().int().min(1).max(10).default(3),
			release: z.string().optional(),
			docType: z.enum(["scripting-api", "rest-api", "scripting-guide", "developer-guide", "product-doc", "release-notes", "glossary"]).optional(),
			publication: z.string().optional(),
			chunkType: z.enum(["overview", "section", "method", "endpoint", "parameter", "returns", "example", "definition"]).optional(),
			topicType: z.string().optional(),
			includeMetadata: z.boolean().default(false).describe("Include detailed source metadata such as API parameters and examples. Omit for compact agent-oriented results."),
		}),
		annotations: { readOnlyHint: true, idempotentHint: true },
	}, async ({ query, includeMetadata, ...options }) => result(formatSearchResults(await context.search(query, options), includeMetadata)));

	server.registerTool("get_servicenow_document", {
		title: "Get ServiceNow document",
		description: MCP_TOOL_DESCRIPTIONS.document,
		inputSchema: z.object({ sourcePath: z.string().min(1), release: z.string().optional() }),
		annotations: { readOnlyHint: true, idempotentHint: true },
	}, async ({ sourcePath, release }) => result(context.getDocument(sourcePath, release)));

	server.registerTool("get_servicenow_document_outline", {
		title: "Get ServiceNow document outline",
		description: MCP_TOOL_DESCRIPTIONS.outline,
		inputSchema: z.object({ sourcePath: z.string().min(1), release: z.string().optional() }),
		annotations: { readOnlyHint: true, idempotentHint: true },
	}, async ({ sourcePath, release }) => result(context.getDocumentOutline(sourcePath, release)));

	server.registerTool("list_servicenow_publications", {
		title: "List ServiceNow publications",
		description: MCP_TOOL_DESCRIPTIONS.publications,
		inputSchema: z.object({ release: z.string().optional() }),
		annotations: { readOnlyHint: true, idempotentHint: true },
	}, async ({ release }) => result(context.listPublications(release)));

	server.registerTool("get_documentation_search_status", {
		title: "Get DocumentationSearch status",
		description: MCP_TOOL_DESCRIPTIONS.status,
		inputSchema: z.object({}),
		annotations: { readOnlyHint: true, idempotentHint: true },
	}, async () => result(context.status()));

	server.registerTool("update_servicenow_docs", {
		title: "Update ServiceNow documentation",
		description: MCP_TOOL_DESCRIPTIONS.update,
		inputSchema: z.object({
			family: z.string().default("australia"),
			branch: z.string().optional(),
			area: z.enum(["all-docs", "scripting", "server", "client", "scripts"]).default("all-docs"),
			refresh: z.boolean().default(false),
			limit: z.number().int().positive().optional(),
			concurrency: z.number().int().min(1).max(64).optional(),
		}),
		annotations: { readOnlyHint: false, idempotentHint: true },
	}, async (options) => result(await context.update(options)));

	const shutdown = async () => {
		context.close();
		await server.close();
	};
	process.once("SIGINT", () => void shutdown().finally(() => process.exit(0)));
	process.once("SIGTERM", () => void shutdown().finally(() => process.exit(0)));
	await server.connect(new StdioServerTransport());
}

const invokedDirectly = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (invokedDirectly) {
	startMcpServer().catch((error: unknown) => {
		console.error(error instanceof Error ? error.message : String(error));
		process.exitCode = 1;
	});
}