#!/usr/bin/env node
import { fileURLToPath } from "node:url";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { PACKAGE_NAME, PACKAGE_VERSION } from "./config.js";
import { DocumentationSearch } from "./service-context.js";

function result(value: unknown) {
	return {
		content: [{ type: "text" as const, text: JSON.stringify(value, null, 2) }],
	};
}

export async function startMcpServer(dataDirectory?: string): Promise<void> {
	const context = new DocumentationSearch({ dataDirectory: dataDirectory ?? process.env.DOCUMENTATIONSEARCH_HOME ?? process.env.SERVICECONTEXT_HOME });
	const server = new McpServer({ name: PACKAGE_NAME, version: PACKAGE_VERSION });

	server.registerTool("search_servicenow_docs", {
		title: "Search ServiceNow documentation",
		description: "Hybrid semantic and keyword search over the local ServiceNow documentation index.",
		inputSchema: z.object({
			query: z.string().min(1),
			limit: z.number().int().min(1).max(50).default(10),
			threshold: z.number().min(-1).max(1).default(0.3),
			deduplicateReleases: z.boolean().default(false),
			release: z.string().optional(),
			docType: z.enum(["scripting-api", "rest-api", "scripting-guide", "developer-guide", "product-doc", "release-notes", "glossary"]).optional(),
			publication: z.string().optional(),
			chunkType: z.enum(["overview", "section", "method", "endpoint", "parameter", "returns", "example", "definition"]).optional(),
			topicType: z.string().optional(),
		}),
		annotations: { readOnlyHint: true, idempotentHint: true },
	}, async ({ query, ...options }) => result(await context.search(query, options)));

	server.registerTool("get_servicenow_document", {
		title: "Get ServiceNow document",
		description: "Fetch complete indexed chunks for a ServiceNowDocs source path.",
		inputSchema: z.object({ sourcePath: z.string().min(1), release: z.string().optional() }),
		annotations: { readOnlyHint: true, idempotentHint: true },
	}, async ({ sourcePath, release }) => result(context.getDocument(sourcePath, release)));

	server.registerTool("get_servicenow_document_outline", {
		title: "Get ServiceNow document outline",
		description: "Fetch headings and concise previews for a ServiceNowDocs source path.",
		inputSchema: z.object({ sourcePath: z.string().min(1), release: z.string().optional() }),
		annotations: { readOnlyHint: true, idempotentHint: true },
	}, async ({ sourcePath, release }) => result(context.getDocumentOutline(sourcePath, release)));

	server.registerTool("list_servicenow_publications", {
		title: "List ServiceNow publications",
		description: "List locally indexed publications and document counts.",
		inputSchema: z.object({ release: z.string().optional() }),
		annotations: { readOnlyHint: true, idempotentHint: true },
	}, async ({ release }) => result(context.listPublications(release)));

	const statusHandler = async () => result(context.status());
	server.registerTool("get_documentation_search_status", {
		title: "Get DocumentationSearch status",
		description: "Inspect local index size, releases, location, and embedding model configuration.",
		inputSchema: z.object({}),
		annotations: { readOnlyHint: true, idempotentHint: true },
	}, statusHandler);

	server.registerTool("get_servicecontext_status", {
		title: "Get DocumentationSearch status (legacy alias)",
		description: "Deprecated alias for get_documentation_search_status.",
		inputSchema: z.object({}),
		annotations: { readOnlyHint: true, idempotentHint: true },
	}, statusHandler);

	server.registerTool("update_servicenow_docs", {
		title: "Update ServiceNow documentation",
		description: "Download and incrementally index public ServiceNowDocs content. This can take time on first use.",
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