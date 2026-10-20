#!/usr/bin/env node
import { fileURLToPath } from "node:url";
import { McpServer, ResourceTemplate } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import {
	DEFAULT_FAMILY,
	DEFAULT_MAX_RESULTS_PER_SOURCE,
	DEFAULT_SEARCH_LIMIT,
	DEFAULT_SIMILARITY_THRESHOLD,
	MAX_DOWNLOAD_CONCURRENCY,
	MAX_RESULTS_PER_SOURCE,
	MAX_SEARCH_LIMIT,
	PACKAGE_NAME,
	PACKAGE_VERSION,
} from "./config.js";
import { DocumentationSearch, type DocumentationSearchOptions } from "./service-context.js";
import { CHUNK_TYPES, DOC_TYPES, DOCUMENTATION_AREAS, type SearchResult } from "./types.js";

// Compact JSON: indentation only costs the calling model tokens.
function result(value: unknown) {
	return {
		content: [{ type: "text" as const, text: JSON.stringify(value) }],
	};
}

// Tool failures are reported in the result, not thrown, so the calling model sees them and can correct its request.
function errorResult(message: string) {
	return {
		isError: true,
		content: [{ type: "text" as const, text: message }],
	};
}

function notIndexedMessage(sourcePath: string, release: string | undefined): string {
	return `No indexed document ${sourcePath}${release ? ` in release ${release}` : ""}. Use a sourcePath exactly as returned by search_servicenow_docs.`;
}

type CompactSearchResult = Omit<SearchResult, "metadata" | "contentHash" | "context"> & {
	sourceUrl?: string;
	breadcrumb?: string;
};

export const DOCUMENT_URI_TEMPLATE = "servicenow-doc://{release}/{+sourcePath}";

export function documentUri(release: string, sourcePath: string): string {
	return `servicenow-doc://${encodeURIComponent(release)}/${sourcePath.split("/").map(encodeURIComponent).join("/")}`;
}

export const MCP_TOOL_DESCRIPTIONS = {
	search: "Hybrid semantic and keyword search over the local ServiceNow documentation index. Start with the user's original wording and a release filter when known; inspect that baseline before applying docType, publication, chunkType, or topicType filters. The documentation uses its own vocabulary, so everyday wording can miss the page that answers it: for a scripting task that does not name an API (\"update a record without running business rules\"), search again with the API object and method you expect (\"GlideRecord setWorkflow business rules\") and docType: \"scripting-api\"; for a question that describes a feature instead of naming it, search again with its canonical ServiceNow name (\"Discovery\", \"Domain separation\"). docType is the indexed source classification, not the ServiceNow topic; leave it unset for configuration procedures. Use maxResultsPerSource: 1 for broad topic discovery. Set rerank: true to rescore the top candidates with a local cross-encoder when the first results look close but not right.",
	document: "Fetch complete indexed chunks for a selected ServiceNowDocs source path. Use only when the matching search result and its outline cannot establish the answer; full documents can be large.",
	outline: "Fetch headings and concise previews for a selected ServiceNowDocs source path. Prefer this after a relevant but incomplete search result or when document scope is unclear; use the same sourcePath selected from search.",
	publications: "List locally indexed publications and document counts. Use this to discover publication filters after inspecting an unfiltered search result.",
	status: "Inspect local index size, indexed releases, location, and embedding configuration. Call before release-specific searches or before initializing/updating an index.",
	update: "Download and incrementally index public ServiceNowDocs content. This mutates the local index; use only when requested. Use limit for a smoke test, avoid refresh for routine updates, and inspect failures before relying on complete coverage.",
} as const;

export const MCP_SERVER_INSTRUCTIONS = "Local index of the public ServiceNow documentation. Call get_documentation_search_status first to see which releases are indexed. Search with the user's own wording and a release filter, inspect those results, then refine: for scripting tasks search again with the API object and method you expect, with docType \"scripting-api\"; for described features search with the feature's ServiceNow name. Read a promising result's outline before fetching the full document, and cite the release and sourceUrl in answers. Only call update_servicenow_docs when the user asks for it.";

function breadcrumbText(value: unknown): string | undefined {
	if (!Array.isArray(value)) return undefined;
	const items = value.filter((item): item is string => typeof item === "string" && item.trim() !== "");
	return items.length ? [...items].reverse().join(" > ") : undefined;
}

export function formatSearchResults(results: SearchResult[], includeMetadata: boolean): Array<SearchResult | CompactSearchResult> {
	if (includeMetadata) return results;
	return results.map(({ metadata, contentHash: _contentHash, context: _context, ...searchResult }) => {
		// Prefer the public docs page over the raw GitHub markdown; ServiceNowDocs escapes underscores in canonical URLs.
		const canonicalUrl = typeof metadata.canonical_url === "string" ? metadata.canonical_url.replaceAll("\\_", "_") : undefined;
		const sourceUrl = canonicalUrl ?? (typeof metadata.url === "string" ? metadata.url : undefined);
		const breadcrumb = breadcrumbText(metadata.breadcrumb);
		return { ...searchResult, ...(sourceUrl ? { sourceUrl } : {}), ...(breadcrumb ? { breadcrumb } : {}) };
	});
}

const searchResultSchema = z.looseObject({
	id: z.number().optional(),
	docType: z.string(),
	publication: z.string(),
	sourcePath: z.string(),
	release: z.string(),
	chunkType: z.string(),
	chunkIndex: z.number(),
	title: z.string(),
	heading: z.string().nullable(),
	content: z.string(),
	objectName: z.string().nullable(),
	methodName: z.string().nullable(),
	similarity: z.number(),
	score: z.number(),
	sourceUrl: z.string().optional(),
	breadcrumb: z.string().optional(),
	metadata: z.record(z.string(), z.unknown()).optional(),
});

/** Builds an MCP server whose tools and resources read from (and update) the given index. */
export function createMcpServer(context: DocumentationSearch): McpServer {
	const server = new McpServer({ name: PACKAGE_NAME, version: PACKAGE_VERSION }, { instructions: MCP_SERVER_INSTRUCTIONS });
	let updating = false;

	server.registerTool("search_servicenow_docs", {
		title: "Search ServiceNow documentation",
		description: MCP_TOOL_DESCRIPTIONS.search,
		inputSchema: z.object({
			query: z.string().min(1),
			limit: z.number().int().min(1).max(MAX_SEARCH_LIMIT).default(DEFAULT_SEARCH_LIMIT),
			threshold: z.number().min(-1).max(1).default(DEFAULT_SIMILARITY_THRESHOLD).describe("Minimum cosine similarity; exact API object and method keyword matches are retained below this value"),
			deduplicateReleases: z.boolean().default(false),
			maxResultsPerSource: z.number().int().min(1).max(MAX_RESULTS_PER_SOURCE).default(DEFAULT_MAX_RESULTS_PER_SOURCE),
			rerank: z.boolean().default(false).describe("Rescore the top candidates with a local cross-encoder; adds several hundred milliseconds"),
			release: z.string().optional(),
			docType: z.enum(DOC_TYPES).optional(),
			publication: z.string().optional(),
			chunkType: z.enum(CHUNK_TYPES).optional(),
			topicType: z.string().optional(),
			includeMetadata: z.boolean().default(false).describe("Include detailed source metadata such as API parameters and examples. Omit for compact agent-oriented results."),
		}),
		outputSchema: z.object({ results: z.array(searchResultSchema) }),
		annotations: { readOnlyHint: true, idempotentHint: true, openWorldHint: false },
	}, async ({ query, includeMetadata, ...options }) => {
		const results = formatSearchResults(await context.search(query, options), includeMetadata);
		const documents = new Map(results.map((item) => [`${item.release}\u0000${item.sourcePath}`, item]));
		return {
			structuredContent: { results },
			content: [
				{ type: "text" as const, text: JSON.stringify(results) },
				...[...documents.values()].map((item) => ({ type: "resource_link" as const, uri: documentUri(item.release, item.sourcePath), name: item.sourcePath, title: item.title, mimeType: "application/json" })),
			],
		};
	});

	server.registerResource("servicenow-document", new ResourceTemplate(DOCUMENT_URI_TEMPLATE, { list: undefined }), {
		title: "ServiceNow document",
		description: "Complete indexed chunks for one ServiceNowDocs source path in one release, as returned by get_servicenow_document.",
		mimeType: "application/json",
	}, async (uri, { release, sourcePath }) => {
		let releaseValue: string;
		let pathValue: string;
		try {
			[releaseValue, pathValue] = [release, sourcePath].map((value) => decodeURIComponent(Array.isArray(value) ? value.join("/") : String(value))) as [string, string];
		} catch {
			throw new Error(`Invalid document URI ${uri.href}`);
		}
		const chunks = context.getDocument(pathValue, releaseValue);
		if (!chunks.length) throw new Error(`No indexed document ${pathValue} in release ${releaseValue}`);
		return { contents: [{ uri: uri.href, mimeType: "application/json", text: JSON.stringify(chunks) }] };
	});

	server.registerTool("get_servicenow_document", {
		title: "Get ServiceNow document",
		description: MCP_TOOL_DESCRIPTIONS.document,
		inputSchema: z.object({ sourcePath: z.string().min(1), release: z.string().optional() }),
		annotations: { readOnlyHint: true, idempotentHint: true, openWorldHint: false },
	}, async ({ sourcePath, release }) => {
		const chunks = context.getDocument(sourcePath, release);
		return chunks.length ? result(chunks) : errorResult(notIndexedMessage(sourcePath, release));
	});

	server.registerTool("get_servicenow_document_outline", {
		title: "Get ServiceNow document outline",
		description: MCP_TOOL_DESCRIPTIONS.outline,
		inputSchema: z.object({ sourcePath: z.string().min(1), release: z.string().optional() }),
		annotations: { readOnlyHint: true, idempotentHint: true, openWorldHint: false },
	}, async ({ sourcePath, release }) => {
		const outline = context.getDocumentOutline(sourcePath, release);
		return outline.length ? result(outline) : errorResult(notIndexedMessage(sourcePath, release));
	});

	server.registerTool("list_servicenow_publications", {
		title: "List ServiceNow publications",
		description: MCP_TOOL_DESCRIPTIONS.publications,
		inputSchema: z.object({ release: z.string().optional() }),
		annotations: { readOnlyHint: true, idempotentHint: true, openWorldHint: false },
	}, async ({ release }) => result(context.listPublications(release)));

	server.registerTool("get_documentation_search_status", {
		title: "Get DocumentationSearch status",
		description: MCP_TOOL_DESCRIPTIONS.status,
		inputSchema: z.object({}),
		annotations: { readOnlyHint: true, idempotentHint: true, openWorldHint: false },
	}, async () => result(context.status()));

	server.registerTool("update_servicenow_docs", {
		title: "Update ServiceNow documentation",
		description: MCP_TOOL_DESCRIPTIONS.update,
		inputSchema: z.object({
			family: z.string().default(DEFAULT_FAMILY),
			branch: z.string().optional(),
			area: z.enum(DOCUMENTATION_AREAS).default("all-docs"),
			refresh: z.boolean().default(false),
			limit: z.number().int().positive().optional(),
			concurrency: z.number().int().min(1).max(MAX_DOWNLOAD_CONCURRENCY).optional(),
		}),
		// Not idempotent or merely additive: an update removes sources deleted upstream and records a new manifest.
		annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: false, openWorldHint: true },
	}, async (options, extra) => {
		// Two updates writing the same index would each compute changes from a stale snapshot and race their writes.
		if (updating) return errorResult("An update is already running for this index; wait for it to finish.");
		const progressToken = extra._meta?.progressToken;
		let progress = 0;
		// Indexing a full family takes a long time; progress notifications keep clients informed and their request timers reset.
		const onProgress = progressToken === undefined ? undefined : (message: string) => {
			progress += 1;
			void extra.sendNotification({ method: "notifications/progress", params: { progressToken, progress, message } }).catch(() => undefined);
		};
		updating = true;
		try {
			// A cancelled request stops between documents; what was committed before then stays indexed.
			return result(await context.update({ ...options, onProgress, signal: extra.signal }));
		} catch (error) {
			if (extra.signal.aborted) return errorResult("The update was cancelled; documents indexed before cancellation were kept.");
			throw error;
		} finally {
			updating = false;
		}
	});

	return server;
}

/** Serves the index over stdio until the client disconnects or the process is signalled. */
export async function startMcpServer(options: DocumentationSearchOptions = {}): Promise<void> {
	const context = new DocumentationSearch(options);
	const server = createMcpServer(context);
	let shuttingDown = false;
	const shutdown = () => {
		if (shuttingDown) return;
		shuttingDown = true;
		void (async () => {
			try {
				// The server stops taking requests before the index and models it serves from are released.
				await server.close();
			} finally {
				await context.close();
				process.exit(0);
			}
		})();
	};
	process.once("SIGINT", shutdown);
	process.once("SIGTERM", shutdown);
	// The client closing stdin is the normal end of a stdio session; without this, worker threads and model sessions
	// could keep an orphaned server alive.
	process.stdin.once("end", shutdown);
	process.stdin.once("close", shutdown);
	server.server.onclose = shutdown;
	await server.connect(new StdioServerTransport());
	// Loads the query model and the in-memory vector index in the background so the first search is fast.
	if (context.database.stats().chunks) void context.warmUp().catch((error: unknown) => console.error(`Warm-up failed: ${error instanceof Error ? error.message : String(error)}`));
}

const invokedDirectly = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (invokedDirectly) {
	startMcpServer().catch((error: unknown) => {
		console.error(error instanceof Error ? error.message : String(error));
		process.exitCode = 1;
	});
}
