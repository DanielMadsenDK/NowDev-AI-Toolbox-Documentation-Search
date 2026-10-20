#!/usr/bin/env node
import { Command, CommanderError, InvalidArgumentError, Option } from "commander";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
	DEFAULT_DOWNLOAD_CONCURRENCY,
	DEFAULT_FAMILY,
	DEFAULT_MAX_EMBEDDING_CHARACTERS,
	DEFAULT_MAX_RESULTS_PER_SOURCE,
	DEFAULT_SEARCH_LIMIT,
	DEFAULT_SIMILARITY_THRESHOLD,
	MAX_DOWNLOAD_CONCURRENCY,
	MAX_RESULTS_PER_SOURCE,
	MAX_SEARCH_LIMIT,
	PACKAGE_VERSION,
	resolvePaths,
} from "./config.js";
import { harvestIndexEmbeddings, type HarvestResult } from "./embedding-cache.js";
import { EMBEDDING_DTYPES, HashEmbeddingProvider, type EmbeddingDevice, type TransformersEmbeddingOptions } from "./embedder.js";
import { DEFAULT_EMBEDDING_PROFILE, EMBEDDING_PROFILES, type EmbeddingProfileName } from "./embedding-profiles.js";
import { DocumentationSearch, type DocumentationSearchOptions } from "./service-context.js";
import { CHUNK_TYPES, DOC_TYPES, DOCUMENTATION_AREAS, type ChunkType, type DocType, type DocumentationArea } from "./types.js";

interface GlobalOptions {
	dataDir?: string;
	json?: boolean;
	deterministicEmbeddings?: boolean;
	device?: EmbeddingDevice;
	embeddingDtype?: TransformersEmbeddingOptions["dtype"];
	embeddingBatchSize?: number;
	embeddingMaxCharacters?: number;
	embeddingThreads?: number;
	embeddingWorkers?: number;
	embeddingCache: boolean;
	embeddingProfile?: EmbeddingProfileName;
	embeddingDimensions?: number;
	embeddingEndpoint?: string;
	embeddingEndpointModel?: string;
	embeddingApiKeyEnv: string;
	embeddingEndpointBatchSize?: number;
	embeddingEndpointConcurrency?: number;
	embeddingEndpointTimeout?: number;
	vectorOversample?: number;
}

function integerOption(name: string, minimum: number, maximum: number) {
	return (value: string): number => {
		if (!/^-?\d+$/.test(value)) throw new InvalidArgumentError(`${name} must be an integer`);
		const parsed = Number(value);
		if (!Number.isSafeInteger(parsed) || parsed < minimum || parsed > maximum) {
			throw new InvalidArgumentError(`${name} must be between ${minimum} and ${maximum}`);
		}
		return parsed;
	};
}

function numberOption(name: string, minimum: number, maximum: number) {
	return (value: string): number => {
		const parsed = Number(value);
		if (!Number.isFinite(parsed) || parsed < minimum || parsed > maximum) {
			throw new InvalidArgumentError(`${name} must be a finite number between ${minimum} and ${maximum}`);
		}
		return parsed;
	};
}

function contextOptions(command: Command): DocumentationSearchOptions {
	const options = command.optsWithGlobals<GlobalOptions>();
	if (Boolean(options.embeddingEndpoint) !== Boolean(options.embeddingEndpointModel)) {
		throw new Error("--embedding-endpoint and --embedding-endpoint-model must be specified together");
	}
	if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(options.embeddingApiKeyEnv)) {
		throw new Error("--embedding-api-key-env must be a valid environment variable name");
	}
	const endpointApiKey = options.embeddingEndpoint ? process.env[options.embeddingApiKeyEnv] : undefined;
	if (options.embeddingEndpoint && !endpointApiKey) {
		throw new Error(`Embedding API key environment variable ${options.embeddingApiKeyEnv} is not set`);
	}
	return {
		dataDirectory: options.dataDir,
		embeddingDevice: options.device,
		embeddingDtype: options.embeddingDtype,
		embeddingBatchSize: options.embeddingBatchSize,
		embeddingMaxCharacters: options.embeddingMaxCharacters,
		embeddingThreads: options.embeddingThreads,
		embeddingWorkers: options.embeddingWorkers,
		// Leaves the library default (on, except for custom providers such as --deterministic-embeddings) unless disabled.
		embeddingCache: options.embeddingCache ? undefined : false,
		embeddingProfile: options.embeddingProfile,
		embeddingDimensions: options.embeddingDimensions,
		embeddingEndpoint: options.embeddingEndpoint,
		embeddingEndpointModel: options.embeddingEndpointModel,
		embeddingEndpointApiKey: endpointApiKey,
		embeddingEndpointBatchSize: options.embeddingEndpointBatchSize,
		embeddingEndpointConcurrency: options.embeddingEndpointConcurrency,
		embeddingEndpointTimeoutMilliseconds: options.embeddingEndpointTimeout === undefined ? undefined : options.embeddingEndpointTimeout * 1000,
		vectorOversample: options.vectorOversample,
		embeddingProvider: options.deterministicEmbeddings ? new HashEmbeddingProvider() : undefined,
	};
}

function output(command: Command, value: unknown): void {
	const options = command.optsWithGlobals<GlobalOptions>();
	if (options.json || typeof value !== "string") console.log(JSON.stringify(value, null, 2));
	else console.log(value);
}

async function withContext(command: Command, action: (context: DocumentationSearch) => Promise<void> | void): Promise<void> {
	const context = new DocumentationSearch(contextOptions(command));
	try {
		await action(context);
	} finally {
		await context.close();
	}
}

function addUpdateOptions(command: Command): Command {
	return command
		.option("-f, --family <family>", "ServiceNow family release", DEFAULT_FAMILY)
		.option("-b, --branch <branch>", "Git branch (defaults to family)")
		.addOption(new Option("-a, --area <area>", "documentation subset to index").choices(DOCUMENTATION_AREAS).default("all-docs" satisfies DocumentationArea))
		.option("--refresh", "re-download and re-index every discovered source")
		.option("--concurrency <count>", "concurrent downloads", integerOption("concurrency", 1, MAX_DOWNLOAD_CONCURRENCY), DEFAULT_DOWNLOAD_CONCURRENCY)
		.option("--limit <count>", "limit sources for a smoke test", integerOption("limit", 1, 1_000_000));
}

const program = new Command()
	.name("nowdev-ai-toolbox-documentationsearch")
	.description("Download, index, and semantically search ServiceNow documentation locally")
	.version(PACKAGE_VERSION)
	.option("--data-dir <directory>", "override the local data directory")
	.option("--json", "emit JSON")
	.option("--deterministic-embeddings", "use lightweight hash embeddings for testing")
	.addOption(new Option("--embedding-profile <profile>", `curated ONNX embedding profile (new-index default: ${DEFAULT_EMBEDDING_PROFILE})`).choices(Object.keys(EMBEDDING_PROFILES)))
	.option("--embedding-dimensions <count>", "truncate to this many embedding dimensions (only supported by Matryoshka-capable profiles, e.g. nomic-embed-text-v1.5)", integerOption("embedding-dimensions", 1, 4096))
	.addOption(new Option("--device <device>", "device for document embedding while indexing: cpu, dml (DirectML on Windows), or webgpu (Metal on macOS, Direct3D 12/Vulkan elsewhere); searches always run on the CPU").choices(["cpu", "dml", "webgpu"]).default("cpu"))
	.addOption(new Option("--embedding-dtype <dtype>", "ONNX precision for the embedding model; remembered by the index (new-index default: q8, or fp16 with --device webgpu)").choices([...EMBEDDING_DTYPES]))
	.option("--embedding-batch-size <count>", "texts per embedding inference batch (default: 32 CPU, 8 DirectML)", integerOption("embedding-batch-size", 1, 1024))
	.option(`--embedding-max-characters <count>`, `override the profile-specific passage cap (MiniLM: 1024; other built-in profiles: ${DEFAULT_MAX_EMBEDDING_CHARACTERS})`, integerOption("embedding-max-characters", 256, 1_000_000))
	.option("--embedding-threads <count>", "total CPU threads for embedding inference, split across embedding workers (default: the host's logical core count)", integerOption("embedding-threads", 1, 1024))
	.option("--embedding-workers <count>", "parallel CPU embedding workers for indexing, each with its own model copy (default: embedding threads, at most 8; 1 embeds in-process)", integerOption("embedding-workers", 1, 64))
	.option("--no-embedding-cache", "embed every changed passage instead of reusing vectors from the embedding cache")
	.option("--embedding-endpoint <url>", "OpenAI-compatible /embeddings endpoint used only for document indexing")
	.option("--embedding-endpoint-model <model>", "model ID sent to the document embedding endpoint")
	.option("--embedding-api-key-env <name>", "environment variable containing the endpoint API key", "OPENROUTER_API_KEY")
	.option("--embedding-endpoint-batch-size <count>", "texts per remote embedding request (default: 64)", integerOption("embedding-endpoint-batch-size", 1, 1024))
	.option("--embedding-endpoint-concurrency <count>", "concurrent remote embedding requests (default: 4)", integerOption("embedding-endpoint-concurrency", 1, 32))
	.option("--embedding-endpoint-timeout <seconds>", "remote request timeout before retrying (default: 30)", integerOption("embedding-endpoint-timeout", 1, 600))
	.option("--vector-oversample <count>", "binary vector coarse candidates per final candidate (default: 4)", integerOption("vector-oversample", 1, 64))
	// Parse errors are reported once, by the handler at the end, after the global options (and so --json) are known.
	.exitOverride()
	.configureOutput({ outputError: () => undefined });

for (const commandName of ["init", "update"] as const) {
	addUpdateOptions(program.command(commandName).description(commandName === "init" ? "Initialize a local documentation index" : "Incrementally update a local documentation index"))
		.action(async (options, command) => withContext(command, async (context) => {
			const json = (command as Command).optsWithGlobals<GlobalOptions>().json;
			const result = await context.update({
				family: options.family,
				branch: options.branch,
				area: options.area,
				refresh: Boolean(options.refresh),
				concurrency: options.concurrency,
				limit: options.limit,
				onProgress: json ? undefined : (message) => console.error(message),
			});
			output(command, result);
		}));
}

program.command("search")
	.description("Run hybrid semantic and full-text search")
	.argument("<query>", "natural-language search query")
	.option("-n, --limit <count>", "maximum results", integerOption("limit", 1, MAX_SEARCH_LIMIT), DEFAULT_SEARCH_LIMIT)
	.option("-f, --family <family>", "filter by family release")
	.addOption(new Option("--doc-type <type>", "filter by document type").choices(DOC_TYPES))
	.option("--publication <publication>", "filter by publication")
	.addOption(new Option("--chunk-type <type>", "filter by chunk type").choices(CHUNK_TYPES))
	.option("--topic-type <type>", "filter by topic type")
	.option("--threshold <number>", "minimum cosine similarity for semantic candidates; lexical matches remain eligible", numberOption("threshold", -1, 1), DEFAULT_SIMILARITY_THRESHOLD)
	.option("--deduplicate-releases", "return only the best-scoring release of each source chunk")
	.option("--max-per-source <count>", "maximum results from one source document", integerOption("max-per-source", 1, MAX_RESULTS_PER_SOURCE), DEFAULT_MAX_RESULTS_PER_SOURCE)
	.option("--rerank", "rescore the top candidates with a local cross-encoder (downloads a small model on first use)")
	.action(async (query, options, command) => withContext(command, async (context) => {
		output(command, await context.search(query, {
			limit: options.limit,
			threshold: options.threshold,
			deduplicateReleases: Boolean(options.deduplicateReleases),
			maxResultsPerSource: options.maxPerSource,
			rerank: Boolean(options.rerank),
			release: options.family,
			docType: options.docType as DocType | undefined,
			publication: options.publication,
			chunkType: options.chunkType as ChunkType | undefined,
			topicType: options.topicType,
		}));
	}));

program.command("get")
	.description("Fetch all chunks for a source document")
	.argument("<source-path>", "path inside ServiceNowDocs")
	.option("-f, --family <family>", "filter by family release")
	.option("--outline", "return headings and previews only")
	.action(async (sourcePath, options, command) => withContext(command, (context) => {
		output(command, options.outline ? context.getDocumentOutline(sourcePath, options.family) : context.getDocument(sourcePath, options.family));
	}));

program.command("publications")
	.description("List indexed publications")
	.option("-f, --family <family>", "filter by family release")
	.action(async (options, command) => withContext(command, (context) => output(command, context.listPublications(options.family))));

program.command("status")
	.description("Show local index status and model configuration")
	.action(async (_options, command) => withContext(command, (context) => output(command, context.status())));

program.command("reset-index")
	.description("Remove the SQLite index while preserving models, the documentation clone, and the embedding cache")
	.requiredOption("--yes", "confirm destructive index removal")
	.option("--clear-embedding-cache", "also remove the embedding cache, so the next build embeds every passage again")
	.action(async (options, command) => {
		const globals = (command as Command).optsWithGlobals<GlobalOptions>();
		const paths = resolvePaths(globals.dataDir);
		let embeddingCache: HarvestResult | { removed: string } | undefined;
		if (options.clearEmbeddingCache) {
			await Promise.all([paths.embeddingCache, `${paths.embeddingCache}-shm`, `${paths.embeddingCache}-wal`].map((filename) => fs.rm(filename, { force: true })));
			embeddingCache = { removed: paths.embeddingCache };
		} else if (globals.embeddingCache && !globals.deterministicEmbeddings) {
			// Saves the index's vectors first, so the rebuild only embeds passages whose text changed.
			embeddingCache = harvestIndexEmbeddings(paths.database, paths.embeddingCache, globals.json ? undefined : (message) => console.error(message));
		}
		await Promise.all([paths.database, `${paths.database}-shm`, `${paths.database}-wal`].map((filename) => fs.rm(filename, { force: true })));
		output(command, {
			removed: paths.database,
			preserved: [paths.models, paths.repository, ...(options.clearEmbeddingCache ? [] : [paths.embeddingCache])],
			...(embeddingCache ? { embeddingCache } : {}),
		});
	});

program.command("skill")
	.description("Print the bundled AI skill or its installed path")
	.option("--path", "print the installed SKILL.md path")
	.action(async (options) => {
		const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
		const skillPath = path.join(packageRoot, "skills", "nowdev-ai-toolbox-documentationsearch", "SKILL.md");
		console.log(options.path ? skillPath : await fs.readFile(skillPath, "utf8"));
	});

program.command("mcp")
	.description("Start the stdio MCP server")
	.action(async (_options, command) => {
		const { startMcpServer } = await import("./mcp.js");
		await startMcpServer(contextOptions(command));
	});

program.parseAsync().catch((error: unknown) => {
	// Help and --version also arrive here, as CommanderErrors that already printed their output and exit with code 0.
	if (error instanceof CommanderError && error.exitCode === 0) return;
	const parseError = error instanceof CommanderError;
	const message = error instanceof Error ? error.message : String(error);
	// A parse error can stop option processing before --json is reached, so only then is argv consulted directly.
	if (program.opts<GlobalOptions>().json || (parseError && process.argv.includes("--json"))) {
		console.error(JSON.stringify({ error: message.replace(/^error:\s*/, "") }));
	} else {
		console.error(message);
		if (parseError) console.error("(add --help for additional information)");
	}
	process.exitCode = parseError ? error.exitCode : 1;
});