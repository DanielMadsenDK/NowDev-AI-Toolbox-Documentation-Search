#!/usr/bin/env node
import { Command, InvalidArgumentError, Option } from "commander";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { DEFAULT_MAX_EMBEDDING_CHARACTERS, PACKAGE_VERSION } from "./config.js";
import { resolvePaths } from "./config.js";
import { HashEmbeddingProvider, type EmbeddingDevice } from "./embedder.js";
import { DEFAULT_EMBEDDING_PROFILE, EMBEDDING_PROFILES, type EmbeddingProfileName } from "./embedding-profiles.js";
import { DocumentationSearch } from "./service-context.js";
import type { ChunkType, DocType } from "./types.js";

interface GlobalOptions {
	dataDir?: string;
	json?: boolean;
	deterministicEmbeddings?: boolean;
	device?: EmbeddingDevice;
	embeddingBatchSize?: number;
	embeddingMaxCharacters?: number;
	embeddingThreads?: number;
	embeddingProfile?: EmbeddingProfileName;
	embeddingDimensions?: number;
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

function createContext(command: Command): DocumentationSearch {
	const options = command.optsWithGlobals<GlobalOptions>();
	return new DocumentationSearch({
		dataDirectory: options.dataDir,
		embeddingDevice: options.device,
		embeddingBatchSize: options.embeddingBatchSize,
		embeddingMaxCharacters: options.embeddingMaxCharacters,
		embeddingThreads: options.embeddingThreads,
		embeddingProfile: options.embeddingProfile,
		embeddingDimensions: options.embeddingDimensions,
		embeddingProvider: options.deterministicEmbeddings ? new HashEmbeddingProvider() : undefined,
	});
}

function output(command: Command, value: unknown): void {
	const options = command.optsWithGlobals<GlobalOptions>();
	if (options.json || typeof value !== "string") console.log(JSON.stringify(value, null, 2));
	else console.log(value);
}

async function withContext(command: Command, action: (context: DocumentationSearch) => Promise<void> | void): Promise<void> {
	const context = createContext(command);
	try {
		await action(context);
	} finally {
		context.close();
	}
}

function addUpdateOptions(command: Command): Command {
	return command
		.option("-f, --family <family>", "ServiceNow family release", "australia")
		.option("-b, --branch <branch>", "Git branch (defaults to family)")
		.option("-a, --area <area>", "all-docs, scripting, server, client, or scripts", "all-docs")
		.option("--refresh", "re-download and re-index every discovered source")
		.option("--concurrency <count>", "concurrent downloads", integerOption("concurrency", 1, 64), 8)
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
	.addOption(new Option("--device <device>", "embedding execution device").choices(["cpu", "dml", "webgpu"]).default("cpu"))
	.option("--embedding-batch-size <count>", "texts per embedding inference batch (default: 32 CPU, 8 DirectML)", integerOption("embedding-batch-size", 1, 1024))
	.option(`--embedding-max-characters <count>`, `override the profile-specific passage cap (MiniLM: 1024; other built-in profiles: ${DEFAULT_MAX_EMBEDDING_CHARACTERS})`, integerOption("embedding-max-characters", 256, 1_000_000))
	.option("--embedding-threads <count>", "ONNX Runtime intra-op thread count for CPU inference (default: the host's logical core count)", integerOption("embedding-threads", 1, 1024))
	.configureOutput({
		outputError: (message, write) => {
			if (process.argv.includes("--json")) write(`${JSON.stringify({ error: message.trim().replace(/^error:\s*/, "") })}\n`);
			else write(message);
		},
	})
	.showHelpAfterError(!process.argv.includes("--json"));

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
	.option("-n, --limit <count>", "maximum results", integerOption("limit", 1, 50), 10)
	.option("-f, --family <family>", "filter by family release")
	.option("--doc-type <type>", "filter by document type")
	.option("--publication <publication>", "filter by publication")
	.option("--chunk-type <type>", "filter by chunk type")
	.option("--topic-type <type>", "filter by topic type")
	.option("--threshold <number>", "minimum cosine similarity, except exact API identifier matches", numberOption("threshold", -1, 1), 0.3)
	.option("--deduplicate-releases", "return only the best-scoring release of each source chunk")
	.option("--max-per-source <count>", "maximum results from one source document", integerOption("max-per-source", 1, 10), 3)
	.action(async (query, options, command) => withContext(command, async (context) => {
		output(command, await context.search(query, {
			limit: options.limit,
			threshold: options.threshold,
			deduplicateReleases: Boolean(options.deduplicateReleases),
			maxResultsPerSource: options.maxPerSource,
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
	.description("Remove the SQLite index while preserving models and the documentation clone")
	.requiredOption("--yes", "confirm destructive index removal")
	.action(async (_options, command) => {
		const paths = resolvePaths((command as Command).optsWithGlobals<GlobalOptions>().dataDir);
		await Promise.all([paths.database, `${paths.database}-shm`, `${paths.database}-wal`].map((filename) => fs.rm(filename, { force: true })));
		output(command, { removed: paths.database, preserved: [paths.models, paths.repository] });
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
	.action(async () => {
		const { startMcpServer } = await import("./mcp.js");
		await startMcpServer(program.opts<GlobalOptions>().dataDir);
	});

program.parseAsync().catch((error: unknown) => {
	const message = error instanceof Error ? error.message : String(error);
	console.error(process.argv.includes("--json") ? JSON.stringify({ error: message }) : message);
	process.exitCode = 1;
});