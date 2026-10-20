# DocumentationSearch

`@danielmadsendk/nowdev-ai-toolbox-documentationsearch` downloads public ServiceNow documentation, builds a local semantic index, and exposes it through a Node.js API, CLI, or MCP server. PostgreSQL and hosted embedding credentials are not required.

## Requirements

- Node.js 22.13 or newer on an even-numbered (LTS) release line: 22, 24, or 26+
- Internet access during initial documentation and model download
- Several GB of free cache space for a full family; size varies by release and embedding profile. The embedding cache adds roughly 1 GB per 700,000 distinct passages with the default profile

DocumentationSearch uses Node's built-in `node:sqlite` module, so it does not depend on the `better-sqlite3` native addon or require its build toolchain. The `sqlite-vec` package supplies the platform-specific vector extension loaded by `node:sqlite`.

## Install

```bash
npm install --global @danielmadsendk/nowdev-ai-toolbox-documentationsearch
nowdev-ai-toolbox-documentationsearch init --family australia
```

The explicit `init` command creates a shallow, single-branch clone of ServiceNowDocs and downloads the quantized embedding model on first use. Nothing is downloaded by npm's installation lifecycle. Subsequent updates use one shallow Git fetch, compare Git blob SHAs, and embed only new or changed files, and within those files only passages that are not already in the embedding cache. The clone also provides a consistent local snapshot when thousands of chunks are being generated. If Git synchronization is unavailable, DocumentationSearch falls back to the GitHub tree and raw-content APIs with retry handling.

On Windows, the clone enables Git's long-path support so deeply nested documentation files can be checked out. If the local Git or Windows configuration still rejects a path, the source automatically falls back to GitHub API downloads.

Document failures are isolated. Successfully indexed documents are committed, while each failed source is returned in the command result with its `download`, `chunk`, or `embed` stage and error. Failed sources are not marked as current, so the next `update` retries them automatically. If a previously indexed document changes but its replacement fails, the previous indexed version remains available until a later retry succeeds.

## CLI

```bash
nowdev-ai-toolbox-documentationsearch update --family australia
nowdev-ai-toolbox-documentationsearch search "query active incidents assigned to a user" --family australia
nowdev-ai-toolbox-documentationsearch get markdown/api-reference/server-api-reference/c_GlideRecordAPI.md --family australia
nowdev-ai-toolbox-documentationsearch get markdown/api-reference/server-api-reference/c_GlideRecordAPI.md --outline
nowdev-ai-toolbox-documentationsearch publications --family australia
nowdev-ai-toolbox-documentationsearch status
```

Use `--area scripting` to index only scripting references, or `--limit 5` for a smoke test. Use `--json` for machine-readable output.

The default embedding profile for new indexes is `all-minilm-l6-v2`, which provides faster CPU indexing and 384-dimensional vectors. A new index can instead use one of the curated ONNX/Transformers.js profiles:

```bash
node dist/cli.js --embedding-profile multilingual-e5-small --data-dir ~/.cache/documentationsearch-e5 init --family australia
```

Supported profiles are `bge-base-en-v1.5`, `nomic-embed-text-v1.5`, `nomic-embed-text-v1`, `multilingual-e5-small`, `mxbai-embed-xsmall-v1`, and `all-minilm-l6-v2`. `mxbai-embed-xsmall-v1` is a retrieval-trained 384-dimensional model with the same six-layer cost as MiniLM but a 2048-character passage cap; on the bundled retrieval benchmark it indexed about twice as fast and produced a smaller index with equal top-10 coverage, at a slightly lower MRR than the MiniLM default. The profile fixes the model, vector dimensions, pooling, normalization preparation, and retrieval prefixes as required by its model card. Use a separate data directory for each profile. The selected profile is written to the index immediately so interrupted indexing cannot later resume with a different model.

`nomic-embed-text-v1.5` was trained with Matryoshka Representation Learning, so its 768-dimensional vectors can be truncated to a smaller width without a separate model. Pass `--embedding-dimensions <count>` (64-768) alongside that profile to build a smaller, cheaper-to-store index:

```bash
node dist/cli.js --embedding-profile nomic-embed-text-v1.5 --embedding-dimensions 256 --data-dir ~/.cache/documentationsearch-nomic-256 init --family australia
```

Like the profile choice, the effective dimension count is written to the index immediately, so later `update`/`search` calls against the same data directory don't need to repeat the flag. Profiles without Matryoshka training reject `--embedding-dimensions`.

Source preparation is streamed in bounded windows: documents are downloaded and chunked, embedded in shared batches sorted by token count, committed, and released before the next window is prepared. This keeps full-corpus indexing from retaining every parsed document in memory while preserving full detail embeddings.

Embedding runs on the native ONNX Runtime CPU provider by default. On Windows, DirectML is the most practical GPU option when your graphics driver supports it. DirectML defaults to a smaller batch and a token budget because transformer memory grows with both batch size and passage length; the provider automatically halves a batch after a native out-of-memory error:

```bash
nowdev-ai-toolbox-documentationsearch --device dml init --family australia --area all-docs
```

Use `--embedding-batch-size 16` to trade more GPU memory for throughput when the default is stable. Passage caps are profile-specific: MiniLM defaults to 1024 characters near its 256-token limit, while the other built-in profiles default to 2048. The full source remains available through `get`. Adjust this with `--embedding-max-characters`. Changing the cap requires rebuilding the index with `reset-index --yes` because it changes document vectors. The device only affects `init` and `update`: searches, including the MCP server's, always embed queries on the CPU with the index's model and dtype. A batch that runs out of GPU memory is halved. Any other GPU failure during indexing (a lost or suspended device, a driver fault, or an operator the provider cannot run) moves the remaining work to the CPU workers with the same model and dtype, and `status` and the update summary report the device transition. GPU provider initialization is environment-dependent; `--device cpu` selects the native CPU path explicitly.

`--device webgpu` uses ONNX Runtime's WebGPU provider: Metal on macOS, and Direct3D 12 or Vulkan on Windows and Linux. A new index built with it defaults to fp16 (`--embedding-dtype fp16`), because the q8 model's quantized operators run slower on WebGPU than on the CPU. The dtype is part of the index, so use a separate data directory for a GPU-built index. The index remembers its dtype, so later `update`, `search`, and MCP calls pick it up without flags:

```bash
nowdev-ai-toolbox-documentationsearch --device webgpu --data-dir ~/.cache/documentationsearch-gpu init --family australia
nowdev-ai-toolbox-documentationsearch --data-dir ~/.cache/documentationsearch-gpu search "GlideRecord addQuery" --family australia
```

On an M5 MacBook Air, Metal fp16 indexed about as fast as the default CPU workers (roughly 390 chunks per second), while using about a third of one CPU core, which keeps the machine responsive during a long build. Search quality matched the q8 index. The CPU workers remain the default on every platform. ONNX Runtime still marks the WebGPU provider as experimental: with onnxruntime-node 1.24.3, a Metal run triggered a GPU fault that froze the macOS desktop, while 1.30.0, which fixes several WebGPU out-of-bounds reads, ran cleanly in repeated tests. ONNX Runtime's CoreML provider (Neural Engine) is not supported; it crashed in testing.

CPU indexing runs the model in parallel worker threads, each with its own single-threaded ONNX Runtime session. ONNX Runtime's intra-op threading barely speeds up a small encoder, while independent sessions scale with cores. `--embedding-threads <count>` sets the total CPU budget (default: the host's logical core count), and `--embedding-workers <count>` sets how many workers share it. The default is one worker per thread, at most 8, and no more than fit in about 35% of physical memory, because each worker holds its own copy of the model: roughly 300 MB for MiniLM but about 1.5 GB for the 768-dimensional nomic and BGE profiles. On a 16 GB machine that allows 8 MiniLM workers but 3 nomic workers; six nomic workers pushed a 16 GB Mac into heavy swapping and cut throughput tenfold. `--embedding-workers 1` embeds in one in-process session. Lower `--embedding-threads` on a shared server or in a container whose CPU limit is below the host's core count. Search queries use a separate in-process session, so a search during an MCP update doesn't wait behind indexing. Workers shut down after about ten seconds without work. A GPU device (`dml` or `webgpu`) embeds in one in-process session; after a GPU failure, the remaining work moves to the CPU workers.

Document vectors are also kept in a content-addressed embedding cache, `embedding-cache.sqlite` in the data directory. Entries are keyed by the exact passage text and the embedding configuration: model, dtype, pooling, dimensions, document prefix, and passage cap. Indexing looks up every passage there first and sends only the misses to the model. That makes `update --refresh`, documents where a change leaves most passages untouched, a second family that repeats most of the text, and rebuilds after `reset-index` skip most or all of the embedding work. `reset-index` copies the index's vectors into the cache before deleting the index; indexes built before database schema 3 keep no reusable vectors and are skipped. Use `--no-embedding-cache` to embed everything, and `reset-index --yes --clear-embedding-cache` to delete the cache. Entries are never evicted automatically; `status` reports the cache size. The cache is off for a custom `embeddingProvider`, including `--deterministic-embeddings`, unless the `embeddingCache: true` option is passed.

### Remote document embedding

Document embedding during `init` or `update` can be offloaded to an OpenAI-compatible embeddings endpoint such as OpenRouter. The API key is read from an environment variable; it is never accepted as a CLI value, written to the index, or included in status output.

```powershell
$env:OPENROUTER_API_KEY = "<your key>"
nowdev-ai-toolbox-documentationsearch `
  --embedding-profile bge-base-en-v1.5 `
  --embedding-endpoint https://openrouter.ai/api/v1/embeddings `
  --embedding-endpoint-model baai/bge-base-en-v1.5 `
  init --family australia
```

Use `--embedding-api-key-env <name>` to read the key from a different environment variable. Each request contains up to 64 texts, and four requests run concurrently by default. Tune these independently with `--embedding-endpoint-batch-size <count>` and `--embedding-endpoint-concurrency <count>`. Higher concurrency can improve throughput when the endpoint has spare capacity, but can trigger provider throttling or upstream stalls; use 4 unless the endpoint is known to handle more. Requests time out after 30 seconds and retry transient failures; override this with `--embedding-endpoint-timeout <seconds>`. The endpoint URL must use HTTPS, except for local `http://localhost` services, and cannot contain credentials or query parameters.

Remote tokenizers can occasionally count a token-dense passage above the model limit even when it fits the profile's character cap. A context-limit response causes the client to isolate that passage, shorten it with headroom, and retry it automatically instead of skipping the document.

Only document batches are sent to the endpoint. Query embeddings continue to use the selected local profile, so later `search` and MCP lookups work fully offline and do not need the endpoint flags or API key. The endpoint model must produce the same embedding space, dimensions, prefixes, and normalization behavior as the selected local profile. For example, OpenRouter's `baai/bge-base-en-v1.5` corresponds to the local `bge-base-en-v1.5` profile. Mixing merely same-sized but different models produces invalid semantic scores and requires rebuilding the index.

Search `--limit` accepts integers from 1 to 50. `--threshold` is the minimum cosine similarity for semantic candidates and accepts values from -1 to 1; lexical FTS5 candidates remain eligible independently, so exact terms and API identifiers are not lost when their vector similarity is low. The keyword leg removes stopwords, splits camelCase and dotted identifiers (`gs.getProperty` also matches `GlideSystem` and `get property`), expands common ServiceNow acronyms such as `ACL`, `ATF`, `BR`, `SI`, and `CMDB`, and requires all remaining terms before relaxing the most frequent ones, so ordinary questions no longer match most of the index. Results use hybrid reciprocal-rank fusion, weighted FTS5 title, heading, API identifier, and breadcrumb matches, exact API object/method boosts, and query-aware chunk-type weighting. Identical passages republished under several publications are collapsed to one result. Results are limited to three chunks per source by default; adjust this with `--max-per-source` or use `--max-per-source 1` for more diverse results. Use `--deduplicate-releases` to retain only the highest-ranked release of each source path and chunk index when searching across multiple releases. Unrestricted semantic search divides its candidate budget across indexed releases; filtering by family searches that release directly and is preferable when the target family is known. For binary coarse vector retrieval, `--vector-oversample` controls the number of coarse candidates per final semantic candidate (default: 4; range: 1-64).

`--rerank` rescores the top 30 fused candidates with a local cross-encoder (`jinaai/jina-reranker-v1-turbo-en`, downloaded on first use). It is off by default: on the bundled retrieval benchmark it helps some how-to, glossary, and parameter questions but hurts others, and it adds several hundred milliseconds of CPU time per search.

The current index schema (database schema 3) indexes cleaned passage text: HTML tables become `Header: value` rows, markdown escapes, links, and image placeholders are removed, and each passage is prefixed with its title, breadcrumb path, H2/H3 section path, and product. Page-level frontmatter is stored once per source instead of on every chunk. FTS5 uses Porter stemming and a weighted breadcrumb/heading context column. The vector index is partitioned by release and stores document type, publication, chunk type, and topic type as sqlite-vec metadata so supplied equality filters are applied before nearest-neighbor candidates are selected. For dimensions divisible by eight, sqlite-vec holds binary-quantized codes for candidate selection and full-precision vectors are kept in a plain table for exact cosine re-scoring. Long-running hosts such as the MCP server load the binary codes into memory at startup, which makes candidate selection a millisecond-scale scan. Existing indexes from earlier schema versions must be removed and rebuilt. Where passage text is unchanged, the rebuild reuses the vectors that `reset-index` saved to the embedding cache:

```bash
nowdev-ai-toolbox-documentationsearch reset-index --yes
nowdev-ai-toolbox-documentationsearch init --family australia --area all-docs
```

Data is stored in the operating system cache directory. Set `NOWDEV_AI_TOOLBOX_DOCUMENTATIONSEARCH_HOME` or pass `--data-dir` to choose another location. Set `GITHUB_TOKEN` when unauthenticated GitHub API rate limits are too restrictive.

## Node API

```ts
import { DocumentationSearch } from "@danielmadsendk/nowdev-ai-toolbox-documentationsearch";

const documentationSearch = new DocumentationSearch();
await documentationSearch.update({ family: "australia", area: "scripting" });

const results = await documentationSearch.search("GlideRecord pagination", {
  release: "australia",
  limit: 10,
});

await documentationSearch.close();
```

`DocumentationSearch` accepts a custom `EmbeddingProvider`, allowing hosted or organization-specific models without changing storage or ranking. It also accepts `embeddingEndpoint`, `embeddingEndpointModel`, and `embeddingEndpointApiKey` options to offload document embedding while retaining local query embedding. `embeddingThreads` and `embeddingWorkers` set the CPU embedding budget, and `embeddingCache: false` turns off the embedding cache. `update` takes an `AbortSignal` as `signal` to stop between documents, and `close()` resolves once the embedding and reranker models are released. Changing embedding dimensions requires a separate data directory or rebuilding the index.

## Retrieval benchmark

`eval/retrieval/queries.yaml` holds golden queries with their expected source documents, grouped into categories: exact API identifiers, natural-language API questions, how-to tasks, acronyms, glossary terms, parameter/return questions, and product concepts. Run them against an existing index to measure ranking changes before and after tuning:

```bash
npm run eval:retrieval -- --data-dir ~/.cache/nowdev-ai-toolbox-documentationsearch --out results.json
```

The report prints hit@1/5/10, MRR@10, and p50/p95 latency per category, lists the misses with their top results, and writes the full per-query outcome to `--out` for diffing. It also lists unreachable targets, meaning expected pages, methods, or headings that are missing from the index. Such a query cannot be answered whatever the ranking, so an entry there points at a broken golden entry or a chunking bug rather than at search quality. Pass `--search-options '{"rerank":true}'` to evaluate reranking, or `--filter <category>` to run one category.

## MCP

Initialize the index before starting an MCP client. When the package is installed globally, configure the client to launch its global executable directly:

```json
{
  "mcpServers": {
    "nowdev-ai-toolbox-documentationsearch": {
      "command": "nowdev-ai-toolbox-documentationsearch",
      "args": ["mcp"]
    }
  }
}
```

This starts one persistent stdio MCP process which remains available until the MCP client shuts it down. The global npm binary directory must be on `PATH`; run `npm prefix -g` to locate it if the client cannot find the command. For an on-demand package launch instead, use `npx -y @danielmadsendk/nowdev-ai-toolbox-documentationsearch mcp`.

Available tools:

- `search_servicenow_docs`
- `get_servicenow_document`
- `get_servicenow_document_outline`
- `list_servicenow_publications`
- `get_documentation_search_status`
- `update_servicenow_docs`

`search_servicenow_docs` returns compact hits by default: title, heading, snippet, source path, release, breadcrumb, ranking data, and the public servicenow.com URL when the source provides one. Results are returned both as JSON text and as `structuredContent` matching the tool's declared output schema, followed by `resource_link` entries for each matched document. Those links use the `servicenow-doc://{release}/{sourcePath}` resource template, which clients can read to fetch the complete indexed document. Pass `rerank: true` to rescore the top candidates with the local cross-encoder. `update_servicenow_docs` sends MCP progress notifications when the client supplies a progress token. Pass `includeMetadata: true` only when API parameters, examples, and other detailed parsed metadata are needed. The `docType` filter is the indexed source classification, so leave it unset for broad or procedural questions; for example, Scripted REST API setup guidance is classified as `developer-guide`, not `rest-api`.

## AI Skill

The npm package includes an agent skill at `skills/nowdev-ai-toolbox-documentationsearch/SKILL.md`. Print its installed path with:

```bash
nowdev-ai-toolbox-documentationsearch skill --path
```

Print the complete skill to standard output with:

```bash
nowdev-ai-toolbox-documentationsearch skill
```

For agents that discover project skills from `.github/skills`, install it in a workspace with:

```bash
mkdir -p .github/skills/nowdev-ai-toolbox-documentationsearch
nowdev-ai-toolbox-documentationsearch skill > .github/skills/nowdev-ai-toolbox-documentationsearch/SKILL.md
```

The skill teaches agents to inspect index status, use release-filtered hybrid search, retrieve full source documents, update documentation safely, interpret partial indexing failures, and prefer the MCP tools when available.

## Storage and Search

The local database uses Node's built-in `node:sqlite`, an external-content FTS5 index with Porter stemming for keyword retrieval, and `sqlite-vec` for vector candidate selection. Full source text and page-level frontmatter are stored once per source document; per-chunk rows retain compact searchable content and only their chunk-specific metadata. The vector table is partitioned by release, and supplied metadata equality filters are evaluated inside sqlite-vec before candidate selection. Binary-quantized codes select candidates and full-precision vectors from a plain rowid table re-score them with exact cosine distance. Hot queries pin their join order so the FTS5 or vector virtual table always drives the loop. Reciprocal Rank Fusion, weighted BM25 fields, exact normalized API identifier matching, and query-aware chunk-type weighting produce the final result order. FTS5 segments are optimized after non-empty indexing updates.

The default model for new indexes is `Xenova/all-MiniLM-L6-v2`, a Transformers.js-compatible quantized ONNX model using mean pooling and normalized 384-dimensional embeddings. Its profile uses a conservative 1024-character input cap near the model's 256-token sentence-transformer limit. BGE remains available with `--embedding-profile bge-base-en-v1.5` for normalized 768-dimensional embeddings and a 2048-character cap. Model files are cached and reused.

Indexing streams bounded source windows into shared model batches. Chunks are sized for the active profile, looked up in the embedding cache, and the remaining passages are grouped by token count to reduce transformer padding and spread across the embedding workers. Vectors are then restored to their original document order. Each window is committed and released before the next is prepared, and progress is reported throughout the embedding pass.

An index created with another model or pooling strategy must be rebuilt. Preserve the documentation clone, downloaded models, and embedding cache while removing only the SQLite index files with:

```bash
nowdev-ai-toolbox-documentationsearch reset-index --yes
nowdev-ai-toolbox-documentationsearch init --family australia --area all-docs
```

For lightweight tests, `--deterministic-embeddings` uses a deterministic hash vector. It is not a substitute for semantic embeddings and must use a separate data directory from the default model.