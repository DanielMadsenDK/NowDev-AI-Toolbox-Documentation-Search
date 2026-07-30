# DocumentationSearch

`@danielmadsendk/nowdev-ai-toolbox-documentationsearch` downloads public ServiceNow documentation, builds a local semantic index, and exposes it through a Node.js API, CLI, or MCP server. PostgreSQL and hosted embedding credentials are not required.

## Requirements

- Node.js 22.13 or newer
- Internet access during initial documentation and model download
- Several GB of free cache space for a full family; size varies by release and embedding profile

DocumentationSearch uses Node's built-in `node:sqlite` module, so it does not depend on the `better-sqlite3` native addon or require its build toolchain. The `sqlite-vec` package supplies the platform-specific vector extension loaded by `node:sqlite`.

## Install

```bash
npm install --global @danielmadsendk/nowdev-ai-toolbox-documentationsearch
nowdev-ai-toolbox-documentationsearch init --family australia
```

The explicit `init` command creates a shallow, single-branch clone of ServiceNowDocs and downloads the quantized embedding model on first use. Nothing is downloaded by npm's installation lifecycle. Subsequent updates use one shallow Git fetch, compare Git blob SHAs, and embed only new or changed files. The clone also provides a consistent local snapshot when thousands of chunks are being generated. If Git synchronization is unavailable, DocumentationSearch falls back to the GitHub tree and raw-content APIs with retry handling.

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

Supported profiles are `bge-base-en-v1.5`, `nomic-embed-text-v1.5`, `nomic-embed-text-v1`, `multilingual-e5-small`, and `all-minilm-l6-v2`. The profile fixes the model, vector dimensions, pooling, normalization preparation, and retrieval prefixes as required by its model card. Use a separate data directory for each profile. The selected profile is written to the index immediately so interrupted indexing cannot later resume with a different model.

`nomic-embed-text-v1.5` was trained with Matryoshka Representation Learning, so its 768-dimensional vectors can be truncated to a smaller width without a separate model. Pass `--embedding-dimensions <count>` (64-768) alongside that profile to build a smaller, cheaper-to-store index:

```bash
node dist/cli.js --embedding-profile nomic-embed-text-v1.5 --embedding-dimensions 256 --data-dir ~/.cache/documentationsearch-nomic-256 init --family australia
```

Like the profile choice, the effective dimension count is written to the index immediately, so later `update`/`search` calls against the same data directory don't need to repeat the flag. Profiles without Matryoshka training reject `--embedding-dimensions`.

Source preparation is streamed in bounded windows: documents are downloaded and chunked, embedded in length-sorted shared batches, committed, and released before the next window is prepared. This keeps full-corpus indexing from retaining every parsed document in memory while preserving full detail embeddings.

Embedding runs on the native ONNX Runtime CPU provider by default. On Windows, DirectML is the most practical GPU option when your graphics driver supports it. DirectML defaults to a smaller batch and a token budget because transformer memory grows with both batch size and passage length; the provider automatically halves a batch after a native out-of-memory error:

```bash
nowdev-ai-toolbox-documentationsearch --device dml init --family australia --area all-docs
```

Use `--embedding-batch-size 16` to trade more GPU memory for throughput when the default is stable. Passage caps are profile-specific: MiniLM defaults to 1024 characters near its 256-token limit, while the other built-in profiles default to 2048. The full source remains available through `get`. Adjust this with `--embedding-max-characters`. Changing the cap requires rebuilding the index with `reset-index --yes` because it changes document vectors. The provider halves a batch after an ordinary native out-of-memory error and falls back to CPU if DirectML suspends the GPU device. Use `--device webgpu` for ONNX Runtime's experimental WebGPU provider, or `--device cpu` to select the native CPU path explicitly. Device selection does not require rebuilding an existing index, but the same device option should be used for both indexing and searching when comparing performance. GPU provider initialization is environment-dependent; if it fails, rerun with `--device cpu`. `status` reports the active embedding device so a DirectML fallback is visible.

CPU inference (including after a DirectML fallback) uses the host's full logical core count for ONNX Runtime's intra-op thread pool by default. Override this with `--embedding-threads <count>` on machines where using every core isn't appropriate, such as a shared server or a container with a CPU limit lower than the host's core count.

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

Search `--limit` accepts integers from 1 to 50. `--threshold` is the minimum cosine similarity for semantic candidates and accepts values from -1 to 1; lexical FTS5 candidates remain eligible independently, so exact terms and API identifiers are not lost when their vector similarity is low. Results use hybrid reciprocal-rank fusion, weighted FTS5 title and heading matches, exact API object/method boosts, and query-aware chunk-type weighting. Results are limited to three chunks per source by default; adjust this with `--max-per-source` or use `--max-per-source 1` for more diverse results. Use `--deduplicate-releases` to retain only the highest-ranked release of each source path and chunk index when searching across multiple releases. Unrestricted semantic search divides its candidate budget across indexed releases; filtering by family searches that release directly and is preferable when the target family is known. For binary coarse vector retrieval, `--vector-oversample` controls the number of coarse candidates per final semantic candidate (default: 8; range: 1-64). Increase it to 16 when measured recall needs more protection at a modest re-scoring cost.

The current index schema stores API object and method names as dedicated FTS5 fields, partitions the vector index by release, and stores document type, publication, chunk type, and topic type as sqlite-vec metadata so supplied equality filters are applied before nearest-neighbor candidates are selected. For dimensions divisible by eight, it uses binary-quantized coarse candidates and full-float cosine re-scoring to reduce vector scan work without returning approximate distances. Topic and API chunks are prepared at paragraph or code-line boundaries under the active profile's input budget. Existing indexes from earlier schema versions must be removed and rebuilt:

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

documentationSearch.close();
```

`DocumentationSearch` accepts a custom `EmbeddingProvider`, allowing hosted or organization-specific models without changing storage or ranking. It also accepts `embeddingEndpoint`, `embeddingEndpointModel`, and `embeddingEndpointApiKey` options to offload document embedding while retaining local query embedding. Changing embedding dimensions requires a separate data directory or rebuilding the index.

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

`search_servicenow_docs` returns compact hits by default: title, heading, snippet, source path, release, ranking data, and the source URL. Pass `includeMetadata: true` only when API parameters, examples, and other detailed parsed metadata are needed. The `docType` filter is the indexed source classification, so leave it unset for broad or procedural questions; for example, Scripted REST API setup guidance is classified as `developer-guide`, not `rest-api`.

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

The local database uses Node's built-in `node:sqlite`, an external-content FTS5 index for keyword retrieval, and `sqlite-vec` for exact cosine-distance vector retrieval. Full source text is stored once per source document; per-chunk rows retain compact searchable content and structured metadata. The vector table is partitioned by release, and supplied metadata equality filters are evaluated inside sqlite-vec before candidate selection. Reciprocal Rank Fusion, weighted BM25 fields, exact normalized API identifier matching, and query-aware chunk-type weighting produce the final result order. FTS5 segments are optimized after non-empty indexing updates.

The default model for new indexes is `Xenova/all-MiniLM-L6-v2`, a Transformers.js-compatible quantized ONNX model using mean pooling and normalized 384-dimensional embeddings. Its profile uses a conservative 1024-character input cap near the model's 256-token sentence-transformer limit. BGE remains available with `--embedding-profile bge-base-en-v1.5` for normalized 768-dimensional embeddings and a 2048-character cap. Model files are cached and reused.

Indexing streams bounded source windows into shared model batches. Chunks are sized for the active profile and grouped by approximate text length to reduce transformer padding, then vectors are restored to their original document order. Each window is committed and released before the next is prepared, and progress is reported throughout the embedding pass.

An index created with another model or pooling strategy must be rebuilt. Preserve the documentation clone and downloaded models while removing only SQLite index files with:

```bash
nowdev-ai-toolbox-documentationsearch reset-index --yes
nowdev-ai-toolbox-documentationsearch init --family australia --area all-docs
```

For lightweight tests, `--deterministic-embeddings` uses a deterministic hash vector. It is not a substitute for semantic embeddings and must use a separate data directory from the default model.