# DocumentationSearch

`@nowdevaitoolbox/documentationsearch` downloads public ServiceNow documentation, builds a local semantic index, and exposes it through a Node.js API, CLI, or MCP server. PostgreSQL and hosted embedding credentials are not required.

## Requirements

- Node.js 20.12 or newer
- Internet access during initial documentation and model download
- Approximately 500 MB to 1 GB of free cache space for a full family

## Install

```bash
npm install --global @nowdevaitoolbox/documentationsearch
documentationsearch init --family australia
```

The explicit `init` command creates a shallow, single-branch clone of ServiceNowDocs and downloads the quantized embedding model on first use. Nothing is downloaded by npm's installation lifecycle. Subsequent updates use one shallow Git fetch, compare Git blob SHAs, and embed only new or changed files. The clone also provides a consistent local snapshot when thousands of chunks are being generated. If Git synchronization is unavailable, DocumentationSearch falls back to the GitHub tree and raw-content APIs with retry handling.

Document failures are isolated. Successfully indexed documents are committed, while each failed source is returned in the command result with its `download`, `chunk`, or `embed` stage and error. Failed sources are not marked as current, so the next `update` retries them automatically. If a previously indexed document changes but its replacement fails, the previous indexed version remains available until a later retry succeeds.

## CLI

```bash
documentationsearch update --family australia
documentationsearch search "query active incidents assigned to a user" --family australia
documentationsearch get markdown/api-reference/server-api-reference/c_GlideRecordAPI.md --family australia
documentationsearch get markdown/api-reference/server-api-reference/c_GlideRecordAPI.md --outline
documentationsearch publications --family australia
documentationsearch status
```

Use `--area scripting` to index only scripting references, or `--limit 5` for a smoke test. Use `--json` for machine-readable output.

Search `--limit` accepts integers from 1 to 50. `--threshold` is the minimum cosine similarity for every returned result, including keyword matches, and accepts values from -1 to 1. Use `--deduplicate-releases` to retain only the highest-ranked release for each source path and chunk index when searching across multiple releases.

Data is stored in the operating system cache directory. Set `DOCUMENTATIONSEARCH_HOME` or pass `--data-dir` to choose another location. `SERVICECONTEXT_HOME` remains supported as a legacy alias. Set `GITHUB_TOKEN` when unauthenticated GitHub API rate limits are too restrictive.

Legacy CLI binaries, environment variables, TypeScript class exports, and the old MCP status tool remain available as migration aliases. To use an existing cache that has not been moved, set `SERVICECONTEXT_HOME` to its path or move it to the new default cache directory before starting DocumentationSearch.

## Node API

```ts
import { DocumentationSearch } from "@nowdevaitoolbox/documentationsearch";

const documentationSearch = new DocumentationSearch();
await documentationSearch.update({ family: "australia", area: "scripting" });

const results = await documentationSearch.search("GlideRecord pagination", {
  release: "australia",
  limit: 10,
});

documentationSearch.close();
```

`DocumentationSearch` accepts a custom `EmbeddingProvider`, allowing hosted or organization-specific models without changing storage or ranking. `ServiceContext` remains exported as a deprecated compatibility alias. Changing embedding dimensions requires a separate data directory or rebuilding the index.

## MCP

Initialize the index before starting an MCP client, then configure the client to launch:

```json
{
  "mcpServers": {
    "documentationsearch": {
      "command": "npx",
      "args": ["-y", "@nowdevaitoolbox/documentationsearch", "mcp"]
    }
  }
}
```

Available tools:

- `search_servicenow_docs`
- `get_servicenow_document`
- `get_servicenow_document_outline`
- `list_servicenow_publications`
- `get_documentation_search_status`
- `update_servicenow_docs`

## AI Skill

The npm package includes an agent skill at `skills/documentationsearch/SKILL.md`. Print its installed path with:

```bash
documentationsearch skill --path
```

Print the complete skill to standard output with:

```bash
documentationsearch skill
```

For agents that discover project skills from `.github/skills`, install it in a workspace with:

```bash
mkdir -p .github/skills/documentationsearch
documentationsearch skill > .github/skills/documentationsearch/SKILL.md
```

The skill teaches agents to inspect index status, use release-filtered hybrid search, retrieve full source documents, update documentation safely, interpret partial indexing failures, and prefer the MCP tools when available.

## Storage and Search

inThe local SQLite database uses ordinary tables for metadata and update state, FTS5 for keyword retrieval, and `sqlite-vec` for cosine-distance vector retrieval. Reciprocal Rank Fusion and chunk-type weighting produce the final result order.

The default model is `onnx-community/bge-small-en-v1.5-ONNX`, executed locally through Transformers.js with CLS pooling and normalized 384-dimensional embeddings. Indexed passages are embedded without an instruction; searches prepend BGE's recommended `Represent this sentence for searching relevant passages: ` instruction. Model files are cached and reused. The underlying `BAAI/bge-small-en-v1.5` model is MIT licensed.

Indexing combines chunks from all changed documents into shared model batches. Chunks are grouped by approximate text length to reduce transformer padding, then vectors are restored to their original document order. Progress is reported throughout the embedding pass.

An index created with another model or pooling strategy must be rebuilt. Preserve the documentation clone and downloaded models while removing only SQLite index files with:

```bash
documentationsearch reset-index --yes
documentationsearch init --family australia --area all-docs
```

For lightweight tests, `--deterministic-embeddings` uses a deterministic hash vector. It is not a substitute for semantic embeddings and must use a separate data directory from the default model.