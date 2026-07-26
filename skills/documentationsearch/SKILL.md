---
name: documentationsearch
description: Search, retrieve, initialize, and update locally indexed ServiceNow documentation with the DocumentationSearch CLI or MCP tools. Use for questions about ServiceNow APIs, scripting, product behavior, release-specific documentation, method signatures, parameters, examples, or documentation availability.
---

# DocumentationSearch

Use DocumentationSearch as the documentation source for ServiceNow questions. Ground answers in indexed documents and identify the release when it matters.

## Choose an interface

Prefer these MCP tools when they are available:

- `get_documentation_search_status`
- `search_servicenow_docs`
- `get_servicenow_document_outline`
- `get_servicenow_document`
- `list_servicenow_publications`
- `update_servicenow_docs`

Otherwise use the `documentationsearch` CLI. Put global options before the command and request JSON for machine-readable output:

```bash
documentationsearch --json status
documentationsearch --json search "GlideRecord query records" --family australia
```

If the package is not installed globally, substitute:

```bash
npx -y @nowdevaitoolbox/documentationsearch --json status
```

## Search workflow

1. Inspect status before searching:

   ```bash
   documentationsearch --json status
   ```

2. If the requested release is indexed, search with a release filter:

   ```bash
   documentationsearch --json search "natural-language query" --family australia --limit 10
   ```

3. Use the best result's exact `sourcePath` to inspect its structure:

   ```bash
   documentationsearch --json get "SOURCE_PATH" --family australia --outline
   ```

4. Retrieve full content only for the most relevant source:

   ```bash
   documentationsearch --json get "SOURCE_PATH" --family australia
   ```

5. Cite or name the source path and release in the answer. Distinguish documented behavior from inference.

Start with the default similarity threshold. If a plausible query returns nothing, retry once with `--threshold 0`, then judge relevance from title, heading, content, and similarity. Refine the query or use `--doc-type`, `--publication`, `--chunk-type`, or `--topic-type` filters when results are broad.

The threshold is a minimum cosine similarity for every returned result, including keyword matches. When searching multiple releases, use `--deduplicate-releases` if only the best-ranked release of each source chunk is needed.

## Initialize and update

If status shows no documents, ask before starting a large download unless the user explicitly requested indexing. For scripting-only coverage:

```bash
documentationsearch init --family australia --area scripting
```

For a complete family release:

```bash
documentationsearch init --family australia --area all-docs
```

Update an existing index incrementally:

```bash
documentationsearch update --family australia --area all-docs
```

Do not use `--refresh` for routine updates. It re-embeds every discovered document. Use `--limit 5` for smoke tests.

Valid areas are `all-docs`, `scripting`, `server`, `client`, and `scripts`. The option is `--family`, not `--famly`.

## Handle failures

Initialization and updates return a `failures` array. Each item identifies `sourcePath`, `stage`, and `error`. Successful documents are committed even when some documents fail. A later `update` retries failed documents automatically.

Report the failure count and relevant paths when indexing is incomplete. Do not claim complete release coverage while failures remain.

## Embedding consistency

Normal indexes use local `onnx-community/bge-small-en-v1.5-ONNX` embeddings with CLS pooling, normalized 384-dimensional vectors, and BGE's retrieval instruction on queries only. `--deterministic-embeddings` is only for tests and must be used consistently for both indexing and searching in a separate data directory. Never mix deterministic and BGE indexes.

When the configured model or pooling changes, rebuild the index. This removes SQLite data but preserves the model cache and shallow documentation clone:

```bash
documentationsearch reset-index --yes
documentationsearch init --family australia --area all-docs
```

The default data directory is platform-specific and is shown by `status`. On Linux it is normally `~/.cache/documentationsearch`. The legacy `SERVICECONTEXT_HOME` variable remains supported for existing caches.
