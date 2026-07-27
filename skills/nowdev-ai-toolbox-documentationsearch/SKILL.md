---
name: nowdev-ai-toolbox-documentationsearch
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

Otherwise use the `nowdev-ai-toolbox-documentationsearch` CLI. Put global options before the command and request JSON for machine-readable output:

```bash
nowdev-ai-toolbox-documentationsearch --json status
nowdev-ai-toolbox-documentationsearch --json search "GlideRecord query records" --family australia
```

Assume the npm-installed `@nowdevaitoolbox/nowdev-ai-toolbox-documentationsearch` package is available globally and use the `nowdev-ai-toolbox-documentationsearch` command directly.

## Search workflow

1. Inspect status before searching:

   ```bash
   nowdev-ai-toolbox-documentationsearch --json status
   ```

2. If the requested release is indexed, search with a release filter:

   ```bash
   nowdev-ai-toolbox-documentationsearch --json search "natural-language query" --family australia --limit 10
   ```

   Threshold handling:
   - Run the query with the default similarity threshold.
   - If the query returns zero results, retry once with `--threshold 0`.
   - If results are returned from the retry, manually assess relevance by title, heading, content, and similarity score.
   - If results are too broad, apply filters one at a time to narrow them: `--doc-type`, `--publication`, `--chunk-type`, or `--topic-type`.

   If the requested release is not indexed, inform the user which releases are available, ask whether to proceed with the closest available release or initialize the requested one, and do not guess at release-specific behavior.

3. Inspect the returned results before selecting a document. Similarity is a ranking signal, not proof of relevance. Choose the result whose title, heading, and content directly address the question; for broad topics, refine the query or add a filter instead of following an adjacent result. For initial topic discovery, use `--max-per-source 1` or `2` to reduce repeated chunks:

   ```bash
   nowdev-ai-toolbox-documentationsearch --json search "focused query" --family australia --limit 10 --max-per-source 1
   ```

4. Use the selected result's exact `sourcePath` to inspect its structure:

   ```bash
   nowdev-ai-toolbox-documentationsearch --json get "SOURCE_PATH" --family australia --outline
   ```

5. Retrieve full content only for the same source selected in step 4:

   ```bash
   nowdev-ai-toolbox-documentationsearch --json get "SOURCE_PATH" --family australia
   ```

6. Cite or name the source path and release in the answer. Distinguish documented behavior from inference. If no result directly supports the question after refinement, say that the indexed documentation did not establish the answer rather than treating an adjacent result as authoritative.

The threshold is a minimum cosine similarity for every returned result, including keyword matches. When searching multiple releases, use `--deduplicate-releases` if only the best-ranked release of each source chunk is needed.

## Initialize and update

If status shows no documents, ask before starting any download that uses --area all-docs, or any init not explicitly requested by the user in this conversation turn. Use JSON when the result will be inspected programmatically:

```bash
nowdev-ai-toolbox-documentationsearch --json init --family australia --area scripting
```

For scripting-only coverage:

```bash
nowdev-ai-toolbox-documentationsearch init --family australia --area scripting
```

For a complete family release:

```bash
nowdev-ai-toolbox-documentationsearch init --family australia --area all-docs
```

Update an existing index incrementally:

```bash
nowdev-ai-toolbox-documentationsearch update --family australia --area all-docs
```

Do not use `--refresh` for routine updates. It re-embeds every discovered document. Use `--limit 5` for smoke tests. Treat a non-zero exit code or any item in the returned `failures` array as incomplete indexing, even when documents were committed successfully; report the failure count and paths before using the index for release-wide claims.

Embedding uses the native ONNX Runtime CPU provider by default. On Windows, `--device dml` selects DirectML with a conservative default batch size; `--embedding-batch-size` can increase throughput when memory allows. Each passage sent to the model is capped at 2048 characters by default with `--embedding-max-characters`; full source content remains available through `get`. Changing the cap requires resetting and rebuilding the index because it changes document vectors. The provider automatically halves a batch after a native out-of-memory error and falls back to CPU if DirectML's device is lost or suspended. `--device webgpu` selects ONNX Runtime's experimental WebGPU provider, while `--device cpu` is the CPU fallback. Device selection changes where inference runs, not the vector dimensions, so changing devices does not require resetting an existing index. CPU inference defaults to using every logical core for ONNX Runtime's intra-op thread pool; use `--embedding-threads <count>` to cap that on a shared or resource-limited host.

Search combines vector nearest-neighbor retrieval with FTS5 keyword retrieval. FTS5 weights titles and headings more heavily, recognizes multi-word phrases, and API object/method identifiers receive an exact-match boost. The vector index is partitioned by release, so a `--family`-filtered search only scans that release's shard instead of ranking the whole index and filtering afterward. Results default to at most three chunks per source document; use `--max-per-source` to change that. Use `--deduplicate-releases` when searching across releases and only one release of each source chunk is needed.

The current search schema stores API object and method names as dedicated FTS5 fields and partitions the vector index by release. If the tool reports an older search schema, run `nowdev-ai-toolbox-documentationsearch reset-index --yes` and re-index the requested release.

Valid areas are `all-docs`, `scripting`, `server`, `client`, and `scripts`. The option is `--family`, not `--famly`.

## Handle failures

Initialization and updates return a `failures` array. Each item identifies `sourcePath`, `stage`, and `error`. Successful documents are committed even when some documents fail. A later `update` retries failed documents automatically.

Report the failure count and relevant paths when indexing is incomplete. Do not claim complete release coverage while failures remain.

## Embedding consistency

Normal indexes use the Transformers.js-compatible `Xenova/bge-base-en-v1.5` distribution of `BAAI/bge-base-en-v1.5` with CLS pooling and normalized 768-dimensional vectors. BGE passages receive no instruction prefix; searches receive `Represent this sentence for searching relevant passages: `. BGE has a 512-token maximum input, and topic/API chunks are prepared at paragraph or code-line boundaries under a budget derived from the embedding cap (256 characters of headroom below it) so chunking and the embedding cap can't drift out of sync. The default embedding input cap is 2048 characters. `--deterministic-embeddings` is only for tests and must be used consistently for both indexing and searching in a separate data directory. Never mix deterministic and BGE indexes.

When the configured model or pooling changes, rebuild the index. This removes SQLite data but preserves the model cache and shallow documentation clone:

```bash
nowdev-ai-toolbox-documentationsearch reset-index --yes
nowdev-ai-toolbox-documentationsearch init --family australia --area all-docs
```

The default data directory is platform-specific and is shown by `status`. On Linux it is normally `~/.cache/nowdev-ai-toolbox-documentationsearch`. Set `NOWDEV_AI_TOOLBOX_DOCUMENTATIONSEARCH_HOME` to override it.
