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

Assume the npm-installed `@danielmadsendk/nowdev-ai-toolbox-documentationsearch` package is available globally and use the `nowdev-ai-toolbox-documentationsearch` command directly.

Always pass a natural-language query as one shell argument. Use double quotes in Bash or single quotes in PowerShell, and keep the same quoting when constructing commands programmatically:

```powershell
& nowdev-ai-toolbox-documentationsearch --json search 'explain ServiceNow roles ACLs and security' --family australia
```

An unquoted multi-word query is parsed as multiple positional arguments and fails before search runs. Do not use an unquoted query as a benchmark result.

If neither MCP tools nor the CLI are accessible, inform the user that DocumentationSearch is unavailable and that any answer will be based on general training knowledge, not indexed ServiceNow documentation.

## Search workflow

1. Inspect status before searching:

   ```bash
   nowdev-ai-toolbox-documentationsearch --json status
   ```

2. If the requested release is indexed, follow this ordered search decision:

   1. Run an unfiltered baseline query with the original wording and release filter:

      ```bash
      nowdev-ai-toolbox-documentationsearch --json search "natural-language query" --family australia --limit 10
      ```

   2. If the baseline returns zero results, retry once with `--threshold 0`. Manually assess any retry results by title, heading, content, and similarity score; treat the retry as a probe, not as proof of relevance.

   3. After a baseline or retry produces a result set, classify the request:
      - API lookup: mentions an API class, method, signature, parameter, return value, or example.
      - Procedure lookup: asks how to configure, create, debug, schedule, secure, or administer something.
      - Concept lookup: asks what a ServiceNow term or feature means.

      Apply type-specific options only after inspecting the unfiltered result set. For API lookups, preserve the exact identifiers and use a focused probe:

      ```bash
      nowdev-ai-toolbox-documentationsearch --json search "GlideDateTime addDays method" --family australia --doc-type scripting-api --chunk-type method --limit 5 --max-per-source 1
      ```

      For procedure and concept lookups, keep the original wording and use diverse sources:

      ```bash
      nowdev-ai-toolbox-documentationsearch --json search "UI Policy make field mandatory" --family australia --limit 10 --max-per-source 1
      ```

      Refine ambiguous questions by adding the platform mechanism or artifact that controls the behavior. For example, distinguish a global mandatory Dictionary field from a conditional UI Policy, distinguish a generic date-time question from `GlideDateTime` methods such as `getValue` or `getDisplayValue`, and distinguish a generic CMDB import from CMDB Identification and Reconciliation (IRE). Preserve the original intent, but add one canonical ServiceNow term rather than replacing the question with unrelated keywords.

      For comparison or tradeoff questions, do not expect one result set to provide a balanced comparison of every candidate. Search each candidate separately using its canonical term, then compare the retrieved documentation. A result about Business Rules alone does not establish the tradeoffs for Flow Designer or Script Includes.

      When behavior may be release-specific, include the release or feature-generation term in a refinement probe and inspect the release-filtered content for availability limits. Australia documentation, for example, can state that a particular ATF workflow is supported only for replaying an existing request rather than creating a new request.

      If results are too broad, apply one filter at a time: `--publication`, `--chunk-type`, or `--topic-type`. Use `--doc-type` only after inspecting the unfiltered results and confirming that the index classifies the requested material that way. Do not force `--doc-type product-doc` merely because a query sounds procedural. Query rewrites and filters are probes, not automatic replacements for the baseline search.

   4. Validate candidates before using them. Inspect the returned results before selecting a document. Similarity is a ranking signal, not proof of relevance. Choose the result whose title, heading, and content directly address the question; for broad topics, refine the query or add a filter instead of following an adjacent result. For initial topic discovery, use `--max-per-source 1` or `2` to reduce repeated chunks. API results should match the requested object or method in `title`, `heading`, `objectName`, `methodName`, or `sourcePath`. Procedure results should contain the requested feature or operation in the title or heading. Do not treat a glossary entry, adjacent API, generic platform page, or page that only mentions the term as direct evidence.

      ```bash
      nowdev-ai-toolbox-documentationsearch --json search "focused query" --family australia --limit 10 --max-per-source 1
      ```

      If the highest-ranked candidate from the most recent probe fails validation, run one alternate probe that preserves the exact nouns and adds one specific synonym or API identifier. Compare the probes and select the candidate with the strongest direct lexical match, not merely the highest similarity. If no candidate passes validation after two probes, report that the local index did not establish the answer.

      Treat identical or near-identical titles returned from different publications as duplicate evidence, not independent confirmation. Prefer the result whose publication and source path best match the requested product area, and retrieve that source before answering.

      When an API query returns a related object instead of the requested one, such as `GlideElement` for a `GlideForm` query, treat that as an API-ownership warning. Inspect the returned metadata and either search the related object explicitly or state that the result is adjacent rather than answering the original question.

   If the requested release is not indexed, inform the user which releases are available, ask whether to proceed with the closest available release or initialize the requested one, and do not guess at release-specific behavior.

3. Use the selected result's exact `sourcePath` to inspect its structure:

   ```bash
   nowdev-ai-toolbox-documentationsearch --json get "SOURCE_PATH" --family australia --outline
   ```

4. Retrieve full content only for the same source selected in step 3:

   ```bash
   nowdev-ai-toolbox-documentationsearch --json get "SOURCE_PATH" --family australia
   ```

5. Cite or name the source path and release in the answer. Distinguish documented behavior from inference. If no result directly supports the question after refinement, say that the indexed documentation did not establish the answer rather than treating an adjacent result as authoritative.

The threshold is a minimum cosine similarity, except that exact API object or method keyword matches are retained below it. When searching multiple releases, use `--deduplicate-releases` if only the best-ranked release of each source chunk is needed.

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

The current search schema stores source text once per document, uses an external-content FTS5 index, stores API object and method names as dedicated search fields, and partitions the exact sqlite-vec index by release. If the tool reports an older search schema, run `nowdev-ai-toolbox-documentationsearch reset-index --yes` and re-index the requested release.

Valid areas are `all-docs`, `scripting`, `server`, `client`, and `scripts`. The option is `--family`, not `--famly`.

## Handle failures

Initialization and updates return a `failures` array. Each item identifies `sourcePath`, `stage`, and `error`. Successful documents are committed even when some documents fail. A later `update` retries failed documents automatically.

Report the failure count and relevant paths when indexing is incomplete. Do not claim complete release coverage while failures remain.

## Embedding consistency

New indexes default to the curated ONNX profile `all-minilm-l6-v2`, with mean pooling, normalized 384-dimensional vectors, and a 1024-character input cap near its 256-token limit. Other supported profiles include `bge-base-en-v1.5`, `nomic-embed-text-v1.5`, `nomic-embed-text-v1`, and `multilingual-e5-small`. The selected profile persists in the index and is loaded automatically by later commands. Topic and API chunks are prepared under a budget derived from that profile's embedding cap, with 256 characters of headroom. `--deterministic-embeddings` is only for tests and must be used consistently for indexing and searching in a separate data directory. Never mix embedding profiles in one index.

When the configured model or pooling changes, rebuild the index. This removes SQLite data but preserves the model cache and shallow documentation clone:

```bash
nowdev-ai-toolbox-documentationsearch reset-index --yes
nowdev-ai-toolbox-documentationsearch init --family australia --area all-docs
```

The default data directory is platform-specific and is shown by `status`. On Linux it is normally `~/.cache/nowdev-ai-toolbox-documentationsearch`. Set `NOWDEV_AI_TOOLBOX_DOCUMENTATIONSEARCH_HOME` to override it.
