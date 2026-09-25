// Indexing benchmark: builds or updates an index for a fixed subset of sources and reports throughput. Run it with
// nothing else busy; parallel CPU work makes embedding look several times slower. Usage (after npm run build):
//   node scripts/bench-indexing.ts --data-dir <dir> [--limit 5000] [--family australia] [--workers <n>] [--threads <n>]
//     [--batch-size <n>] [--device cpu|webgpu|dml] [--dtype q8|fp16|fp32] [--no-cache] [--refresh] [--out results.json]
//     [--dist <path to a built index.js>]
import fs from "node:fs";
import path from "node:path";
import { parseArgs } from "node:util";
import { pathToFileURL } from "node:url";
import type { DocumentationSearch as DocumentationSearchClass, UpdateResult } from "../dist/index.js";

const { values } = parseArgs({
  options: {
    "data-dir": { type: "string" },
    limit: { type: "string" },
    family: { type: "string", default: "australia" },
    workers: { type: "string" },
    threads: { type: "string" },
    "batch-size": { type: "string" },
    device: { type: "string" },
    dtype: { type: "string" },
    "no-cache": { type: "boolean", default: false },
    refresh: { type: "boolean", default: false },
    out: { type: "string" },
    // Benchmarks another build (for example a saved copy of the previous release) with the same harness.
    dist: { type: "string", default: path.join(import.meta.dirname, "..", "dist", "index.js") },
  },
});

if (!values["data-dir"]) throw new Error("--data-dir is required");
const optionalInteger = (name: "limit" | "workers" | "threads" | "batch-size") => values[name] === undefined ? undefined : Number(values[name]);
const { DocumentationSearch } = await import(pathToFileURL(path.resolve(values.dist!)).href) as { DocumentationSearch: typeof DocumentationSearchClass };
const context = new DocumentationSearch({
  dataDirectory: values["data-dir"],
  embeddingWorkers: optionalInteger("workers"),
  embeddingThreads: optionalInteger("threads"),
  embeddingBatchSize: optionalInteger("batch-size"),
  ...(values.device ? { embeddingDevice: values.device as "cpu" } : {}),
  ...(values.dtype ? { embeddingDtype: values.dtype as "q8" } : {}),
  ...(values["no-cache"] ? { embeddingCache: false } : {}),
});
try {
  const started = performance.now();
  const result: UpdateResult = await context.update({
    family: values.family,
    limit: optionalInteger("limit"),
    refresh: values.refresh,
    // Per-source "Prepared" lines would drown the periodic throughput lines.
    onProgress: (message) => {
      if (!message.startsWith("Prepared ")) console.error(message);
    },
  });
  const seconds = (performance.now() - started) / 1000;
  const cachedChunks = result.cachedChunks ?? 0;
  const status = context.status();
  const report = {
    createdAt: new Date().toISOString(),
    dist: values.dist,
    limit: optionalInteger("limit") ?? null,
    workers: status.embedding.workers ?? 1,
    device: status.embedding.device,
    dtype: status.embedding.dtype ?? null,
    seconds: Number(seconds.toFixed(1)),
    sources: result.added + result.changed,
    chunks: result.chunks,
    cachedChunks,
    chunksPerSecond: Number((result.chunks / seconds).toFixed(1)),
    embeddedChunksPerSecond: Number(((result.chunks - cachedChunks) / seconds).toFixed(1)),
    failures: result.failures.length,
    metrics: status.embedding.metrics ?? null,
  };
  console.log(JSON.stringify(report, null, 2));
  if (values.out) fs.writeFileSync(values.out, `${JSON.stringify(report, null, 2)}\n`);
} finally {
  context.close();
}
