// Retrieval benchmark: runs the golden queries in eval/retrieval/queries.yaml against a real index and
// reports hit@k, MRR@10, and latency per category. Usage (after npm run build):
//   node scripts/eval-retrieval.ts --data-dir <dir> [--out results.json] [--search-options '{"rerank":false}'] [--profile <name>] [--dtype <dtype>]
import fs from "node:fs";
import path from "node:path";
import { parseArgs } from "node:util";
import YAML from "yaml";
import { DocumentationSearch, type EmbeddingProfileName, type SearchOptions, type SearchResult } from "../dist/index.js";

interface RelevantTarget {
  path: string;
  method?: string;
  heading?: string;
}

interface GoldenQuery {
  id: string;
  category: string;
  query: string;
  relevant: RelevantTarget[];
}

interface QueryOutcome {
  id: string;
  category: string;
  query: string;
  rank: number | null;
  documentRank: number | null;
  milliseconds: number;
  top: string[];
}

const { values } = parseArgs({
  options: {
    "data-dir": { type: "string" },
    queries: { type: "string", default: path.join(import.meta.dirname, "..", "eval", "retrieval", "queries.yaml") },
    out: { type: "string" },
    "search-options": { type: "string", default: "{}" },
    profile: { type: "string" },
    dtype: { type: "string" },
    filter: { type: "string" },
    "vector-oversample": { type: "string" },
    "reranker-model": { type: "string" },
  },
});

function matches(result: SearchResult, target: RelevantTarget, documentOnly: boolean): boolean {
  if (result.sourcePath !== target.path) return false;
  if (documentOnly) return true;
  if (target.method && result.methodName?.toLowerCase() !== target.method.toLowerCase()) return false;
  if (target.heading && !result.heading?.toLowerCase().includes(target.heading.toLowerCase())) return false;
  return true;
}

function firstRank(results: SearchResult[], targets: RelevantTarget[], documentOnly: boolean): number | null {
  const index = results.findIndex((result) => targets.some((target) => matches(result, target, documentOnly)));
  return index === -1 ? null : index + 1;
}

function percentile(sorted: number[], fraction: number): number {
  if (!sorted.length) return 0;
  return sorted[Math.min(sorted.length - 1, Math.ceil(fraction * sorted.length) - 1)]!;
}

function summarize(outcomes: QueryOutcome[]) {
  if (!outcomes.length) return { queries: 0, hit1: 0, hit5: 0, hit10: 0, mrr10: 0, documentMrr10: 0, p50Milliseconds: 0, p95Milliseconds: 0 };
  const hit = (k: number) => outcomes.filter((outcome) => outcome.rank !== null && outcome.rank <= k).length / outcomes.length;
  const mrr = outcomes.reduce((total, outcome) => total + (outcome.rank ? 1 / outcome.rank : 0), 0) / outcomes.length;
  const documentMrr = outcomes.reduce((total, outcome) => total + (outcome.documentRank ? 1 / outcome.documentRank : 0), 0) / outcomes.length;
  const latencies = outcomes.map((outcome) => outcome.milliseconds).sort((left, right) => left - right);
  return {
    queries: outcomes.length,
    hit1: hit(1),
    hit5: hit(5),
    hit10: hit(10),
    mrr10: mrr,
    documentMrr10: documentMrr,
    p50Milliseconds: percentile(latencies, 0.5),
    p95Milliseconds: percentile(latencies, 0.95),
  };
}

const golden = YAML.parse(fs.readFileSync(values.queries!, "utf8")) as { release?: string; queries: GoldenQuery[] };
// Releases are stored lowercased, as search() lowercases its release filter.
const release = golden.release?.toLowerCase();
const selected = values.filter ? golden.queries.filter((query) => query.id.includes(values.filter!) || query.category === values.filter) : golden.queries;
const searchOptions = JSON.parse(values["search-options"]!) as SearchOptions;
const context = new DocumentationSearch({
  dataDirectory: values["data-dir"],
  ...(values["reranker-model"] ? { rerankerModel: values["reranker-model"] } : {}),
  ...(values["vector-oversample"] ? { vectorOversample: Number(values["vector-oversample"]) } : {}),
  ...(values.profile ? { embeddingProfile: values.profile as EmbeddingProfileName } : {}),
  ...(values.dtype ? { embeddingDtype: values.dtype as "q8" } : {}),
});
// A target missing from the index (page, method, or heading) makes its query unanswerable. That points at a broken
// golden entry or a chunking bug rather than at ranking: this check found a mistyped path and 89 GlideDateTime methods
// that a code-fence repair had folded into their neighbours.
// Without a golden release, a target counts as reachable in any indexed release (a NULL release matches every row).
const reachable = {
  source: context.database.db.prepare("SELECT 1 FROM sources WHERE (?1 IS NULL OR release = ?1) AND source_path = ?2 LIMIT 1"),
  method: context.database.db.prepare("SELECT 1 FROM documents WHERE (?1 IS NULL OR release = ?1) AND source_path = ?2 AND lower(method_name) = lower(?3) LIMIT 1"),
  heading: context.database.db.prepare("SELECT 1 FROM documents WHERE (?1 IS NULL OR release = ?1) AND source_path = ?2 AND lower(heading) LIKE '%' || lower(?3) || '%' LIMIT 1"),
};
const describe = (target: RelevantTarget) => `${target.path}${target.method ? `#${target.method}` : target.heading ? `#${target.heading}` : ""}`;
const unreachable = selected.flatMap((item) => item.relevant
  .filter((target) => !reachable.source.get(release ?? null, target.path)
    || (target.method && !reachable.method.get(release ?? null, target.path, target.method))
    || (target.heading && !reachable.heading.get(release ?? null, target.path, target.heading)))
  .map((target) => `${item.id}: ${describe(target)}`));
try {
  // Warm-up so model loading does not count toward the first query's latency.
  await context.search("warm up", { release, ...searchOptions });
  const outcomes: QueryOutcome[] = [];
  for (const item of selected) {
    const started = performance.now();
    const results = await context.search(item.query, { release, limit: 10, ...searchOptions });
    const milliseconds = performance.now() - started;
    outcomes.push({
      id: item.id,
      category: item.category,
      query: item.query,
      rank: firstRank(results, item.relevant, false),
      documentRank: firstRank(results, item.relevant, true),
      milliseconds,
      top: results.slice(0, 3).map((result) => `${result.sourcePath}${result.methodName ? `#${result.methodName}` : result.heading ? `#${result.heading}` : ""}`),
    });
  }
  const categories = [...new Set(outcomes.map((outcome) => outcome.category))];
  const report = {
    createdAt: new Date().toISOString(),
    embeddingProfile: context.status().embedding.profile,
    searchOptions,
    overall: summarize(outcomes),
    unreachable,
    categories: Object.fromEntries(categories.map((category) => [category, summarize(outcomes.filter((outcome) => outcome.category === category))])),
    outcomes,
  };
  const format = (value: number) => value.toFixed(3);
  console.log(`${"category".padEnd(16)} ${"n".padStart(3)} ${"hit@1".padStart(6)} ${"hit@5".padStart(6)} ${"hit@10".padStart(6)} ${"MRR".padStart(6)} ${"docMRR".padStart(6)} ${"p50ms".padStart(7)} ${"p95ms".padStart(7)}`);
  for (const [name, summary] of [...Object.entries(report.categories), ["overall", report.overall] as const]) {
    console.log(`${name.padEnd(16)} ${String(summary.queries).padStart(3)} ${format(summary.hit1).padStart(6)} ${format(summary.hit5).padStart(6)} ${format(summary.hit10).padStart(6)} ${format(summary.mrr10).padStart(6)} ${format(summary.documentMrr10).padStart(6)} ${summary.p50Milliseconds.toFixed(0).padStart(7)} ${summary.p95Milliseconds.toFixed(0).padStart(7)}`);
  }
  if (unreachable.length) {
    console.log(`\nUnreachable targets (${unreachable.length}), not in the index:`);
    for (const target of unreachable) console.log(`  ${target}`);
  }
  const misses = outcomes.filter((outcome) => outcome.rank === null);
  if (misses.length) {
    console.log(`\nMisses (${misses.length}):`);
    for (const miss of misses) console.log(`  ${miss.id}: "${miss.query}" -> ${miss.top.join(", ") || "(no results)"}`);
  }
  if (values.out) fs.writeFileSync(values.out, `${JSON.stringify(report, null, 2)}\n`);
} finally {
  await context.close();
}
