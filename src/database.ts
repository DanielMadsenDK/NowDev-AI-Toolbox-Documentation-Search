import fs from "node:fs";
import path from "node:path";
import Database from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";
import type { DocumentChunk, IndexManifest, SearchFilters, SearchResult } from "./types.js";

interface StoredSource {
  blobSha: string;
  contentHash: string;
}

interface SearchRow {
  id: number;
  doc_type: DocumentChunk["docType"];
  publication: string;
  source_path: string;
  release: string;
  chunk_type: DocumentChunk["chunkType"];
  chunk_index: number;
  title: string;
  heading: string | null;
  content: string;
  topic_type: string | null;
  product: string | null;
  classification: string | null;
  last_updated: string | null;
  object_name: string | null;
  method_name: string | null;
  metadata: string;
  content_hash: string;
  distance?: number;
  rank?: number;
}

function vectorBuffer(vector: Float32Array): Buffer {
  return Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength);
}

function ftsQuery(query: string): string {
  const terms = query.toLowerCase().match(/[a-z0-9_]+/g) ?? [];
  if (!terms.length) return "";
  const quotedTerms = terms.map((term) => `"${term.replaceAll('"', '""')}"`);
  const phrase = terms.length > 1 ? `"${terms.join(" ").replaceAll('"', '""')}"` : null;
  const allTerms = terms.length > 1 ? `(${quotedTerms.join(" AND ")})` : null;
  return [phrase, allTerms, ...quotedTerms].filter(Boolean).join(" OR ");
}

function queryTerms(query: string): string[] {
  return query.toLowerCase().match(/[a-z0-9_]+/g) ?? [];
}

function looksLikeApiQuery(query: string): boolean {
  return /\b(?:glide[a-z0-9_]*|sn_[a-z0-9_]+|api|method|parameter|argument|endpoint|script)\b/i.test(query)
    || /[A-Z_][A-Za-z0-9_]*/.test(query);
}

function identifierBoost(row: SearchRow, terms: Set<string>): number {
  const objectMatch = row.object_name !== null && terms.has(row.object_name.toLowerCase());
  const methodMatch = row.method_name !== null && terms.has(row.method_name.toLowerCase());
  if (objectMatch && methodMatch) return 1.8;
  if (methodMatch) return 1.6;
  if (objectMatch) return 1.15;
  return 1;
}

function chunkTypeMultiplier(row: SearchRow, query: string): number {
  const base: Partial<Record<DocumentChunk["chunkType"], number>> = { overview: 1, section: 0.95, definition: 0.85, method: 0.8, endpoint: 0.8, parameter: 0.5, returns: 0.5, example: 0.45 };
  const multiplier = base[row.chunk_type] ?? 0.9;
  if (/\b(?:parameter|argument|option|property|properties)\b/i.test(query) && row.chunk_type === "parameter") return multiplier * 1.35;
  if (/\b(?:return|returns|output|result|response|type)\b/i.test(query) && row.chunk_type === "returns") return multiplier * 1.35;
  if (/\b(?:example|sample|code|syntax)\b/i.test(query) && row.chunk_type === "example") return multiplier * 1.35;
  if (/\b(?:how|configure|setup|set up|use)\b/i.test(query) && (row.chunk_type === "overview" || row.chunk_type === "section")) return multiplier * 1.1;
  return multiplier;
}

function rowToChunk(row: SearchRow): DocumentChunk {
  return {
    id: row.id,
    docType: row.doc_type,
    publication: row.publication,
    sourcePath: row.source_path,
    release: row.release,
    chunkType: row.chunk_type,
    chunkIndex: row.chunk_index,
    title: row.title,
    heading: row.heading,
    content: row.content,
    topicType: row.topic_type,
    product: row.product,
    classification: row.classification,
    lastUpdated: row.last_updated,
    objectName: row.object_name,
    methodName: row.method_name,
    metadata: JSON.parse(row.metadata) as Record<string, unknown>,
    contentHash: row.content_hash,
  };
}

function filterValues(filters: SearchFilters): Array<string | null> {
  return [filters.release?.toLowerCase() ?? null, filters.docType ?? null, filters.publication ?? null, filters.chunkType ?? null, filters.topicType ?? null]
    .flatMap((value) => [value, value]);
}

export class DocumentationSearchDatabase {
  readonly db: Database.Database;

  constructor(readonly filename: string, readonly dimensions: number) {
    fs.mkdirSync(path.dirname(filename), { recursive: true });
    this.db = new Database(filename);
    this.db.pragma("journal_mode = WAL");
    this.db.pragma("foreign_keys = ON");
    this.db.pragma("synchronous = NORMAL");
    this.db.pragma("temp_store = MEMORY");
    this.db.pragma("cache_size = -64000");
    this.db.pragma("mmap_size = 268435456");
    sqliteVec.load(this.db);
    this.migrate();
  }

  private migrate(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS sources (
        release TEXT NOT NULL,
        source_path TEXT NOT NULL,
        blob_sha TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        PRIMARY KEY (release, source_path)
      );
      CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY,
        doc_type TEXT NOT NULL, publication TEXT NOT NULL, source_path TEXT NOT NULL,
        release TEXT NOT NULL, chunk_type TEXT NOT NULL, chunk_index INTEGER NOT NULL,
        title TEXT NOT NULL, heading TEXT, content TEXT NOT NULL, topic_type TEXT,
        product TEXT, classification TEXT, last_updated TEXT, object_name TEXT,
        method_name TEXT, metadata TEXT NOT NULL, content_hash TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(release, source_path);
      CREATE INDEX IF NOT EXISTS idx_documents_object_method ON documents(object_name, method_name);
      CREATE INDEX IF NOT EXISTS idx_documents_filters ON documents(release, doc_type, publication, chunk_type, topic_type);
      CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(title, heading, object_name, method_name, content);
    `);
    const ftsSchema = this.db.prepare("SELECT sql FROM sqlite_master WHERE name = 'documents_fts'").pluck().get() as string | undefined;
    if (!ftsSchema?.includes("object_name") || !ftsSchema.includes("method_name")) {
      throw new Error("Index uses an older search schema. Run reset-index --yes and re-index the documentation.");
    }
    const storedDimensions = this.db.prepare("SELECT value FROM settings WHERE key = 'dimensions'").pluck().get() as string | undefined;
    if (storedDimensions && Number(storedDimensions) !== this.dimensions) {
      throw new Error(`Index uses ${storedDimensions}-dimensional embeddings, but provider uses ${this.dimensions}. Remove the index or use a matching model.`);
    }
    this.db.prepare("INSERT OR IGNORE INTO settings(key, value) VALUES ('dimensions', ?)").run(String(this.dimensions));
    this.db.exec(`CREATE VIRTUAL TABLE IF NOT EXISTS document_vectors USING vec0(embedding float[${this.dimensions}] distance_metric=cosine, release text partition key)`);
    const vectorSchema = this.db.prepare("SELECT sql FROM sqlite_master WHERE name = 'document_vectors'").pluck().get() as string | undefined;
    if (!vectorSchema?.includes("distance_metric=cosine")) {
      throw new Error("Index uses the legacy L2 vector metric. Remove the index and run nowdev-ai-toolbox-documentationsearch init to rebuild it with cosine distance.");
    }
    if (!vectorSchema.includes("partition key")) {
      throw new Error("Index predates release-partitioned vector search. Run reset-index --yes and re-index the documentation.");
    }
  }

  close(): void {
    this.db.close();
  }

  sourceMap(release: string): Map<string, StoredSource> {
    const rows = this.db.prepare("SELECT source_path, blob_sha, content_hash FROM sources WHERE release = ?").all(release) as Array<{ source_path: string; blob_sha: string; content_hash: string }>;
    return new Map(rows.map((row) => [row.source_path, { blobSha: row.blob_sha, contentHash: row.content_hash }]));
  }

  replaceSources(release: string, sources: Array<{ path: string; blobSha: string; contentHash: string; chunks: DocumentChunk[]; embeddings: Float32Array[] }>, deletedPaths: string[]): void {
    const remove = this.db.transaction((sourcePaths: string[]) => {
      const ids = this.db.prepare("SELECT id FROM documents WHERE release = ? AND source_path = ?");
      const deleteVector = this.db.prepare("DELETE FROM document_vectors WHERE rowid = ?");
      const deleteFts = this.db.prepare("DELETE FROM documents_fts WHERE rowid = ?");
      const deleteDocs = this.db.prepare("DELETE FROM documents WHERE release = ? AND source_path = ?");
      const deleteSource = this.db.prepare("DELETE FROM sources WHERE release = ? AND source_path = ?");
      for (const sourcePath of sourcePaths) {
        for (const row of ids.all(release, sourcePath) as Array<{ id: number }>) {
          deleteVector.run(row.id);
          deleteFts.run(row.id);
        }
        deleteDocs.run(release, sourcePath);
        deleteSource.run(release, sourcePath);
      }
    });
    const insert = this.db.transaction(() => {
      remove([...deletedPaths, ...sources.map((source) => source.path)]);
      const insertDocument = this.db.prepare(`INSERT INTO documents (
        doc_type, publication, source_path, release, chunk_type, chunk_index, title, heading,
        content, topic_type, product, classification, last_updated, object_name, method_name, metadata, content_hash
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`);
      const insertFts = this.db.prepare("INSERT INTO documents_fts(rowid, title, heading, object_name, method_name, content) VALUES (?, ?, ?, ?, ?, ?)");
      const insertVector = this.db.prepare("INSERT INTO document_vectors(rowid, embedding, release) VALUES (?, ?, ?)");
      const insertSource = this.db.prepare("INSERT INTO sources(release, source_path, blob_sha, content_hash) VALUES (?, ?, ?, ?)");
      for (const source of sources) {
        if (source.chunks.length !== source.embeddings.length) throw new Error(`Embedding count mismatch for ${source.path}`);
        source.chunks.forEach((chunk, index) => {
          const result = insertDocument.run(chunk.docType, chunk.publication, chunk.sourcePath, chunk.release, chunk.chunkType, chunk.chunkIndex, chunk.title, chunk.heading, chunk.content, chunk.topicType, chunk.product, chunk.classification, chunk.lastUpdated, chunk.objectName, chunk.methodName, JSON.stringify(chunk.metadata), chunk.contentHash);
          const id = BigInt(result.lastInsertRowid);
          insertFts.run(id, chunk.title, chunk.heading, chunk.objectName, chunk.methodName, chunk.content);
          insertVector.run(id, vectorBuffer(source.embeddings[index]!), chunk.release);
        });
        insertSource.run(release, source.path, source.blobSha, source.contentHash);
      }
    });
    insert();
  }

  search(query: string, embedding: Float32Array, filters: SearchFilters, limit = 10, threshold = 0.3, deduplicateReleases = false, maxResultsPerSource = 3): SearchResult[] {
    const boundedLimit = Math.min(Math.max(limit, 1), 50);
    const boundedMaxPerSource = Math.min(Math.max(Math.floor(maxResultsPerSource), 1), 10);
    const candidates = Math.max(100, boundedLimit * 20);
    const filtersSql = `
      AND (? IS NULL OR d.release = ?)
      AND (? IS NULL OR d.doc_type = ?)
      AND (? IS NULL OR d.publication = ?)
      AND (? IS NULL OR d.chunk_type = ?)
      AND (? IS NULL OR d.topic_type = ?)`;
    // v.release (not d.release) so sqlite-vec recognizes the constraint as a partition-key filter and restricts the KNN scan to that release's shard instead of ranking the whole index and filtering afterward.
    const semanticFiltersSql = `
      AND (? IS NULL OR v.release = ?)
      AND (? IS NULL OR d.doc_type = ?)
      AND (? IS NULL OR d.publication = ?)
      AND (? IS NULL OR d.chunk_type = ?)
      AND (? IS NULL OR d.topic_type = ?)`;
    const filtersParameters = filterValues(filters);
    const semantic = this.db.prepare(`
      SELECT d.*, v.distance FROM document_vectors v JOIN documents d ON d.id = v.rowid
      WHERE v.embedding MATCH ? AND k = ?${semanticFiltersSql}
      ORDER BY v.distance LIMIT ?
    `).all(vectorBuffer(embedding), candidates, ...filtersParameters, candidates) as SearchRow[];
    const keywordExpression = ftsQuery(query);
    const keyword = keywordExpression ? this.db.prepare(`
      SELECT d.*, bm25(documents_fts, 10.0, 7.0, 9.0, 9.0, 1.0) AS rank, vec_distance_cosine(v.embedding, ?) AS distance
      FROM documents_fts f
      JOIN documents d ON d.id = f.rowid
      JOIN document_vectors v ON v.rowid = d.id
      WHERE documents_fts MATCH ?${filtersSql} ORDER BY f.rank LIMIT ?
    `).all(vectorBuffer(embedding), keywordExpression, ...filtersParameters, candidates) as SearchRow[] : [];
    const terms = new Set(queryTerms(query));
    const keywordWeight = looksLikeApiQuery(query) ? 1.25 : 1;
    const scores = new Map<number, { row: SearchRow; score: number; similarity: number }>();
    semantic.forEach((row, index) => {
      const similarity = 1 - (row.distance ?? 1);
      if (similarity < threshold) return;
      scores.set(row.id, { row, score: 1 / (60 + index + 1), similarity });
    });
    keyword.forEach((row, index) => {
      const similarity = 1 - (row.distance ?? 1);
      if (similarity < threshold) return;
      const current = scores.get(row.id);
      scores.set(row.id, { row, score: (current?.score ?? 0) + keywordWeight / (60 + index + 1), similarity: current?.similarity ?? similarity });
    });
    const ranked = [...scores.values()]
      .map(({ row, score, similarity }) => {
        const chunk = rowToChunk(row);
        const metadata = Object.fromEntries(Object.entries(chunk.metadata).filter(([key]) => key !== "full_content" && key !== "section_content"));
        return { ...chunk, metadata, content: row.content.slice(0, 1000), score: score * chunkTypeMultiplier(row, query) * identifierBoost(row, terms), similarity };
      })
      .sort((left, right) => right.score - left.score);
    const sourceCounts = new Map<string, number>();
    const deduplicationKeys = new Set<string>();
    const selected: SearchResult[] = [];
    for (const result of ranked) {
      if (deduplicateReleases) {
        const deduplicationKey = `${result.sourcePath}\u0000${result.chunkIndex}`;
        if (deduplicationKeys.has(deduplicationKey)) continue;
        deduplicationKeys.add(deduplicationKey);
      }
      const sourceKey = deduplicateReleases ? result.sourcePath : `${result.release}\u0000${result.sourcePath}`;
      const count = sourceCounts.get(sourceKey) ?? 0;
      if (count >= boundedMaxPerSource) continue;
      sourceCounts.set(sourceKey, count + 1);
      selected.push(result);
      if (selected.length >= boundedLimit) break;
    }
    return selected;
  }

  getDocument(sourcePath: string, release?: string): DocumentChunk[] {
    const rows = (release
      ? this.db.prepare("SELECT * FROM documents WHERE source_path = ? AND release = ? ORDER BY chunk_index").all(sourcePath, release.toLowerCase())
      : this.db.prepare("SELECT * FROM documents WHERE source_path = ? ORDER BY release DESC, chunk_index").all(sourcePath)) as SearchRow[];
    return rows.map(rowToChunk);
  }

  listPublications(release?: string): Array<{ publication: string; docType: string; release: string; documentCount: number }> {
    const rows = (release
      ? this.db.prepare("SELECT publication, doc_type, release, COUNT(DISTINCT source_path) AS document_count FROM documents WHERE chunk_type = 'overview' AND release = ? GROUP BY publication, doc_type, release ORDER BY doc_type, publication").all(release.toLowerCase())
      : this.db.prepare("SELECT publication, doc_type, release, COUNT(DISTINCT source_path) AS document_count FROM documents WHERE chunk_type = 'overview' GROUP BY publication, doc_type, release ORDER BY doc_type, publication, release").all()) as Array<{ publication: string; doc_type: string; release: string; document_count: number }>;
    return rows.map((row) => ({ publication: row.publication, docType: row.doc_type, release: row.release, documentCount: row.document_count }));
  }

  stats(): { documents: number; chunks: number; releases: string[] } {
    const chunks = this.db.prepare("SELECT COUNT(*) FROM documents").pluck().get() as number;
    const documents = this.db.prepare("SELECT COUNT(*) FROM sources").pluck().get() as number;
    const releases = this.db.prepare("SELECT DISTINCT release FROM sources ORDER BY release").pluck().all() as string[];
    return { documents, chunks, releases };
  }

  setManifest(manifest: IndexManifest): void {
    this.db.prepare("INSERT INTO settings(key, value) VALUES ('manifest', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value").run(JSON.stringify(manifest));
  }

  manifest(): IndexManifest | null {
    const value = this.db.prepare("SELECT value FROM settings WHERE key = 'manifest'").pluck().get() as string | undefined;
    return value ? JSON.parse(value) as IndexManifest : null;
  }
}