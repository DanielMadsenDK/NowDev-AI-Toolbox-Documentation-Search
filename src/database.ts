import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { DatabaseSync, type StatementSync } from "node:sqlite";
import * as sqliteVec from "sqlite-vec";
import { analyzeQuery, groupExpression, lexicalTiers, splitIdentifier, type QueryAnalysis } from "./query.js";
import type { DocumentChunk, IndexManifest, SearchFilters, SearchResult } from "./types.js";
import { BinaryVectorIndex } from "./vector-index.js";

interface StoredSource {
  blobSha: string;
  contentHash: string;
}

const DATABASE_SCHEMA_VERSION = 3;
// sqlite-vec rejects KNN queries for more neighbours than this.
const VEC0_MAX_K = 4096;
// Every table, index, and trigger migrate() creates. An index that has all of them opens without writing.
const SCHEMA_OBJECTS = [
  "sources", "documents", "idx_documents_source_chunk", "idx_documents_filters", "idx_documents_source_path", "idx_sources_source_path",
  "document_embeddings", "documents_fts", "documents_ai", "documents_ad", "documents_au", "document_vectors",
];
const RESET_HINT = "Run nowdev-ai-toolbox-documentationsearch reset-index --yes and re-index the documentation.";

export function readStoredEmbeddingProfile(filename: string): string | null {
  return readSetting(filename, "embedding_profile");
}

export function readStoredEmbeddingDtype(filename: string): string | null {
  const stored = readSetting(filename, "embedding_dtype");
  if (stored) return stored;
  // Indexes written before the dtype was stored on its own still record it in their manifest.
  const manifest = readSetting(filename, "manifest");
  if (!manifest) return null;
  try {
    const dtype = (JSON.parse(manifest) as { dtype?: unknown }).dtype;
    return typeof dtype === "string" ? dtype : null;
  } catch {
    return null;
  }
}

export function readStoredDimensions(filename: string): number | null {
  const value = readSetting(filename, "dimensions");
  return value ? Number(value) : null;
}

function readSetting(filename: string, key: string): string | null {
  if (!fs.existsSync(filename)) return null;
  const database = new DatabaseSync(filename, { readOnly: true });
  try {
    const hasSettings = scalar<number>(database.prepare("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'settings'").get());
    if (!hasSettings) return null;
    return scalar<string>(database.prepare("SELECT value FROM settings WHERE key = ?").get(key)) ?? null;
  } finally {
    database.close();
  }
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
  context: string | null;
  fingerprint: string | null;
  metadata?: string;
  source_metadata?: string | null;
  content_hash: string;
  distance?: number;
  rank?: number;
}

function vectorBuffer(vector: Float32Array): Uint8Array {
  return new Uint8Array(vector.buffer, vector.byteOffset, vector.byteLength);
}

function scalar<T>(row: Record<string, unknown> | undefined): T | undefined {
  return row ? Object.values(row)[0] as T : undefined;
}

function scalarRows<T>(rows: Record<string, unknown>[]): T[] {
  return rows.map((row) => Object.values(row)[0] as T);
}

const OMITTED_METADATA_KEYS = new Set(["full_content", "section_content"]);

// Splits chunk metadata into the entries every chunk of a source shares (stored once on the source row) and
// each chunk's own entries (stored on the chunk). Frontmatter repeated on every chunk dominated index size.
function splitMetadata(chunks: DocumentChunk[]): { shared: Record<string, unknown>; own: Array<Record<string, unknown>> } {
  const entries = chunks.map((chunk) => Object.entries(chunk.metadata).filter(([key]) => !OMITTED_METADATA_KEYS.has(key)).map(([key, value]) => [key, JSON.stringify(value)] as const));
  const shared = new Map(entries[0] ?? []);
  for (const chunkEntries of entries.slice(1)) {
    const current = new Map(chunkEntries);
    for (const [key, value] of shared) if (current.get(key) !== value) shared.delete(key);
  }
  return {
    shared: Object.fromEntries([...shared].map(([key, value]) => [key, JSON.parse(value) as unknown])),
    own: entries.map((chunkEntries) => Object.fromEntries(chunkEntries.filter(([key]) => !shared.has(key)).map(([key, value]) => [key, JSON.parse(value) as unknown]))),
  };
}

// Identical passages published under several publications differ only in their breadcrumb and product lines.
function contentFingerprint(content: string): string {
  return createHash("sha1").update(content.replace(/^(?:Path|Product): .*$/gm, "").trim()).digest("hex").slice(0, 16);
}

// Hot queries join with CROSS JOIN, which pins SQLite's join order: the FTS or vector virtual table must drive the
// loop. Planner statistics (from ANALYZE or PRAGMA optimize) could otherwise choose to scan documents and probe the
// virtual table per row, which is catastrophic on a large index and breaks vec0's KNN constraint.
// Every documents column except metadata, which can be tens of kilobytes for API chunks and is only loaded for the final results.
const CANDIDATE_COLUMNS = "d.id, d.doc_type, d.publication, d.source_path, d.release, d.chunk_type, d.chunk_index, d.title, d.heading, d.content, d.topic_type, d.product, d.classification, d.last_updated, d.object_name, d.method_name, d.context, d.fingerprint, d.content_hash";
const TERM_FREQUENCY_CACHE_LIMIT = 2048;

function identifierBoost(row: SearchRow, terms: Set<string>): number {
  const matches = (identifier: string | null) => {
    if (!identifier) return false;
    const identifierWords = splitIdentifier(identifier);
    return identifierWords.length > 0 && (terms.has(identifier.toLowerCase()) || identifierWords.every((term) => terms.has(term)));
  };
  const objectMatch = matches(row.object_name);
  const methodMatch = matches(row.method_name);
  if (objectMatch && methodMatch) return 1.8;
  if (methodMatch) return 1.6;
  if (objectMatch) return 1.15;
  return 1;
}

function chunkTypeMultiplier(row: SearchRow, query: string, analysis: QueryAnalysis): number {
  const base: Partial<Record<DocumentChunk["chunkType"], number>> = { overview: 1, section: 0.95, definition: 0.85, method: 0.8, endpoint: 0.8, parameter: 0.5, returns: 0.5, example: 0.45 };
  const multiplier = base[row.chunk_type] ?? 0.9;
  if (analysis.definitionLike && row.chunk_type === "definition") return multiplier * 1.4;
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
    context: row.context,
    metadata: { ...JSON.parse(row.source_metadata ?? "{}") as Record<string, unknown>, ...JSON.parse(row.metadata ?? "{}") as Record<string, unknown> },
    contentHash: row.content_hash,
  };
}

function filterValues(filters: SearchFilters): Array<string | null> {
  return [filters.release?.toLowerCase() ?? null, filters.docType ?? null, filters.publication ?? null, filters.chunkType ?? null, filters.topicType ?? null]
    .flatMap((value) => [value, value]);
}

export class DocumentationSearchDatabase {
  readonly db: DatabaseSync;
  private readonly semanticStatements = new Map<number, StatementSync>();
  private readonly statements = new Map<string, StatementSync>();
  private readonly termFrequencies = new Map<string, number>();
  private releaseCache?: string[];
  private vectorIndex?: BinaryVectorIndex;
  private vectorIndexLoading?: Promise<void>;
  private writeGeneration = 0;
  private readonly usesBinaryCoarseVectors: boolean;
  private readonly coarseVectorOversample: number;

  constructor(readonly filename: string, readonly dimensions: number, coarseVectorOversample = 4) {
    if (!Number.isInteger(coarseVectorOversample) || coarseVectorOversample < 1 || coarseVectorOversample > 64) {
      throw new Error("coarseVectorOversample must be an integer between 1 and 64");
    }
    this.usesBinaryCoarseVectors = dimensions % 8 === 0;
    this.coarseVectorOversample = coarseVectorOversample;
    fs.mkdirSync(path.dirname(filename), { recursive: true });
    this.db = new DatabaseSync(filename, { allowExtension: true, timeout: 5_000 });
    try {
      this.db.exec(`
        PRAGMA journal_mode = WAL;
        PRAGMA foreign_keys = ON;
        PRAGMA synchronous = NORMAL;
        PRAGMA temp_store = MEMORY;
        PRAGMA cache_size = -64000;
        PRAGMA mmap_size = 268435456;
      `);
      sqliteVec.load(this.db);
      // Only sqlite-vec is needed; leaving extension loading on would let any SQL run against this handle load one.
      this.db.enableLoadExtension(false);
      this.migrate();
    } catch (error) {
      this.db.close();
      throw error;
    }
  }

  private statement(sql: string): StatementSync {
    let statement = this.statements.get(sql);
    if (!statement) {
      statement = this.db.prepare(sql);
      this.statements.set(sql, statement);
    }
    return statement;
  }

  private migrate(): void {
    this.db.exec("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)");
    const storedSchemaVersion = scalar<string>(this.db.prepare("SELECT value FROM settings WHERE key = 'database_schema_version'").get());
    const hasDocuments = scalar<number>(this.db.prepare("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'documents'").get());
    if (storedSchemaVersion ? Number(storedSchemaVersion) !== DATABASE_SCHEMA_VERSION : hasDocuments) {
      throw new Error(`Index uses database schema ${storedSchemaVersion ?? "1"}, but this version requires ${DATABASE_SCHEMA_VERSION}. ${RESET_HINT}`);
    }
    const storedDimensions = scalar<string>(this.db.prepare("SELECT value FROM settings WHERE key = 'dimensions'").get());
    if (storedDimensions && Number(storedDimensions) !== this.dimensions) {
      throw new Error(`Index uses ${storedDimensions}-dimensional embeddings, but provider uses ${this.dimensions}. Remove the index or use a matching model.`);
    }
    // vec0 holds only what nearest-neighbour candidate selection reads: binary codes (or full vectors when the
    // dimension count cannot be bit-packed) plus the filter columns. Full-precision vectors for re-scoring and for
    // keyword-candidate similarity live in a plain rowid table, where a point lookup costs microseconds.
    const vectorColumn = this.usesBinaryCoarseVectors ? `coarse_embedding bit[${this.dimensions}]` : `embedding float[${this.dimensions}] distance_metric=cosine`;
    const existing = new Set(scalarRows<string>(this.db.prepare("SELECT name FROM sqlite_master").all()));
    // A complete index opens without writing, so a search process can start while an indexer holds the write lock.
    if (storedSchemaVersion && SCHEMA_OBJECTS.every((name) => existing.has(name))) {
      this.checkVectorLayout();
      return;
    }
    this.transaction(() => this.createSchema(vectorColumn));
    this.checkVectorLayout();
  }

  private createSchema(vectorColumn: string): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS sources (
        release TEXT NOT NULL,
        source_path TEXT NOT NULL,
        blob_sha TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        content TEXT NOT NULL,
        metadata TEXT NOT NULL DEFAULT '{}',
        PRIMARY KEY (release, source_path)
      );
      CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY,
        doc_type TEXT NOT NULL, publication TEXT NOT NULL, source_path TEXT NOT NULL,
        release TEXT NOT NULL, chunk_type TEXT NOT NULL, chunk_index INTEGER NOT NULL,
        title TEXT NOT NULL, heading TEXT, content TEXT NOT NULL, topic_type TEXT,
        product TEXT, classification TEXT, last_updated TEXT, object_name TEXT,
        method_name TEXT, context TEXT, fingerprint TEXT, metadata TEXT NOT NULL, content_hash TEXT NOT NULL
      );
      CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_source_chunk ON documents(release, source_path, chunk_index);
      CREATE INDEX IF NOT EXISTS idx_documents_filters ON documents(release, doc_type, publication, chunk_type, topic_type);
      -- Document lookups without a release filter by path alone, which the release-first keys above cannot serve.
      CREATE INDEX IF NOT EXISTS idx_documents_source_path ON documents(source_path, release, chunk_index);
      CREATE INDEX IF NOT EXISTS idx_sources_source_path ON sources(source_path, release);
      CREATE TABLE IF NOT EXISTS document_embeddings (id INTEGER PRIMARY KEY, embedding BLOB NOT NULL);
      CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
        title, heading, object_name, method_name, context, content,
        content='documents', content_rowid='id',
        tokenize='porter unicode61 remove_diacritics 2'
      );
      CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
        INSERT INTO documents_fts(rowid, title, heading, object_name, method_name, context, content)
        VALUES (new.id, new.title, new.heading, new.object_name, new.method_name, new.context, new.content);
      END;
      CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
        INSERT INTO documents_fts(documents_fts, rowid, title, heading, object_name, method_name, context, content)
        VALUES ('delete', old.id, old.title, old.heading, old.object_name, old.method_name, old.context, old.content);
      END;
      CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
        INSERT INTO documents_fts(documents_fts, rowid, title, heading, object_name, method_name, context, content)
        VALUES ('delete', old.id, old.title, old.heading, old.object_name, old.method_name, old.context, old.content);
        INSERT INTO documents_fts(rowid, title, heading, object_name, method_name, context, content)
        VALUES (new.id, new.title, new.heading, new.object_name, new.method_name, new.context, new.content);
      END;
      CREATE VIRTUAL TABLE IF NOT EXISTS document_vectors USING vec0(
        ${vectorColumn},
        release text partition key,
        doc_type text,
        publication text,
        chunk_type text,
        topic_type text
      );
    `);
    // FTS5 stores the rank configuration persistently, so it only needs writing when the schema is created.
    this.db.prepare("INSERT INTO documents_fts(documents_fts, rank) VALUES ('rank', 'bm25(10.0, 7.0, 9.0, 9.0, 4.0, 1.0)')").run();
    this.db.prepare("INSERT OR IGNORE INTO settings(key, value) VALUES ('dimensions', ?)").run(String(this.dimensions));
    this.db.prepare("INSERT OR IGNORE INTO settings(key, value) VALUES ('database_schema_version', ?)").run(String(DATABASE_SCHEMA_VERSION));
  }

  private checkVectorLayout(): void {
    const vectorSchema = scalar<string>(this.db.prepare("SELECT sql FROM sqlite_master WHERE name = 'document_vectors'").get());
    if (!vectorSchema?.includes(this.usesBinaryCoarseVectors ? "coarse_embedding bit" : "embedding float")) {
      throw new Error(`Index vector layout does not match ${this.dimensions}-dimensional embeddings. ${RESET_HINT}`);
    }
  }

  close(): void {
    try {
      // Lets SQLite refresh planner statistics for tables whose contents changed during this connection. Best-effort,
      // so it gives up at once instead of waiting out the busy timeout while another process holds the write lock.
      this.db.exec("PRAGMA busy_timeout = 0");
      this.db.exec("PRAGMA optimize");
    } catch {
      // A read-only or busy database cannot be optimized; closing must still succeed.
    }
    this.db.close();
  }

  private transaction<T>(action: () => T): T {
    this.db.exec("BEGIN IMMEDIATE");
    try {
      const result = action();
      this.db.exec("COMMIT");
      return result;
    } catch (error) {
      if (this.db.isTransaction) this.db.exec("ROLLBACK");
      throw error;
    }
  }

  private setting(key: string): string | null {
    return scalar<string>(this.statement("SELECT value FROM settings WHERE key = ?").get(key)) ?? null;
  }

  /** Writes only when the value changes, so reopening an index with the same settings takes no write lock. */
  private setSetting(key: string, value: string): void {
    if (this.setting(key) === value) return;
    this.statement("INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value").run(key, value);
  }

  embeddingProfile(): string | null {
    return this.setting("embedding_profile");
  }

  setEmbeddingProfile(profile: string): void {
    this.setSetting("embedding_profile", profile);
  }

  embeddingDtype(): string | null {
    return this.setting("embedding_dtype");
  }

  setEmbeddingDtype(dtype: string): void {
    this.setSetting("embedding_dtype", dtype);
  }

  sourceMap(release: string): Map<string, StoredSource> {
    const rows = this.statement("SELECT source_path, blob_sha, content_hash FROM sources WHERE release = ?").all(release) as Array<{ source_path: string; blob_sha: string; content_hash: string }>;
    return new Map(rows.map((row) => [row.source_path, { blobSha: row.blob_sha, contentHash: row.content_hash }]));
  }

  replaceSources(release: string, sources: Array<{ path: string; blobSha: string; contentHash: string; content?: string; chunks: DocumentChunk[]; embeddings: Float32Array[] }>, deletedPaths: string[], manifest?: IndexManifest): void {
    const remove = (sourcePaths: string[]) => {
      if (!sourcePaths.length) return;
      this.db.exec("CREATE TEMP TABLE IF NOT EXISTS affected_sources(source_path TEXT PRIMARY KEY); DELETE FROM affected_sources;");
      const insertAffected = this.statement("INSERT OR IGNORE INTO affected_sources(source_path) VALUES (?)");
      sourcePaths.forEach((sourcePath) => insertAffected.run(sourcePath));
      this.db.exec("CREATE TEMP TABLE IF NOT EXISTS affected_document_ids(id INTEGER PRIMARY KEY); DELETE FROM affected_document_ids;");
      this.statement("INSERT INTO affected_document_ids SELECT id FROM documents WHERE release = ? AND source_path IN (SELECT source_path FROM affected_sources)").run(release);
      this.db.exec("DELETE FROM document_vectors WHERE rowid IN (SELECT id FROM affected_document_ids)");
      this.db.exec("DELETE FROM document_embeddings WHERE id IN (SELECT id FROM affected_document_ids)");
      this.statement("DELETE FROM documents WHERE release = ? AND source_path IN (SELECT source_path FROM affected_sources)").run(release);
      this.statement("DELETE FROM sources WHERE release = ? AND source_path IN (SELECT source_path FROM affected_sources)").run(release);
    };
    this.invalidateReadCaches();
    this.transaction(() => {
      remove([...deletedPaths, ...sources.map((source) => source.path)]);
      const insertDocument = this.statement(`INSERT INTO documents (
        doc_type, publication, source_path, release, chunk_type, chunk_index, title, heading,
        content, topic_type, product, classification, last_updated, object_name, method_name, context, fingerprint, metadata, content_hash
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`);
      const insertVector = this.usesBinaryCoarseVectors
        ? this.statement("INSERT INTO document_vectors(rowid, coarse_embedding, release, doc_type, publication, chunk_type, topic_type) VALUES (?, vec_quantize_binary(?), ?, ?, ?, ?, ?)")
        : this.statement("INSERT INTO document_vectors(rowid, embedding, release, doc_type, publication, chunk_type, topic_type) VALUES (?, ?, ?, ?, ?, ?, ?)");
      const insertEmbedding = this.statement("INSERT INTO document_embeddings(id, embedding) VALUES (?, ?)");
      const insertSource = this.statement("INSERT INTO sources(release, source_path, blob_sha, content_hash, content, metadata) VALUES (?, ?, ?, ?, ?, ?)");
      for (const source of sources) {
        if (source.chunks.length !== source.embeddings.length) throw new Error(`Embedding count mismatch for ${source.path}`);
        if (source.chunks.some((chunk) => chunk.release !== release || chunk.sourcePath !== source.path)) {
          throw new Error(`Chunk source identity mismatch for ${source.path}`);
        }
        const metadata = splitMetadata(source.chunks);
        source.chunks.forEach((chunk, index) => {
          const result = insertDocument.run(chunk.docType, chunk.publication, chunk.sourcePath, chunk.release, chunk.chunkType, chunk.chunkIndex, chunk.title, chunk.heading, chunk.content, chunk.topicType, chunk.product, chunk.classification, chunk.lastUpdated, chunk.objectName, chunk.methodName, chunk.context ?? null, contentFingerprint(chunk.content), JSON.stringify(metadata.own[index]), chunk.contentHash);
          const id = BigInt(result.lastInsertRowid);
          const vector = vectorBuffer(source.embeddings[index]!);
          insertVector.run(id, vector, chunk.release, chunk.docType, chunk.publication, chunk.chunkType, chunk.topicType ?? "");
          insertEmbedding.run(id, vector);
        });
        const fallbackContent = source.chunks.find((chunk) => typeof chunk.metadata.full_content === "string")?.metadata.full_content ?? source.chunks.map((chunk) => chunk.content).join("\n\n");
        insertSource.run(release, source.path, source.blobSha, source.contentHash, source.content ?? String(fallbackContent), JSON.stringify(metadata.shared));
      }
      if (manifest) this.setManifest(manifest);
    });
  }

  private invalidateReadCaches(): void {
    this.writeGeneration += 1;
    this.releaseCache = undefined;
    this.termFrequencies.clear();
    this.vectorIndex = undefined;
    // A load in flight now reads outdated rows; it stops at its next yield, and the next warmUp() starts afresh.
    this.vectorIndexLoading = undefined;
  }

  /**
   * Loads every binary vector code into memory so semantic candidate selection is a ~10 ms scan instead of a vec0
   * KNN query whose cost grows with the oversampled candidate count. Intended for long-running processes; the load
   * yields to the event loop periodically. Writes drop the in-memory copy, and searches fall back to vec0 until the
   * next warm-up.
   */
  warmUp(): Promise<void> {
    if (!this.usesBinaryCoarseVectors || this.vectorIndex) return Promise.resolve();
    if (!this.vectorIndexLoading) {
      const generation = this.writeGeneration;
      const isStale = () => generation !== this.writeGeneration;
      const loading: Promise<void> = BinaryVectorIndex.load(this.db, this.dimensions, isStale)
        .then((index) => {
          if (!isStale()) this.vectorIndex = index;
        }, (error: unknown) => {
          // A load overtaken by a write is not a failure; searches use vec0 until the next warm-up.
          if (!isStale()) throw error;
        })
        .finally(() => {
          if (this.vectorIndexLoading === loading) this.vectorIndexLoading = undefined;
        });
      this.vectorIndexLoading = loading;
    }
    return this.vectorIndexLoading;
  }

  get vectorIndexLoaded(): boolean {
    return this.vectorIndex !== undefined;
  }

  private semanticSearch(embedding: Float32Array, candidates: number, release: string, filters: SearchFilters): SearchRow[] {
    const vector = vectorBuffer(embedding);
    if (this.vectorIndex) {
      const ids = this.vectorIndex.nearest(embedding, candidates * this.coarseVectorOversample, { release, docType: filters.docType, publication: filters.publication, chunkType: filters.chunkType, topicType: filters.topicType });
      if (!ids.length) return [];
      return this.statement(`SELECT ${CANDIDATE_COLUMNS}, vec_distance_cosine(e.embedding, ?) AS distance
        FROM json_each(?) j CROSS JOIN document_embeddings e ON e.id = j.value CROSS JOIN documents d ON d.id = j.value
        ORDER BY distance, d.id LIMIT ?`).all(vector, JSON.stringify(ids), candidates) as unknown as SearchRow[];
    }
    const values = [filters.docType, filters.publication, filters.chunkType, filters.topicType];
    const mask = values.reduce((result, value, index) => result | (value === undefined ? 0 : 1 << index), 0);
    const parameters = values.filter((value): value is string => value !== undefined);
    const predicates = ["v.doc_type = ?", "v.publication = ?", "v.chunk_type = ?", "v.topic_type = ?"]
      .filter((_, index) => mask & (1 << index));
    const where = predicates.length ? ` AND ${predicates.join(" AND ")}` : "";
    let statement = this.semanticStatements.get(mask);
    if (!statement) {
      statement = this.usesBinaryCoarseVectors
        ? this.db.prepare(`WITH coarse_matches AS (
            SELECT v.rowid FROM document_vectors v
            WHERE v.coarse_embedding MATCH vec_quantize_binary(?) AND k = ? AND v.release = ?${where}
          )
          SELECT ${CANDIDATE_COLUMNS}, vec_distance_cosine(e.embedding, ?) AS distance
          FROM coarse_matches c CROSS JOIN document_embeddings e ON e.id = c.rowid CROSS JOIN documents d ON d.id = c.rowid
          ORDER BY distance, d.id LIMIT ?`)
        : this.db.prepare(`SELECT ${CANDIDATE_COLUMNS}, v.distance FROM document_vectors v CROSS JOIN documents d ON d.id = v.rowid
          WHERE v.embedding MATCH ? AND k = ? AND v.release = ?${where}
          ORDER BY v.distance, d.id LIMIT ?`);
      this.semanticStatements.set(mask, statement);
    }
    return this.usesBinaryCoarseVectors
      ? statement.all(vector, Math.min(candidates * this.coarseVectorOversample, VEC0_MAX_K), release, ...parameters, vector, candidates) as unknown as SearchRow[]
      : statement.all(vector, Math.min(candidates, VEC0_MAX_K), release, ...parameters, candidates) as unknown as SearchRow[];
  }

  private releases(): string[] {
    this.releaseCache ??= scalarRows<string>(this.statement("SELECT DISTINCT release FROM sources ORDER BY release").all());
    return [...this.releaseCache];
  }

  private termFrequency(expression: string): number {
    let frequency = this.termFrequencies.get(expression);
    if (frequency === undefined) {
      frequency = scalar<number>(this.statement("SELECT count(*) FROM documents_fts WHERE documents_fts MATCH ?").get(expression)) ?? 0;
      if (this.termFrequencies.size >= TERM_FREQUENCY_CACHE_LIMIT) this.termFrequencies.clear();
      this.termFrequencies.set(expression, frequency);
    }
    return frequency;
  }

  // Runs progressively relaxed conjunctions (see lexicalTiers) until enough candidates are collected. Rows from stricter
  // tiers keep their position ahead of rows that only match a relaxed tier, so matching more query terms ranks higher.
  private keywordSearch(analysis: QueryAnalysis, embedding: Float32Array, filters: SearchFilters, candidates: number): SearchRow[] {
    if (!analysis.groups.length) return [];
    const statement = this.statement(`
      SELECT ${CANDIDATE_COLUMNS}, rank, vec_distance_cosine(e.embedding, ?) AS distance
      FROM documents_fts f
      CROSS JOIN documents d ON d.id = f.rowid
      CROSS JOIN document_embeddings e ON e.id = d.id
      WHERE documents_fts MATCH ?
        AND (? IS NULL OR d.release = ?)
        AND (? IS NULL OR d.doc_type = ?)
        AND (? IS NULL OR d.publication = ?)
        AND (? IS NULL OR d.chunk_type = ?)
        AND (? IS NULL OR d.topic_type = ?)
      ORDER BY rank LIMIT ?
    `);
    const frequencies = analysis.groups.map((group) => this.termFrequency(groupExpression(group)));
    const filtersParameters = filterValues(filters);
    const rows = new Map<number, SearchRow>();
    for (const expression of lexicalTiers(analysis, frequencies)) {
      const tierRows = statement.all(vectorBuffer(embedding), expression, ...filtersParameters, candidates) as unknown as SearchRow[];
      for (const row of tierRows) if (!rows.has(row.id)) rows.set(row.id, row);
      if (rows.size >= candidates) break;
    }
    return [...rows.values()].slice(0, candidates);
  }

  /** Loads the metadata candidate rows leave out, for the final results only, in one query. */
  private toResults(selected: Array<{ row: SearchRow; score: number; similarity: number }>): SearchResult[] {
    if (!selected.length) return [];
    const stored = this.statement(`SELECT d.id, d.metadata, s.metadata AS source_metadata
      FROM json_each(?) j CROSS JOIN documents d ON d.id = j.value LEFT JOIN sources s ON s.release = d.release AND s.source_path = d.source_path`)
      .all(JSON.stringify(selected.map(({ row }) => row.id))) as Array<{ id: number; metadata: string; source_metadata: string | null }>;
    const metadata = new Map(stored.map((item) => [item.id, item]));
    return selected.map(({ row, score, similarity }) => {
      const item = metadata.get(row.id);
      const chunk = rowToChunk({ ...row, metadata: item?.metadata ?? "{}", source_metadata: item?.source_metadata ?? null });
      return { ...chunk, content: row.content.slice(0, 1000), score, similarity };
    });
  }

  search(query: string, embedding: Float32Array, filters: SearchFilters, limit = 10, threshold = 0.3, deduplicateReleases = false, maxResultsPerSource = 3): SearchResult[] {
    const boundedLimit = Math.min(Math.max(limit, 1), 50);
    const boundedMaxPerSource = Math.min(Math.max(Math.floor(maxResultsPerSource), 1), 10);
    const candidates = Math.max(100, boundedLimit * 20);
    const semanticCandidates = Math.max(100, boundedLimit * 10);
    const releases = filters.release ? [filters.release.toLowerCase()] : this.releases();
    const candidatesPerRelease = filters.release ? semanticCandidates : Math.max(1, Math.ceil(semanticCandidates / Math.max(releases.length, 1)));
    const semantic = releases
      .flatMap((release) => this.semanticSearch(embedding, candidatesPerRelease, release, filters))
      .sort((left, right) => (left.distance ?? 1) - (right.distance ?? 1) || left.id - right.id)
      .slice(0, semanticCandidates);
    const analysis = analyzeQuery(query);
    const keyword = this.keywordSearch(analysis, embedding, filters, candidates);
    const keywordWeight = analysis.apiLike ? 1.25 : 1;
    const scores = new Map<number, { row: SearchRow; score: number; similarity: number; keywordRank: number }>();
    semantic.forEach((row, index) => {
      const similarity = 1 - (row.distance ?? 1);
      if (similarity < threshold) return;
      scores.set(row.id, { row, score: 1 / (60 + index + 1), similarity, keywordRank: Number.POSITIVE_INFINITY });
    });
    keyword.forEach((row, index) => {
      const similarity = 1 - (row.distance ?? 1);
      const current = scores.get(row.id);
      scores.set(row.id, { row, score: (current?.score ?? 0) + keywordWeight / (60 + index + 1), similarity: current?.similarity ?? similarity, keywordRank: index });
    });
    // Exact fused-score ties go to the better keyword rank: lexical evidence is the more specific signal.
    const ranked = [...scores.values()]
      .map(({ row, score, similarity, keywordRank }) => ({ row, similarity, keywordRank, score: score * chunkTypeMultiplier(row, query, analysis) * identifierBoost(row, analysis.words) }))
      .sort((left, right) => right.score - left.score || left.keywordRank - right.keywordRank || left.row.id - right.row.id);
    const sourceCounts = new Map<string, number>();
    const deduplicationKeys = new Set<string>();
    const fingerprints = new Set<string>();
    const selected: Array<{ row: SearchRow; score: number; similarity: number }> = [];
    for (const { row, score, similarity } of ranked) {
      if (deduplicateReleases) {
        const deduplicationKey = `${row.source_path}\u0000${row.chunk_index}`;
        if (deduplicationKeys.has(deduplicationKey)) continue;
        deduplicationKeys.add(deduplicationKey);
      }
      // The same passage republished under another publication adds nothing to the result list.
      if (row.fingerprint) {
        const fingerprintKey = deduplicateReleases ? row.fingerprint : `${row.release}\u0000${row.fingerprint}`;
        if (fingerprints.has(fingerprintKey)) continue;
        fingerprints.add(fingerprintKey);
      }
      const sourceKey = deduplicateReleases ? row.source_path : `${row.release}\u0000${row.source_path}`;
      const count = sourceCounts.get(sourceKey) ?? 0;
      if (count >= boundedMaxPerSource) continue;
      sourceCounts.set(sourceKey, count + 1);
      selected.push({ row, score, similarity });
      if (selected.length >= boundedLimit) break;
    }
    return this.toResults(selected);
  }

  getDocument(sourcePath: string, release?: string): DocumentChunk[] {
    const columns = "d.*, s.metadata AS source_metadata FROM documents d LEFT JOIN sources s ON s.release = d.release AND s.source_path = d.source_path";
    const rows = (release
      ? this.statement(`SELECT ${columns} WHERE d.source_path = ? AND d.release = ? ORDER BY d.chunk_index`).all(sourcePath, release.toLowerCase())
      : this.statement(`SELECT ${columns} WHERE d.source_path = ? ORDER BY d.release DESC, d.chunk_index`).all(sourcePath)) as unknown as SearchRow[];
    return rows.map(rowToChunk);
  }

  getSourceContent(sourcePath: string, release?: string): string | null {
    const row = release
      ? this.statement("SELECT content FROM sources WHERE source_path = ? AND release = ?").get(sourcePath, release.toLowerCase())
      : this.statement("SELECT content FROM sources WHERE source_path = ? ORDER BY release DESC LIMIT 1").get(sourcePath);
    return scalar<string>(row) ?? null;
  }

  listPublications(release?: string): Array<{ publication: string; docType: string; release: string; documentCount: number }> {
    const rows = (release
      ? this.statement("SELECT publication, doc_type, release, COUNT(DISTINCT source_path) AS document_count FROM documents WHERE chunk_type = 'overview' AND release = ? GROUP BY publication, doc_type, release ORDER BY doc_type, publication").all(release.toLowerCase())
      : this.statement("SELECT publication, doc_type, release, COUNT(DISTINCT source_path) AS document_count FROM documents WHERE chunk_type = 'overview' GROUP BY publication, doc_type, release ORDER BY doc_type, publication, release").all()) as Array<{ publication: string; doc_type: string; release: string; document_count: number }>;
    return rows.map((row) => ({ publication: row.publication, docType: row.doc_type, release: row.release, documentCount: row.document_count }));
  }

  /** Merges FTS segments and truncates the WAL after a batch of writes. */
  optimizeSearchIndex(): void {
    this.db.prepare("INSERT INTO documents_fts(documents_fts) VALUES ('optimize')").run();
    this.db.exec("PRAGMA wal_checkpoint(TRUNCATE)");
  }

  stats(): { documents: number; chunks: number; releases: string[] } {
    const chunks = scalar<number>(this.statement("SELECT COUNT(*) FROM documents").get())!;
    const documents = scalar<number>(this.statement("SELECT COUNT(*) FROM sources").get())!;
    return { documents, chunks, releases: this.releases() };
  }

  setManifest(manifest: IndexManifest): void {
    this.setSetting("manifest", JSON.stringify(manifest));
  }

  manifest(): IndexManifest | null {
    const value = this.setting("manifest");
    return value ? JSON.parse(value) as IndexManifest : null;
  }
}
