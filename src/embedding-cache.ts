import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { DatabaseSync, type StatementSync } from "node:sqlite";
import { pathToFileURL } from "node:url";
import type { IndexManifest } from "./types.js";

const CACHE_SCHEMA_VERSION = 1;
const HARVEST_BATCH_SIZE = 5_000;
const HARVEST_PROGRESS_INTERVAL = 100_000;

/** The manifest fields that determine a document vector for a given passage text. */
export type EmbeddingIdentity = Pick<IndexManifest, "embeddingProvider" | "embeddingModel" | "documentEmbeddingModel" | "dimensions" | "pooling" | "dtype" | "layerNorm" | "documentPrefix" | "maxEmbeddingCharacters">;

/**
 * Canonical cache namespace for an embedding configuration. Derived from the same manifest fields that decide
 * whether an index can be reopened, so a cached vector is only reused where the index itself would accept it.
 */
export function embeddingFingerprint(identity: EmbeddingIdentity): string {
  const maxCharacters = identity.maxEmbeddingCharacters;
  return JSON.stringify({
    provider: identity.embeddingProvider,
    model: identity.embeddingModel,
    documentModel: identity.documentEmbeddingModel ?? null,
    dimensions: identity.dimensions,
    pooling: identity.pooling,
    dtype: identity.dtype ?? null,
    layerNorm: identity.layerNorm ?? false,
    documentPrefix: identity.documentPrefix ?? "",
    // A manifest round-trips an unlimited (Infinity) cap as null through JSON.
    maxEmbeddingCharacters: typeof maxCharacters === "number" && Number.isFinite(maxCharacters) ? maxCharacters : null,
  });
}

/** Cache key for a passage: the text passed to the provider, before its prefix and cap, which the namespace fixes. */
export function embeddingTextHash(text: string): Uint8Array {
  return createHash("sha1").update(text).digest();
}

export function embeddingCacheSize(filename: string): number {
  return [filename, `${filename}-wal`].reduce((total, file) => {
    try {
      return total + fs.statSync(file).size;
    } catch {
      return total;
    }
  }, 0);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Document vectors keyed by passage text, stored apart from the index so they survive reset-index, --refresh, and
 * indexing another family. Purely an optimization: every failure disables the cache for the rest of the run
 * (reported through onError) instead of failing indexing.
 */
export class EmbeddingCache {
  private readonly select: StatementSync;
  private readonly insert: StatementSync;
  private disabled = false;

  private constructor(
    private readonly db: DatabaseSync,
    private readonly namespace: number,
    private readonly dimensions: number,
    private readonly onError: (message: string) => void,
  ) {
    this.select = db.prepare("SELECT vector FROM embeddings WHERE namespace = ? AND text_hash = ?");
    this.insert = db.prepare("INSERT OR IGNORE INTO embeddings(namespace, text_hash, vector) VALUES (?, ?, ?)");
  }

  /** Opens the cache for one embedding configuration, or reports why it cannot be used and returns undefined. */
  static open(filename: string, identity: EmbeddingIdentity, onError: (message: string) => void): EmbeddingCache | undefined {
    let db: DatabaseSync | undefined;
    try {
      fs.mkdirSync(path.dirname(filename), { recursive: true });
      db = new DatabaseSync(filename, { timeout: 5_000 });
      db.exec(`
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS namespaces (id INTEGER PRIMARY KEY, fingerprint TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS embeddings (
          namespace INTEGER NOT NULL,
          text_hash BLOB NOT NULL,
          vector BLOB NOT NULL,
          PRIMARY KEY (namespace, text_hash)
        ) WITHOUT ROWID;
      `);
      const version = (db.prepare("SELECT value FROM settings WHERE key = 'schema_version'").get() as { value: string } | undefined)?.value;
      if (version === undefined) db.prepare("INSERT INTO settings(key, value) VALUES ('schema_version', ?)").run(String(CACHE_SCHEMA_VERSION));
      else if (Number(version) !== CACHE_SCHEMA_VERSION) {
        throw new Error(`cache schema ${version} is not supported (expected ${CACHE_SCHEMA_VERSION}); run reset-index --yes --clear-embedding-cache`);
      }
      const fingerprint = embeddingFingerprint(identity);
      db.prepare("INSERT OR IGNORE INTO namespaces(fingerprint, created_at) VALUES (?, ?)").run(fingerprint, new Date().toISOString());
      const namespace = Number((db.prepare("SELECT id FROM namespaces WHERE fingerprint = ?").get(fingerprint) as { id: number | bigint }).id);
      return new EmbeddingCache(db, namespace, identity.dimensions, onError);
    } catch (error) {
      try {
        db?.close();
      } catch {
        // Closing a half-opened database cannot make the reported failure any clearer.
      }
      onError(`Embedding cache unavailable (${errorMessage(error)}); embedding without it.`);
      return undefined;
    }
  }

  get enabled(): boolean {
    return !this.disabled;
  }

  /** Returns the cached vector for each hash, or undefined where there is none. */
  lookup(hashes: Uint8Array[]): Array<Float32Array | undefined> {
    if (this.disabled) return [];
    try {
      return hashes.map((hash) => {
        const vector = (this.select.get(this.namespace, hash) as { vector: Uint8Array } | undefined)?.vector;
        if (!vector || vector.byteLength !== this.dimensions * 4) return undefined;
        // Copies into a fresh, 4-byte-aligned buffer.
        return new Float32Array(vector.buffer.slice(vector.byteOffset, vector.byteOffset + vector.byteLength));
      });
    } catch (error) {
      this.disable(error);
      return [];
    }
  }

  store(entries: Array<{ hash: Uint8Array; vector: Float32Array }>): void {
    this.storeBytes(entries.map(({ hash, vector }) => ({ hash, bytes: new Uint8Array(vector.buffer, vector.byteOffset, vector.byteLength) })));
  }

  /** Returns the number of entries that were new to the cache. */
  storeBytes(entries: Array<{ hash: Uint8Array; bytes: Uint8Array }>): number {
    if (this.disabled || !entries.length) return 0;
    try {
      let inserted = 0;
      this.db.exec("BEGIN IMMEDIATE");
      try {
        for (const { hash, bytes } of entries) {
          if (bytes.byteLength === this.dimensions * 4) inserted += Number(this.insert.run(this.namespace, hash, bytes).changes);
        }
        this.db.exec("COMMIT");
      } catch (error) {
        if (this.db.isTransaction) this.db.exec("ROLLBACK");
        throw error;
      }
      return inserted;
    } catch (error) {
      this.disable(error);
      return 0;
    }
  }

  close(): void {
    try {
      this.db.close();
    } catch {
      // Nothing is pending outside a transaction, so a failed close loses no cached vectors.
    }
  }

  private disable(error: unknown): void {
    this.disabled = true;
    this.onError(`Embedding cache disabled for the rest of this run (${errorMessage(error)}).`);
  }
}

function openIndexReadOnly(filename: string): DatabaseSync {
  try {
    const database = new DatabaseSync(filename, { readOnly: true });
    try {
      database.prepare("SELECT 1 FROM sqlite_master LIMIT 1").get();
      return database;
    } catch (error) {
      database.close();
      throw error;
    }
  } catch {
    // A WAL database whose directory is not writable cannot create its shared-memory file; read the main file as-is.
    const url = pathToFileURL(filename);
    url.searchParams.set("immutable", "1");
    return new DatabaseSync(url, { readOnly: true });
  }
}

export type HarvestResult = { harvested: number; vectors: number } | { skipped: string };

/**
 * Copies every (passage, vector) pair of an index into the cache under the index's own embedding configuration, so
 * rebuilding after reset-index only embeds passages whose text changed. Indexes older than database schema 3 keep
 * no plain full-precision vector table and are skipped.
 */
export function harvestIndexEmbeddings(indexFilename: string, cacheFilename: string, onProgress: (message: string) => void = () => undefined): HarvestResult {
  if (!fs.existsSync(indexFilename)) return { skipped: "there is no index" };
  let index: DatabaseSync;
  try {
    index = openIndexReadOnly(indexFilename);
  } catch (error) {
    return { skipped: `the index could not be read (${errorMessage(error)})` };
  }
  let cache: EmbeddingCache | undefined;
  let cacheError: string | undefined;
  try {
    const hasTable = (name: string) => index.prepare("SELECT 1 AS found FROM sqlite_master WHERE type = 'table' AND name = ?").get(name) !== undefined;
    if (!["settings", "documents", "document_embeddings"].every(hasTable)) return { skipped: "the index predates database schema 3 and keeps no reusable vectors" };
    const manifestValue = (index.prepare("SELECT value FROM settings WHERE key = 'manifest'").get() as { value: string } | undefined)?.value;
    if (!manifestValue) return { skipped: "the index has no manifest" };
    const manifest = JSON.parse(manifestValue) as IndexManifest;
    cache = EmbeddingCache.open(cacheFilename, manifest, (message) => { cacheError ??= message; });
    if (!cache) return { skipped: cacheError ?? "the embedding cache could not be opened" };
    let vectors = 0;
    let harvested = 0;
    let batch: Array<{ hash: Uint8Array; bytes: Uint8Array }> = [];
    const flush = () => {
      harvested += cache!.storeBytes(batch);
      batch = [];
    };
    const rows = index.prepare("SELECT d.content AS content, e.embedding AS embedding FROM documents d CROSS JOIN document_embeddings e ON e.id = d.id").iterate() as Iterable<{ content: string; embedding: Uint8Array }>;
    for (const row of rows) {
      batch.push({ hash: embeddingTextHash(row.content), bytes: row.embedding });
      vectors += 1;
      if (batch.length >= HARVEST_BATCH_SIZE) flush();
      if (vectors % HARVEST_PROGRESS_INTERVAL === 0) onProgress(`Saved ${vectors} indexed embeddings to the embedding cache...`);
      if (!cache.enabled) break;
    }
    flush();
    if (!cache.enabled) return { skipped: cacheError ?? "the embedding cache failed" };
    return { harvested, vectors };
  } catch (error) {
    return { skipped: `the index could not be read (${errorMessage(error)})` };
  } finally {
    cache?.close();
    index.close();
  }
}
