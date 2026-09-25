import type { DatabaseSync } from "node:sqlite";

export interface VectorFilters {
  release?: string;
  docType?: string;
  publication?: string;
  chunkType?: string;
  topicType?: string;
}

const FILTER_KEYS = ["release", "docType", "publication", "chunkType", "topicType"] as const;
const LOAD_YIELD_INTERVAL = 50_000;
const MISSING = 0xffff;

function popcount(value: number): number {
  value -= (value >>> 1) & 0x55555555;
  value = (value & 0x33333333) + ((value >>> 2) & 0x33333333);
  return (((value + (value >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

/** Sign-bit quantization in sqlite-vec's vec_quantize_binary layout (bit i of byte i/8, least significant bit first). */
export function quantizeBinary(vector: Float32Array): Uint8Array {
  const code = new Uint8Array(Math.ceil(vector.length / 8));
  for (let index = 0; index < vector.length; index += 1) if (vector[index]! > 0) code[index >> 3]! |= 1 << (index & 7);
  return code;
}

// Brute-force Hamming scan over every binary code held in memory. sqlite-vec's vec0 KNN cost grows with k, so the
// oversampled coarse pass (k in the hundreds) took hundreds of milliseconds on a 700k-chunk index; the same scan over
// packed 32-bit words takes about ten. Loaded once for long-running processes (the MCP server) and dropped on writes.
export class BinaryVectorIndex {
  private constructor(
    private readonly ids: Float64Array,
    private readonly codes: Uint32Array,
    private readonly words: number,
    private readonly attributes: Uint16Array[],
    private readonly dictionaries: Array<Map<string, number>>,
  ) {}

  get size(): number {
    return this.ids.length;
  }

  static async load(db: DatabaseSync, dimensions: number): Promise<BinaryVectorIndex> {
    const words = Math.ceil(dimensions / 32);
    const bytes = dimensions / 8;
    const count = Number((db.prepare("SELECT count(*) AS count FROM documents").get() as { count: number | bigint }).count);
    const maxId = Number((db.prepare("SELECT coalesce(max(id), 0) AS id FROM documents").get() as { id: number | bigint }).id);
    const ids = new Float64Array(count);
    const codes = new Uint32Array(count * words);
    const rowOfId = new Int32Array(maxId + 1).fill(-1);
    const padded = new Uint8Array(words * 4);
    let row = 0;
    for (const item of db.prepare("SELECT rowid, coarse_embedding FROM document_vectors").iterate() as Iterable<{ rowid: number | bigint; coarse_embedding: Uint8Array }>) {
      if (row >= count) break;
      const id = Number(item.rowid);
      if (id > maxId) continue;
      padded.fill(0);
      padded.set(item.coarse_embedding.subarray(0, bytes));
      codes.set(new Uint32Array(padded.buffer, 0, words), row * words);
      ids[row] = id;
      rowOfId[id] = row;
      row += 1;
      if (row % LOAD_YIELD_INTERVAL === 0) await new Promise((resolve) => setImmediate(resolve));
    }
    const attributes = FILTER_KEYS.map(() => new Uint16Array(row).fill(MISSING));
    const dictionaries = FILTER_KEYS.map(() => new Map<string, number>());
    let scanned = 0;
    for (const item of db.prepare("SELECT id, release, doc_type, publication, chunk_type, coalesce(topic_type, '') AS topic_type FROM documents").iterate() as Iterable<{ id: number | bigint; release: string; doc_type: string; publication: string; chunk_type: string; topic_type: string }>) {
      const index = rowOfId[Number(item.id)] ?? -1;
      if (index >= 0) {
        [item.release, item.doc_type, item.publication, item.chunk_type, item.topic_type].forEach((value, column) => {
          const dictionary = dictionaries[column]!;
          let code = dictionary.get(value);
          if (code === undefined) {
            code = dictionary.size;
            dictionary.set(value, code);
          }
          attributes[column]![index] = code;
        });
      }
      scanned += 1;
      if (scanned % LOAD_YIELD_INTERVAL === 0) await new Promise((resolve) => setImmediate(resolve));
    }
    return new BinaryVectorIndex(ids.subarray(0, row), codes.subarray(0, row * words), words, attributes, dictionaries);
  }

  /** Returns up to `count` document ids closest to `vector` by Hamming distance, nearest first. */
  nearest(vector: Float32Array, count: number, filters: VectorFilters): number[] {
    const required: Array<[Uint16Array, number]> = [];
    for (const [column, key] of FILTER_KEYS.entries()) {
      const value = filters[key];
      if (value === undefined) continue;
      const code = this.dictionaries[column]!.get(value);
      if (code === undefined) return [];
      required.push([this.attributes[column]!, code]);
    }
    const queryBytes = new Uint8Array(this.words * 4);
    queryBytes.set(quantizeBinary(vector).subarray(0, queryBytes.length));
    const query = new Uint32Array(queryBytes.buffer);
    const rows = this.ids.length;
    const distances = new Uint16Array(rows);
    const histogram = new Uint32Array(this.words * 32 + 2);
    const excluded = histogram.length - 1;
    for (let row = 0; row < rows; row += 1) {
      let eligible = true;
      for (const [attribute, code] of required) {
        if (attribute[row] !== code) {
          eligible = false;
          break;
        }
      }
      if (!eligible) {
        distances[row] = excluded;
        continue;
      }
      let distance = 0;
      const offset = row * this.words;
      for (let word = 0; word < this.words; word += 1) distance += popcount(this.codes[offset + word]! ^ query[word]!);
      distances[row] = distance;
      histogram[distance]! += 1;
    }
    // Counting selection: find the distance cutoff that yields `count` rows instead of sorting every row.
    let cutoff = 0;
    let selected = 0;
    while (cutoff < excluded && selected + histogram[cutoff]! < count) selected += histogram[cutoff++]!;
    const result: Array<{ id: number; distance: number }> = [];
    let ties = count - selected;
    for (let row = 0; row < rows; row += 1) {
      const distance = distances[row]!;
      if (distance < cutoff) result.push({ id: this.ids[row]!, distance });
      else if (distance === cutoff && cutoff < excluded && ties > 0) {
        result.push({ id: this.ids[row]!, distance });
        ties -= 1;
      }
    }
    return result.sort((left, right) => left.distance - right.distance).map((item) => item.id);
  }
}
