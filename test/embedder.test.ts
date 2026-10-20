import os from "node:os";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createEmbeddingBatches, DEFAULT_MAX_EMBEDDING_WORKERS, EndpointDocumentEmbeddingProvider, HashEmbeddingProvider, TransformersEmbeddingProvider, truncateEmbeddingText } from "../src/embedder.js";
import { EmbeddingWorkerPool } from "../src/embedding-pool.js";
import { DEFAULT_EMBEDDING_PROFILE, EMBEDDING_PROFILES } from "../src/embedding-profiles.js";
import { TransformersReranker } from "../src/reranker.js";

const fakeWorkerUrl = new URL("./fixtures/fake-embedding-worker.mjs", import.meta.url);
const disposables: Array<{ dispose(): Promise<void> }> = [];
afterEach(async () => {
  await Promise.all(disposables.splice(0).map((item) => item.dispose()));
});

function counters() {
  const buffer = new SharedArrayBuffer(3 * Int32Array.BYTES_PER_ELEMENT);
  return { buffer, peak: () => Atomics.load(new Int32Array(buffer), 1), jobs: () => Atomics.load(new Int32Array(buffer), 2) };
}

function pool(options: Partial<ConstructorParameters<typeof EmbeddingWorkerPool>[0]> & { workerData?: Record<string, unknown> } = {}): EmbeddingWorkerPool {
  const created = new EmbeddingWorkerPool({ size: 2, workerUrl: fakeWorkerUrl, ...options, workerData: { dimensions: 4, ...options.workerData } });
  disposables.push(created);
  return created;
}

function pooledProvider(options: ConstructorParameters<typeof TransformersEmbeddingProvider>[0] = {}): TransformersEmbeddingProvider {
  // sortByLength: false skips token counting, so no tokenizer (or model) is ever loaded.
  const provider = new TransformersEmbeddingProvider({ workers: 3, dimensions: 4, batchSize: 2, sortByLength: false, workerUrl: fakeWorkerUrl, ...options });
  disposables.push(provider);
  return provider;
}

describe("createEmbeddingBatches", () => {
  it("groups similarly sized texts while preserving original indices", () => {
    const texts = ["medium text", "x", "the longest text in this test", "short"];
    const batches = createEmbeddingBatches(texts, 2);
    expect(batches.map((batch) => batch.map((item) => item.text))).toEqual([
      ["x", "short"],
      ["medium text", "the longest text in this test"],
    ]);
    const restored = new Array<string>(texts.length);
    batches.flat().forEach((item) => { restored[item.index] = item.text; });
    expect(restored).toEqual(texts);
  });

  it("keeps estimated token usage within the batch budget", () => {
    const batches = createEmbeddingBatches(["short", "x".repeat(40), "y".repeat(40)], 3, false, 23);
    expect(batches.map((batch) => batch.length)).toEqual([2, 1]);
  });

  it("rejects an individual input that exceeds the batch token budget", () => {
    expect(() => createEmbeddingBatches(["x".repeat(40)], 3, false, 12)).toThrow("exceeds the configured batch token budget");
  });

  it("orders by real token counts when they are known", () => {
    // Character length and token count disagree: the long prose tokenizes shorter than the short identifier soup.
    const texts = ["plain prose that is long", "a.b_c(d)", "mid"];
    const batches = createEmbeddingBatches(texts, 2, true, Number.POSITIVE_INFINITY, [5, 9, 1]);
    expect(batches.map((batch) => batch.map((item) => item.index))).toEqual([[2, 0], [1]]);
    expect(() => createEmbeddingBatches(texts, 2, true, Number.POSITIVE_INFINITY, [1])).toThrow("Expected 3 token counts");
  });

  it("splits the CPU thread budget across embedding workers", () => {
    vi.spyOn(os, "totalmem").mockReturnValue(32 * 1024 ** 3);
    const budgeted = new TransformersEmbeddingProvider({ threads: 4 });
    expect(budgeted).toMatchObject({ workers: 4, threadsPerWorker: 1, batchSize: 16 });
    const capped = new TransformersEmbeddingProvider({ threads: 64 });
    expect(capped).toMatchObject({ workers: DEFAULT_MAX_EMBEDDING_WORKERS, threadsPerWorker: 8 });
    const inProcess = new TransformersEmbeddingProvider({ threads: 6, workers: 1 });
    expect(inProcess).toMatchObject({ workers: 1, threadsPerWorker: 6, batchSize: 32 });
    // GPU devices keep the pool for the CPU fallback after a GPU failure.
    expect(new TransformersEmbeddingProvider({ device: "dml", threads: 8 })).toMatchObject({ workers: 8, batchSize: 8 });
    vi.restoreAllMocks();
  });

  it("caps the default worker count by each worker's model memory", () => {
    vi.spyOn(os, "totalmem").mockReturnValue(16 * 1024 ** 3);
    // Large models get few workers: six ~1.5 GB nomic workers pushed a 16 GB Mac deep into swap.
    const nomic = new TransformersEmbeddingProvider({ ...EMBEDDING_PROFILES["nomic-embed-text-v1.5"], threads: 10 });
    expect(nomic).toMatchObject({ workers: 3, threadsPerWorker: 3 });
    expect(new TransformersEmbeddingProvider({ threads: 10 }).workers).toBe(DEFAULT_MAX_EMBEDDING_WORKERS);
    expect(new TransformersEmbeddingProvider({ model: "custom/model", threads: 10 }).workers).toBe(3);
    // An explicit worker count is the user's decision, and some worker always remains.
    expect(new TransformersEmbeddingProvider({ ...EMBEDDING_PROFILES["nomic-embed-text-v1.5"], threads: 10, workers: 6 }).workers).toBe(6);
    vi.spyOn(os, "totalmem").mockReturnValue(2 * 1024 ** 3);
    expect(new TransformersEmbeddingProvider({ ...EMBEDDING_PROFILES["nomic-embed-text-v1.5"], threads: 10 }).workers).toBe(1);
    vi.restoreAllMocks();
  });

  it("exposes the configured execution device and batch size", () => {
    const provider = new TransformersEmbeddingProvider({ device: "webgpu", batchSize: 64 });
    expect(provider.device).toBe("webgpu");
    expect(provider.batchSize).toBe(64);
  });

  it("uses a conservative default batch size for DirectML", () => {
    const provider = new TransformersEmbeddingProvider({ device: "dml" });
    expect(provider.batchSize).toBe(8);
    expect(provider.maxBatchTokens).toBe(4096);
    expect(provider.maxEmbeddingCharacters).toBe(EMBEDDING_PROFILES[DEFAULT_EMBEDDING_PROFILE].maxEmbeddingCharacters);
  });

  it("defaults CPU thread count to the available parallelism and allows overriding it", () => {
    const defaultProvider = new TransformersEmbeddingProvider();
    expect(defaultProvider.threads).toBe(os.availableParallelism());
    const overriddenProvider = new TransformersEmbeddingProvider({ threads: 2 });
    expect(overriddenProvider.threads).toBe(2);
  });

  it("takes unset options from the default embedding profile", () => {
    const provider = new TransformersEmbeddingProvider();
    const { minDimensions: _minDimensions, ...profile } = EMBEDDING_PROFILES[DEFAULT_EMBEDDING_PROFILE];
    expect(provider).toMatchObject(profile);
  });

  it("defines model-correct curated ONNX embedding profiles", () => {
    expect(DEFAULT_EMBEDDING_PROFILE).toBe("all-minilm-l6-v2");
    expect(EMBEDDING_PROFILES["nomic-embed-text-v1.5"]).toMatchObject({ dimensions: 768, pooling: "mean", layerNorm: true, documentPrefix: "search_document: ", queryPrefix: "search_query: " });
    expect(EMBEDDING_PROFILES["nomic-embed-text-v1"]).toMatchObject({ dimensions: 768, pooling: "mean", layerNorm: false, documentPrefix: "search_document: ", queryPrefix: "search_query: " });
    expect(EMBEDDING_PROFILES["multilingual-e5-small"]).toMatchObject({ dimensions: 384, pooling: "mean", documentPrefix: "passage: ", queryPrefix: "query: " });
    expect(EMBEDDING_PROFILES["all-minilm-l6-v2"]).toMatchObject({ dimensions: 384, pooling: "mean", documentPrefix: "", queryPrefix: "", maxEmbeddingCharacters: 1024 });
  });

  it("only marks the Matryoshka-trained profile as supporting custom dimensions", () => {
    expect(EMBEDDING_PROFILES["nomic-embed-text-v1.5"].minDimensions).toBe(64);
    expect(EMBEDDING_PROFILES["bge-base-en-v1.5"].minDimensions).toBeUndefined();
    expect(EMBEDDING_PROFILES["nomic-embed-text-v1"].minDimensions).toBeUndefined();
    expect(EMBEDDING_PROFILES["multilingual-e5-small"].minDimensions).toBeUndefined();
    expect(EMBEDDING_PROFILES["all-minilm-l6-v2"].minDimensions).toBeUndefined();
  });

  it("caps only the text representation sent to the embedding model", () => {
    const text = "prefix\n" + "x".repeat(20);
    expect(truncateEmbeddingText(text, 12)).toBe("prefix\nxxxxx");
    expect(truncateEmbeddingText(text, 0)).toBe(text);
    expect(truncateEmbeddingText("short", 12)).toBe("short");
    expect(truncateEmbeddingText("abc😀def", 4)).toBe("abc");
  });

  it("rejects embedding caps that cannot hold configured prefixes", () => {
    expect(() => new TransformersEmbeddingProvider({ queryPrefix: "query: ", maxEmbeddingCharacters: 7 })).toThrow("must exceed");
  });

  it("rejects whitespace-only inputs consistently", async () => {
    await expect(new HashEmbeddingProvider(4).embed(["  "])).rejects.toThrow("index 0 is empty");
    await expect(new TransformersEmbeddingProvider().embed(["\n"])).rejects.toThrow("index 0 is empty");
  });
});

describe("EmbeddingWorkerPool", () => {
  it("runs at most one job per worker at a time", async () => {
    const usage = counters();
    const workers = pool({ size: 2, workerData: { counters: usage.buffer, delayMilliseconds: 20 } });
    const results = await Promise.all(["a", "bb", "ccc", "dddd", "eeeee", "ffffff"].map((text) => workers.run([text])));
    expect(results.map((data) => data[0])).toEqual([1, 2, 3, 4, 5, 6]);
    expect(usage.jobs()).toBe(6);
    expect(usage.peak()).toBe(2);
  });

  it("rejects a job whose worker reports an error and keeps serving others", async () => {
    const workers = pool();
    await expect(workers.run(["fail"])).rejects.toThrow("simulated failure");
    expect((await workers.run(["ok"]))[0]).toBe(2);
  });

  it("replaces a crashed worker on the next job", async () => {
    const workers = pool({ size: 1 });
    await expect(workers.run(["crash"])).rejects.toThrow("exited unexpectedly");
    expect((await workers.run(["again"]))[0]).toBe(5);
  });

  it("fails to start when a worker cannot load", async () => {
    await expect(pool({ workerData: { failStartup: true } }).start()).rejects.toThrow("simulated startup failure");
  });

  it("runs with the workers that started when a later one fails to load", async () => {
    const startups = new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT);
    const workers = pool({ size: 3, workerData: { startups, failStartupFrom: 2 } });
    await workers.start();
    expect(workers.size).toBe(2);
    expect(workers.capacity).toBe(3);
    expect((await workers.run(["abc"]))[0]).toBe(3);
  });

  it("shuts down workers whose start finishes after the pool was disposed", async () => {
    const workers = pool({ size: 2, workerData: { delayMilliseconds: 0 } });
    const starting = workers.start();
    await workers.dispose();
    await expect(starting).rejects.toThrow("disposed");
    expect((workers as unknown as { workers: unknown[] }).workers).toHaveLength(0);
  });

  it("replaces a crashed worker while others keep running", async () => {
    const workers = pool({ size: 2 });
    await workers.start();
    await expect(workers.run(["crash"])).rejects.toThrow("exited unexpectedly");
    await vi.waitFor(() => expect((workers as unknown as { workers: unknown[] }).workers).toHaveLength(2));
    expect(await Promise.all([workers.run(["a"]), workers.run(["bb"])])).toEqual([expect.any(Float32Array), expect.any(Float32Array)]);
  });

  it("keeps a worker error's cause, so out-of-memory detection sees it", async () => {
    const workers = pool();
    const error = await workers.run(["caused"]).catch((failure: unknown) => failure as Error);
    expect(error).toBeInstanceOf(Error);
    expect((error as Error).cause).toMatchObject({ message: "underlying cause" });
  });

  it("shuts down while idle and restarts for the next job", async () => {
    const workers = pool({ idleMilliseconds: 10 });
    await workers.run(["first"]);
    await vi.waitFor(() => expect((workers as unknown as { workers: unknown[] }).workers).toHaveLength(0));
    expect((await workers.run(["second"]))[0]).toBe(6);
  });
});

describe("TransformersEmbeddingProvider worker pool", () => {
  it("embeds bulk batches across workers and restores input order", async () => {
    const provider = pooledProvider();
    const texts = ["a", "bbb", "cc", "dddddd", "eeee", "fffff", "g"];
    const vectors = await provider.embed(texts);
    expect(vectors.map((vector) => vector[0])).toEqual(texts.map((text) => text.length));
    expect(provider.metrics.workerFallbacks).toBe(0);
  });

  it("splits a batch that runs out of memory inside a worker", async () => {
    const provider = pooledProvider({ batchSize: 4 });
    const vectors = await provider.embed(["one", "oom", "three", "four"]);
    expect(vectors.map((vector) => vector[0])).toEqual([3, 3, 5, 4]);
    expect(provider.metrics.batchSplits).toBeGreaterThan(0);
  });

  it("delivers no batch after the pass has failed", async () => {
    const provider = pooledProvider({ batchSize: 1 });
    let delivered = 0;
    let settled = false;
    const pass = provider.embedBatched(["a", "b", "fail", "d", "e", "f", "g", "h"], () => {
      if (settled) throw new Error("batch delivered after the pass failed");
      delivered += 1;
    }).finally(() => { settled = true; });
    await expect(pass).rejects.toThrow("simulated failure");
    const deliveredAtFailure = delivered;
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(delivered).toBe(deliveredAtFailure);
    expect(delivered).toBeLessThan(7);
  });

  it("moves embedding to the CPU after a GPU failure and pools the next pass", async () => {
    const provider = pooledProvider({ device: "webgpu" });
    vi.spyOn(provider as unknown as { inferInProcess(texts: string[]): Promise<Float32Array[]> }, "inferInProcess").mockImplementation(async (texts) => {
      if (provider.activeDevice !== "cpu") throw new Error("WebGPU device was lost");
      return texts.map(() => new Float32Array([9, 0, 0, 0]));
    });
    const errors = vi.spyOn(console, "error").mockImplementation(() => undefined);
    // The failing pass finishes in-process on the CPU...
    expect((await provider.embed(["a", "bb", "ccc"])).map((vector) => vector[0])).toEqual([9, 9, 9]);
    expect(provider.activeDevice).toBe("cpu");
    expect(provider.metrics.deviceTransitions).toBe(1);
    expect(errors).toHaveBeenCalledWith(expect.stringContaining("continuing on the CPU"));
    // ...and later passes use the CPU worker pool, whose fixture vectors carry each text's length.
    expect((await provider.embed(["dddd", "e"])).map((vector) => vector[0])).toEqual([4, 1]);
    errors.mockRestore();
  });

  it("embeds queries on the CPU even when indexing on a GPU", async () => {
    const provider = pooledProvider({ device: "webgpu" });
    const inProcess = vi.spyOn(provider as unknown as { inferInProcess(texts: string[], query?: boolean): Promise<Float32Array[]> }, "inferInProcess")
      .mockImplementation(async (texts) => texts.map(() => new Float32Array(4)));
    await provider.embedQuery("GlideRecord");
    expect(inProcess).toHaveBeenCalledWith(["GlideRecord"], true);
    expect(provider.activeDevice).toBe("webgpu");
  });

  it("falls back to in-process embedding when the worker script is missing", async () => {
    const provider = pooledProvider({ workerUrl: new URL("./fixtures/missing-worker.mjs", import.meta.url) });
    const inProcess = vi.spyOn(provider as unknown as { inferInProcess(texts: string[]): Promise<Float32Array[]> }, "inferInProcess")
      .mockImplementation(async (texts) => texts.map(() => new Float32Array([7, 0, 0, 0])));
    const errors = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const vectors = await provider.embed(["a", "b", "c"]);
    expect(vectors.map((vector) => vector[0])).toEqual([7, 7, 7]);
    expect(inProcess).toHaveBeenCalled();
    expect(provider.metrics.workerFallbacks).toBe(1);
    expect(errors).toHaveBeenCalledWith(expect.stringContaining("worker script is missing"));
    errors.mockRestore();
  });
});

describe("TransformersEmbeddingProvider batch token budget", () => {
  it("shortens a text that alone exceeds the batch token budget instead of rejecting it", async () => {
    const provider = new TransformersEmbeddingProvider({ dimensions: 4, maxBatchTokens: 30, maxEmbeddingCharacters: 1000, sortByLength: false, workers: 1, documentPrefix: "", queryPrefix: "" });
    disposables.push(provider);
    const inProcess = vi.spyOn(provider as unknown as { inferInProcess(texts: string[]): Promise<Float32Array[]> }, "inferInProcess")
      .mockImplementation(async (texts) => texts.map(() => new Float32Array(4)));
    await provider.embed(["x".repeat(100)]);
    expect(inProcess.mock.calls[0]![0][0]).toHaveLength(20);
    expect(provider.metrics.truncations).toBe(1);
  });
});

describe("TransformersReranker", () => {
  function reranker(logits: number[], columns: number) {
    const created = new TransformersReranker();
    const disposed: string[] = [];
    const tensor = (name: string) => ({ dispose: () => disposed.push(name) });
    vi.spyOn(created as unknown as { load(): Promise<unknown> }, "load").mockResolvedValue({
      tokenizer: () => ({ input_ids: tensor("input_ids"), attention_mask: tensor("attention_mask") }),
      model: async () => ({ logits: { data: Float32Array.from(logits), dims: [logits.length / columns, columns], dispose: () => disposed.push("logits") } }),
    });
    return { created, disposed };
  }

  it("returns single-logit cross-encoder scores unchanged", async () => {
    const { created, disposed } = reranker([2.5, -1], 1);
    expect(await created.score("query", ["first", "second"])).toEqual([2.5, -1]);
    expect(disposed.sort()).toEqual(["attention_mask", "input_ids", "logits"]);
  });

  it("scores two-class heads as the relevant class's log-odds", async () => {
    const { created } = reranker([1, 3, 2, -2], 2);
    expect(await created.score("query", ["relevant", "irrelevant"])).toEqual([2, -4]);
  });
});

describe("EndpointDocumentEmbeddingProvider", () => {
  it("embeds document batches remotely and queries through the local provider", async () => {
    const requests: Array<{ url: string; authorization: string | null; body: Record<string, unknown> }> = [];
    const fetchImplementation: typeof fetch = async (input, init) => {
      requests.push({
        url: String(input),
        authorization: new Headers(init?.headers).get("authorization"),
        body: JSON.parse(String(init?.body)) as Record<string, unknown>,
      });
      return Response.json({ data: [
        { index: 0, embedding: [3, 4, 0, 0] },
        { index: 1, embedding: [0, 0, 0, 2] },
      ] });
    };
    const local = new HashEmbeddingProvider(4);
    const provider = new EndpointDocumentEmbeddingProvider(local, {
      endpoint: "https://openrouter.ai/api/v1/embeddings",
      apiKey: "test-secret",
      model: "baai/bge-base-en-v1.5",
      fetch: fetchImplementation,
    });
    expect(provider.endpointConcurrency).toBe(4);

    const vectors = await provider.embed(["first", "second"]);
    expect(vectors[0]![0]).toBeCloseTo(0.6);
    expect(vectors[0]![1]).toBeCloseTo(0.8);
    expect([...vectors[0]!.slice(2)]).toEqual([0, 0]);
    expect([...vectors[1]!]).toEqual([0, 0, 0, 1]);
    expect(requests).toEqual([{
      url: "https://openrouter.ai/api/v1/embeddings",
      authorization: "Bearer test-secret",
      body: { model: "baai/bge-base-en-v1.5", input: ["first", "second"], encoding_format: "float" },
    }]);
    expect(await provider.embedQuery("offline query")).toEqual(await local.embedQuery("offline query"));
  });

  it("preserves response order when every vector index is absent", async () => {
    const provider = new EndpointDocumentEmbeddingProvider(new HashEmbeddingProvider(4), {
      endpoint: "https://openrouter.ai/api/v1/embeddings",
      apiKey: "test-secret",
      model: "example/model",
      fetch: async () => Response.json({ data: [{ embedding: [1, 0, 0, 0] }, { embedding: [0, 1, 0, 0] }] }),
    });
    const vectors = await provider.embed(["first", "second"]);
    expect([...vectors[0]!]).toEqual([1, 0, 0, 0]);
    expect([...vectors[1]!]).toEqual([0, 1, 0, 0]);
  });

  it.each([
    ["mixed indexes", [{ index: 0, embedding: [1, 0, 0, 0] }, { embedding: [0, 1, 0, 0] }], "mixture"],
    ["duplicate indexes", [{ index: 0, embedding: [1, 0, 0, 0] }, { index: 0, embedding: [0, 1, 0, 0] }], "unique permutation"],
    ["out-of-range indexes", [{ index: 0, embedding: [1, 0, 0, 0] }, { index: 2, embedding: [0, 1, 0, 0] }], "unique permutation"],
  ])("rejects endpoint responses with %s", async (_name, data, expected) => {
    const provider = new EndpointDocumentEmbeddingProvider(new HashEmbeddingProvider(4), {
      endpoint: "https://openrouter.ai/api/v1/embeddings",
      apiKey: "test-secret",
      model: "example/model",
      fetch: async () => Response.json({ data }),
    });
    await expect(provider.embed(["first", "second"])).rejects.toThrow(expected);
  });

  it("rejects endpoint URLs that could leak credentials", () => {
    expect(() => new EndpointDocumentEmbeddingProvider(new HashEmbeddingProvider(4), {
      endpoint: "https://secret@example.com/embeddings?token=secret",
      apiKey: "test-secret",
      model: "example/model",
    })).toThrow("must not contain credentials");
  });

  it("isolates and shortens inputs rejected by an endpoint context limit", async () => {
    const inputs: string[][] = [];
    const fetchImplementation: typeof fetch = async (_input, init) => {
      const body = JSON.parse(String(init?.body)) as { input: string[] };
      inputs.push(body.input);
      if (body.input.some((text) => text.length > 90)) {
        return Response.json({ error: { message: "You passed 513 input tokens; the model's context length is only 512 tokens (parameter=input_tokens)" } }, { status: 400 });
      }
      return Response.json({ data: body.input.map((_, index) => ({ index, embedding: [1, 0, 0, 0] })) });
    };
    const provider = new EndpointDocumentEmbeddingProvider(new HashEmbeddingProvider(4), {
      endpoint: "https://openrouter.ai/api/v1/embeddings",
      apiKey: "test-secret",
      model: "baai/bge-base-en-v1.5",
      fetch: fetchImplementation,
    });

    const vectors = await provider.embed(["x".repeat(100), "short"]);

    expect(vectors).toHaveLength(2);
    expect(inputs).toContainEqual(["x".repeat(90)]);
    expect(inputs).toContainEqual(["short"]);
    expect(provider.metrics).toMatchObject({ truncations: 1, batchSplits: 1 });
  });

  it("bounds repeated context-limit reductions", async () => {
    let requests = 0;
    const provider = new EndpointDocumentEmbeddingProvider(new HashEmbeddingProvider(4), {
      endpoint: "https://openrouter.ai/api/v1/embeddings",
      apiKey: "test-secret",
      model: "example/model",
      fetch: async () => {
        requests += 1;
        return Response.json({ error: { message: "maximum input length exceeded" } }, { status: 400 });
      },
    });
    await expect(provider.embed(["x".repeat(1000)])).rejects.toThrow("after 8 context-limit reductions");
    expect(requests).toBe(9);
  });

  it("runs remote batches with bounded concurrency", async () => {
    let activeRequests = 0;
    let maximumActiveRequests = 0;
    const releases: Array<() => void> = [];
    const fetchImplementation: typeof fetch = async (_input, init) => {
      activeRequests += 1;
      maximumActiveRequests = Math.max(maximumActiveRequests, activeRequests);
      await new Promise<void>((resolve) => releases.push(resolve));
      activeRequests -= 1;
      const body = JSON.parse(String(init?.body)) as { input: string[] };
      return Response.json({ data: body.input.map((_, index) => ({ index, embedding: [1, 0, 0, 0] })) });
    };
    const provider = new EndpointDocumentEmbeddingProvider(new HashEmbeddingProvider(4), {
      endpoint: "https://openrouter.ai/api/v1/embeddings",
      apiKey: "test-secret",
      model: "baai/bge-base-en-v1.5",
      batchSize: 1,
      concurrency: 2,
      fetch: fetchImplementation,
    });

    const embedding = provider.embed(["one", "two", "three"]);
    await vi.waitFor(() => expect(releases).toHaveLength(2));
    releases.splice(0).forEach((release) => release());
    await vi.waitFor(() => expect(releases).toHaveLength(1));
    releases.splice(0).forEach((release) => release());
    await embedding;

    expect(maximumActiveRequests).toBe(2);
  });

  it("continues draining batches while another request is stalled", async () => {
    let releaseFirstRequest: (() => void) | undefined;
    const requestedInputs: string[] = [];
    const fetchImplementation: typeof fetch = async (_input, init) => {
      const body = JSON.parse(String(init?.body)) as { input: string[] };
      requestedInputs.push(body.input[0]!);
      if (body.input[0] === "one") await new Promise<void>((resolve) => { releaseFirstRequest = resolve; });
      return Response.json({ data: [{ index: 0, embedding: [1, 0, 0, 0] }] });
    };
    const provider = new EndpointDocumentEmbeddingProvider(new HashEmbeddingProvider(4), {
      endpoint: "https://openrouter.ai/api/v1/embeddings",
      apiKey: "test-secret",
      model: "baai/bge-base-en-v1.5",
      batchSize: 1,
      concurrency: 2,
      fetch: fetchImplementation,
    });

    const embedding = provider.embed(["one", "two", "three"]);
    await vi.waitFor(() => expect(requestedInputs).toEqual(["one", "two", "three"]));
    releaseFirstRequest!();
    await embedding;
  });

  it("emits successful worker batches before reporting another worker's failure", async () => {
    const emittedIndexes: number[] = [];
    const provider = new EndpointDocumentEmbeddingProvider(new HashEmbeddingProvider(4), {
      endpoint: "https://openrouter.ai/api/v1/embeddings",
      apiKey: "test-secret",
      model: "example/model",
      batchSize: 1,
      concurrency: 2,
      fetch: async (_input, init) => {
        const body = JSON.parse(String(init?.body)) as { input: string[] };
        if (body.input[0] === "bad") return Response.json({ error: { message: "invalid input" } }, { status: 400 });
        return Response.json({ data: [{ index: 0, embedding: [1, 0, 0, 0] }] });
      },
    });
    await expect(provider.embedBatched(["bad", "good", "later"], (batch) => { emittedIndexes.push(...batch.indexes); })).rejects.toThrow("invalid input");
    expect(emittedIndexes).toEqual([1, 2]);
  });

  it("delegates the active device dynamically to the query provider", () => {
    class MutableDeviceProvider extends HashEmbeddingProvider {
      activeDevice: "dml" | "cpu" = "dml";
    }
    const local = new MutableDeviceProvider(4);
    const provider = new EndpointDocumentEmbeddingProvider(local, {
      endpoint: "https://openrouter.ai/api/v1/embeddings",
      apiKey: "test-secret",
      model: "example/model",
    });
    expect(provider.activeDevice).toBe("dml");
    local.activeDevice = "cpu";
    expect(provider.activeDevice).toBe("cpu");
  });
});