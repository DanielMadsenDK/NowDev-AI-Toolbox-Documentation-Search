import os from "node:os";
import { describe, expect, it, vi } from "vitest";
import { createEmbeddingBatches, EndpointDocumentEmbeddingProvider, HashEmbeddingProvider, TransformersEmbeddingProvider, truncateEmbeddingText } from "../src/embedder.js";
import { DEFAULT_EMBEDDING_PROFILE, EMBEDDING_PROFILES } from "../src/embedding-profiles.js";

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

  it("exposes the configured execution device and batch size", () => {
    const provider = new TransformersEmbeddingProvider({ device: "webgpu", batchSize: 64 });
    expect(provider.device).toBe("webgpu");
    expect(provider.batchSize).toBe(64);
  });

  it("uses a conservative default batch size for DirectML", () => {
    const provider = new TransformersEmbeddingProvider({ device: "dml" });
    expect(provider.batchSize).toBe(8);
    expect(provider.maxBatchTokens).toBe(4096);
    expect(provider.maxEmbeddingCharacters).toBe(2048);
  });

  it("defaults CPU thread count to the available parallelism and allows overriding it", () => {
    const defaultProvider = new TransformersEmbeddingProvider();
    expect(defaultProvider.threads).toBe(os.availableParallelism());
    const overriddenProvider = new TransformersEmbeddingProvider({ threads: 2 });
    expect(overriddenProvider.threads).toBe(2);
  });

  it("uses the BGE embedding profile by default", () => {
    const provider = new TransformersEmbeddingProvider();
    expect(provider.model).toBe("Xenova/bge-base-en-v1.5");
    expect(provider.dimensions).toBe(768);
    expect(provider.pooling).toBe("cls");
    expect(provider.layerNorm).toBe(false);
    expect(provider.documentPrefix).toBe("");
    expect(provider.queryPrefix).toBe("Represent this sentence for searching relevant passages: ");
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