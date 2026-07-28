import os from "node:os";
import { describe, expect, it } from "vitest";
import { createEmbeddingBatches, TransformersEmbeddingProvider, truncateEmbeddingText } from "../src/embedder.js";
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
    const batches = createEmbeddingBatches(["short", "x".repeat(40), "y".repeat(40)], 3, false, 12);
    expect(batches.map((batch) => batch.length)).toEqual([2, 1]);
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

  it("defaults CPU thread count to the host's logical cores and allows overriding it", () => {
    const defaultProvider = new TransformersEmbeddingProvider();
    expect(defaultProvider.threads).toBe(os.cpus().length);
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
  });
});