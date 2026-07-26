import { describe, expect, it } from "vitest";
import { createEmbeddingBatches } from "../src/embedder.js";

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
});