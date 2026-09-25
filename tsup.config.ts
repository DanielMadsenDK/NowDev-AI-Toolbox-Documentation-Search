import { defineConfig } from "tsup";

export default defineConfig({
  // embedding-worker.ts is loaded by path from the bundled embedder (new URL("./embedding-worker.js", import.meta.url)), so it must stay a top-level entry.
  entry: ["src/index.ts", "src/cli.ts", "src/mcp.ts", "src/cli-entry.ts", "src/mcp-entry.ts", "src/embedding-worker.ts"],
  format: ["esm"],
  target: "node22.13",
  removeNodeProtocol: false,
  dts: true,
  sourcemap: true,
  clean: true,
  external: ["node:sqlite", "sqlite-vec", "@huggingface/transformers"],
});