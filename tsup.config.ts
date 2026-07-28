import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts", "src/cli.ts", "src/mcp.ts", "src/cli-entry.ts", "src/mcp-entry.ts"],
  format: ["esm"],
  target: "node22.13",
  removeNodeProtocol: false,
  dts: true,
  sourcemap: true,
  clean: true,
  external: ["node:sqlite", "sqlite-vec", "@huggingface/transformers"],
});