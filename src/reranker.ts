import os from "node:os";
import type { PreTrainedModel, PreTrainedTokenizer } from "@huggingface/transformers";
import type { EmbeddingDevice } from "./embedder.js";

export const DEFAULT_RERANKER_MODEL = "jinaai/jina-reranker-v1-turbo-en";

export interface Reranker {
  readonly model: string;
  /** Returns one relevance logit per document; higher is more relevant. */
  score(query: string, documents: string[]): Promise<number[]>;
  /** Loads the model ahead of the first query. */
  warmUp?(): Promise<void>;
}

export interface TransformersRerankerOptions {
  model?: string;
  cacheDirectory?: string;
  device?: EmbeddingDevice;
  dtype?: "auto" | "fp32" | "fp16" | "q8" | "q4" | "q4f16";
  /** Token budget per query/document pair; longer documents are truncated. */
  maxLength?: number;
  threads?: number;
}

interface LoadedCrossEncoder {
  tokenizer: PreTrainedTokenizer;
  model: PreTrainedModel;
}

// Cross-encoder reranker: scores each (query, passage) pair jointly, which captures intent that the
// bi-encoder embedding and bm25 legs miss. Only the top fused candidates are scored, so cost stays bounded.
export class TransformersReranker implements Reranker {
  readonly model: string;
  private readonly options: TransformersRerankerOptions;
  private loading?: Promise<LoadedCrossEncoder>;

  constructor(options: TransformersRerankerOptions = {}) {
    this.options = options;
    this.model = options.model ?? DEFAULT_RERANKER_MODEL;
  }

  private load(): Promise<LoadedCrossEncoder> {
    if (!this.loading) {
      const loading = import("@huggingface/transformers").then(async (transformers) => {
        if (this.options.cacheDirectory) transformers.env.cacheDir = this.options.cacheDirectory;
        const device = this.options.device ?? "cpu";
        const sessionOptions = device === "cpu" ? { intraOpNumThreads: Math.max(1, Math.floor(this.options.threads ?? os.availableParallelism())) } : undefined;
        const [tokenizer, model] = await Promise.all([
          transformers.AutoTokenizer.from_pretrained(this.model),
          transformers.AutoModelForSequenceClassification.from_pretrained(this.model, { device, dtype: this.options.dtype ?? "q8", session_options: sessionOptions }),
        ]);
        return { tokenizer, model };
      });
      this.loading = loading;
      void loading.catch(() => {
        if (this.loading === loading) this.loading = undefined;
      });
    }
    return this.loading;
  }

  async warmUp(): Promise<void> {
    await this.load();
  }

  async score(query: string, documents: string[]): Promise<number[]> {
    if (!documents.length) return [];
    const { tokenizer, model } = await this.load();
    const inputs = tokenizer(new Array<string>(documents.length).fill(query), {
      text_pair: documents,
      padding: true,
      truncation: true,
      max_length: this.options.maxLength ?? 512,
    });
    const { logits } = await model(inputs) as { logits: { data: ArrayLike<number>; dims: number[] } };
    const columns = logits.dims.at(-1) ?? 1;
    // Single-logit cross-encoders emit one relevance score per pair; two-class heads put "relevant" last.
    return Array.from({ length: documents.length }, (_, index) => Number(logits.data[index * columns + columns - 1]));
  }
}

export function rerankPassage(result: { title: string; heading: string | null; content: string }): string {
  return [result.title, result.heading, result.content].filter(Boolean).join("\n");
}
