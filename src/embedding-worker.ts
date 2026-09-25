// Worker-thread entry for EmbeddingWorkerPool. ONNX Runtime's intra-op threading barely scales for small encoders,
// and sessions in one process do not run concurrently, so bulk embedding runs one single-threaded session per worker.
import { parentPort, workerData } from "node:worker_threads";
import { extractEmbeddings, type EmbeddingWorkerData, type EmbeddingWorkerRequest, type EmbeddingWorkerResponse } from "./embedding-inference.js";

const port = parentPort!;
const settings = workerData as EmbeddingWorkerData;
const transformers = await import("@huggingface/transformers");
if (settings.cacheDirectory) transformers.env.cacheDir = settings.cacheDirectory;
const extractor = await transformers.pipeline("feature-extraction", settings.model, {
  device: "cpu",
  dtype: settings.dtype as "q8",
  session_options: { intraOpNumThreads: settings.threads },
});

port.on("message", async ({ id, texts }: EmbeddingWorkerRequest) => {
  let response: EmbeddingWorkerResponse;
  try {
    response = { id, data: await extractEmbeddings(extractor, transformers.layer_norm, texts, settings) };
  } catch (error) {
    port.postMessage({ id, error: error instanceof Error ? error.message : String(error) } satisfies EmbeddingWorkerResponse);
    return;
  }
  port.postMessage(response, [response.data.buffer as ArrayBuffer]);
});
port.postMessage({ ready: true } satisfies EmbeddingWorkerResponse);
