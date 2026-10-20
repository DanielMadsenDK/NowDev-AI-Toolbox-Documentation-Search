// Worker-thread entry for EmbeddingWorkerPool. ONNX Runtime's intra-op threading barely scales for small encoders,
// and sessions in one process do not run concurrently, so bulk embedding runs one single-threaded session per worker.
import { parentPort, workerData } from "node:worker_threads";
import { extractEmbeddings, loadExtractor, type EmbeddingWorkerData, type EmbeddingWorkerRequest, type EmbeddingWorkerResponse } from "./embedding-inference.js";

const port = parentPort!;
const settings = workerData as EmbeddingWorkerData;
const { extractor, layerNorm } = await loadExtractor(settings.model, { device: "cpu", dtype: settings.dtype, cacheDirectory: settings.cacheDirectory, threads: settings.threads });

port.on("message", async (request: EmbeddingWorkerRequest) => {
  if ("shutdown" in request) {
    // Releases the ONNX Runtime session before the thread ends; terminate() alone leaves native memory to finalizers.
    await extractor.dispose().catch(() => undefined);
    port.close();
    return;
  }
  const { id, texts } = request;
  let response: EmbeddingWorkerResponse;
  try {
    response = { id, data: await extractEmbeddings(extractor, layerNorm, texts, settings) };
  } catch (error) {
    const cause = error instanceof Error && error.cause instanceof Error ? error.cause.message : undefined;
    port.postMessage({ id, error: error instanceof Error ? error.message : String(error), ...(cause ? { cause } : {}) } satisfies EmbeddingWorkerResponse);
    return;
  }
  port.postMessage(response, [response.data.buffer as ArrayBuffer]);
});
port.postMessage({ ready: true } satisfies EmbeddingWorkerResponse);
