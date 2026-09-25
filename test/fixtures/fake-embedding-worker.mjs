// Deterministic stand-in for src/embedding-worker.ts. Row vectors are [text length, 1, 0, ...]. Texts steer failures:
// "fail" posts an error, "oom" reports out-of-memory while batched with other texts, and "crash" ends the worker.
// workerData.counters (a SharedArrayBuffer of three int32s) records [active jobs, peak active jobs, total jobs].
import { parentPort, workerData } from "node:worker_threads";

const dimensions = workerData.dimensions ?? 4;
const counters = workerData.counters ? new Int32Array(workerData.counters) : undefined;
if (workerData.failStartup) throw new Error("simulated startup failure");

parentPort.on("message", async ({ id, texts }) => {
  if (counters) {
    const active = Atomics.add(counters, 0, 1) + 1;
    Atomics.add(counters, 2, 1);
    for (let peak = Atomics.load(counters, 1); active > peak;) {
      const previous = Atomics.compareExchange(counters, 1, peak, active);
      if (previous === peak) break;
      peak = previous;
    }
  }
  await new Promise((resolve) => setTimeout(resolve, workerData.delayMilliseconds ?? 0));
  if (counters) Atomics.sub(counters, 0, 1);
  if (texts.includes("crash")) process.exit(3);
  if (texts.includes("fail")) {
    parentPort.postMessage({ id, error: "simulated failure" });
    return;
  }
  if (texts.length > 1 && texts.includes("oom")) {
    parentPort.postMessage({ id, error: "Failed to allocate memory: out of memory" });
    return;
  }
  const data = new Float32Array(texts.length * dimensions);
  texts.forEach((text, row) => {
    data[row * dimensions] = text.length;
    data[row * dimensions + 1] = 1;
  });
  parentPort.postMessage({ id, data }, [data.buffer]);
});
parentPort.postMessage({ ready: true });
