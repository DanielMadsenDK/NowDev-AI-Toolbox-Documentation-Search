import fs from "node:fs";
import { fileURLToPath } from "node:url";
import { Worker } from "node:worker_threads";
import type { EmbeddingWorkerResponse } from "./embedding-inference.js";

export interface EmbeddingWorkerPoolOptions {
  size: number;
  workerUrl: URL;
  workerData: unknown;
  /** Workers are terminated after this long without work, releasing their model copies. */
  idleMilliseconds?: number;
}

interface Job {
  texts: string[];
  resolve: (data: Float32Array) => void;
  reject: (error: Error) => void;
}

interface PoolWorker {
  worker: Worker;
  job?: Job & { id: number };
}

const DEFAULT_IDLE_MILLISECONDS = 10_000;

function workerError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

/**
 * A fixed set of embedding workers, each holding its own model session. Jobs are handed to idle workers one at a
 * time. Idle workers are unref'd so they never keep the process alive, and the whole pool shuts down after a quiet
 * period; the next job starts it again.
 */
export class EmbeddingWorkerPool {
  readonly size: number;
  private readonly options: EmbeddingWorkerPoolOptions;
  private workers: PoolWorker[] = [];
  private starting?: Promise<void>;
  private readonly queue: Job[] = [];
  private nextJobId = 0;
  private idleTimer?: NodeJS.Timeout;

  constructor(options: EmbeddingWorkerPoolOptions) {
    this.options = options;
    this.size = Math.max(1, Math.floor(options.size));
  }

  /** Whether the worker script exists; a source checkout run without a build has none. */
  get available(): boolean {
    return this.options.workerUrl.protocol !== "file:" || fs.existsSync(fileURLToPath(this.options.workerUrl));
  }

  run(texts: string[]): Promise<Float32Array> {
    return new Promise((resolve, reject) => {
      this.queue.push({ texts, resolve, reject });
      if (this.workers.length) {
        this.dispatch();
        return;
      }
      // Also restarts a pool that shut down while idle.
      this.start().then(() => this.dispatch(), (error: unknown) => {
        for (const job of this.queue.splice(0)) job.reject(workerError(error));
      });
    });
  }

  /** Starts every worker, rejecting if any fails to load its model. */
  start(): Promise<void> {
    this.starting ??= this.spawnAll().catch((error: unknown) => {
      this.starting = undefined;
      throw error;
    });
    return this.starting;
  }

  async dispose(): Promise<void> {
    clearTimeout(this.idleTimer);
    const workers = this.workers;
    this.workers = [];
    this.starting = undefined;
    const disposed = new Error("Embedding worker pool was disposed");
    for (const job of this.queue.splice(0)) job.reject(disposed);
    for (const entry of workers) entry.job?.reject(disposed);
    await Promise.all(workers.map((entry) => entry.worker.terminate()));
  }

  private async spawnAll(): Promise<void> {
    // The first worker downloads the model into the shared cache if it is missing; starting the rest only after it
    // is ready keeps several workers from writing the same files at once.
    const first = await this.spawn();
    const rest = await Promise.allSettled(Array.from({ length: this.size - 1 }, () => this.spawn()));
    const started = [first, ...rest.flatMap((result) => result.status === "fulfilled" ? [result.value] : [])];
    const failure = rest.find((result): result is PromiseRejectedResult => result.status === "rejected");
    if (failure) {
      await Promise.all(started.map((worker) => worker.terminate()));
      throw workerError(failure.reason);
    }
    this.workers = started.map((worker) => this.attach(worker));
  }

  private spawn(): Promise<Worker> {
    return new Promise((resolve, reject) => {
      // The worker needs no Node flags, and inheriting the host's (such as --input-type) can keep it from starting.
      const worker = new Worker(this.options.workerUrl, { workerData: this.options.workerData, execArgv: [] });
      const onError = (error: unknown) => {
        worker.off("exit", onExit);
        void worker.terminate();
        reject(workerError(error));
      };
      const onExit = (code: number) => {
        worker.off("error", onError);
        reject(new Error(`Embedding worker exited with code ${code} before it was ready`));
      };
      worker.once("error", onError);
      worker.once("exit", onExit);
      worker.once("message", () => {
        worker.off("error", onError);
        worker.off("exit", onExit);
        worker.unref();
        resolve(worker);
      });
    });
  }

  private attach(worker: Worker): PoolWorker {
    const entry: PoolWorker = { worker };
    worker.on("message", (message: EmbeddingWorkerResponse) => {
      const job = entry.job;
      if (!job || !("id" in message) || message.id !== job.id) return;
      entry.job = undefined;
      worker.unref();
      if ("error" in message) job.reject(new Error(message.error));
      else job.resolve(message.data);
      this.dispatch();
    });
    const fail = (error: Error) => {
      if (!this.workers.includes(entry)) return;
      this.workers = this.workers.filter((candidate) => candidate !== entry);
      entry.job?.reject(error);
      entry.job = undefined;
      if (!this.workers.length) {
        // Every worker is gone; fail the backlog and let the next run() start a fresh pool.
        this.starting = undefined;
        for (const job of this.queue.splice(0)) job.reject(error);
      }
      this.dispatch();
    };
    worker.on("error", (error) => fail(workerError(error)));
    worker.on("exit", (code) => fail(new Error(`Embedding worker exited unexpectedly with code ${code}`)));
    return entry;
  }

  private dispatch(): void {
    clearTimeout(this.idleTimer);
    for (const entry of this.workers) {
      if (entry.job) continue;
      const job = this.queue.shift();
      if (!job) break;
      entry.job = { ...job, id: this.nextJobId++ };
      // A worker with work in flight keeps the process alive so the awaiting caller is not abandoned.
      entry.worker.ref();
      entry.worker.postMessage({ id: entry.job.id, texts: job.texts });
    }
    if (this.workers.length && !this.queue.length && this.workers.every((entry) => !entry.job)) {
      this.idleTimer = setTimeout(() => void this.dispose(), this.options.idleMilliseconds ?? DEFAULT_IDLE_MILLISECONDS);
      this.idleTimer.unref();
    }
  }
}
