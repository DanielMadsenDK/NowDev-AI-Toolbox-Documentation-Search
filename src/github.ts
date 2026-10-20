import { execFile } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import { API_REFERENCE_PREFIXES, TOP_LEVEL_API_PATH } from "./config.js";
import type { DocumentationArea } from "./types.js";

export type { DocumentationArea } from "./types.js";

const OWNER = "ServiceNow";
const REPOSITORY = "ServiceNowDocs";
const DEFAULT_REPOSITORY_URL = `https://github.com/${OWNER}/${REPOSITORY}.git`;
const execFileAsync = promisify(execFile);
// Longest wait honoured from a Retry-After header or a rate-limit reset; a longer wait fails the request instead.
const MAX_RETRY_DELAY_MILLISECONDS = 60_000;
// Generous enough for the one-time shallow clone of the full repository on a slow connection.
const DEFAULT_GIT_TIMEOUT_MILLISECONDS = 30 * 60_000;

export interface SourceEntry {
  path: string;
  blobSha: string;
}

export interface SourceTree {
  commit: string;
  entries: SourceEntry[];
}

interface GitTreeResponse {
  sha: string;
  truncated?: boolean;
  tree?: Array<{ path?: string; sha?: string; type?: string }>;
}

export interface GitHubDocumentationSourceOptions {
  maxAttempts?: number;
  retryBaseMilliseconds?: number;
  timeoutMilliseconds?: number;
  onRetry?: (message: string) => void;
  repositoryDirectory?: string;
  repositoryUrl?: string;
  /** Kills a git command that runs longer than this (default: 30 minutes). */
  gitTimeoutMilliseconds?: number;
}

function included(sourcePath: string, area: DocumentationArea): boolean {
  if (!sourcePath.startsWith("markdown/") || !sourcePath.endsWith(".md") || path.posix.basename(sourcePath) === "index.md") return false;
  if (area === "all-docs") return true;
  const topLevelApi = TOP_LEVEL_API_PATH.test(sourcePath);
  if (area === "server") return sourcePath.startsWith(API_REFERENCE_PREFIXES.server);
  if (area === "client") return topLevelApi || sourcePath.startsWith(API_REFERENCE_PREFIXES.clientMobile);
  if (area === "scripts") return sourcePath.startsWith(API_REFERENCE_PREFIXES.scripts);
  return topLevelApi || [API_REFERENCE_PREFIXES.server, API_REFERENCE_PREFIXES.clientMobile, API_REFERENCE_PREFIXES.scripts].some((prefix) => sourcePath.startsWith(prefix));
}

function errorDetail(error: unknown): string {
  if (!(error instanceof Error)) return String(error);
  const cause = error.cause instanceof Error ? `: ${error.cause.message}` : "";
  return `${error.message}${cause}`;
}

/** GitHub reports an exhausted primary rate limit as 403 (or 429) with no remaining requests. */
function isRateLimited(response: Response): boolean {
  return (response.status === 403 || response.status === 429) && response.headers.get("x-ratelimit-remaining") === "0";
}

function isRetryable(response: Response): boolean {
  return response.status === 408 || response.status === 429 || response.status >= 500 || isRateLimited(response);
}

/** The wait a response asks for, from Retry-After (seconds or an HTTP date) or a rate-limit reset time. */
function requestedDelay(response: Response | undefined): number | undefined {
  const retryAfter = response?.headers.get("retry-after")?.trim();
  if (retryAfter) {
    if (/^\d+$/.test(retryAfter)) return Number(retryAfter) * 1_000;
    const date = Date.parse(retryAfter);
    if (!Number.isNaN(date)) return Math.max(0, date - Date.now());
  }
  const reset = response && isRateLimited(response) ? response.headers.get("x-ratelimit-reset") : null;
  if (reset && /^\d+$/.test(reset)) return Math.max(0, Number(reset) * 1_000 - Date.now());
  return undefined;
}

function backoffDelay(attempt: number, baseMilliseconds: number): number {
  // Full jitter keeps concurrent downloads that failed together from retrying in lockstep.
  return Math.round(Math.random() * Math.min(baseMilliseconds * 2 ** (attempt - 1), 8_000));
}

/** Resolves a repository-relative path, rejecting one that would leave the repository directory. */
function repositoryFile(repository: string, sourcePath: string): string {
  const normalized = path.posix.normalize(sourcePath);
  if (normalized !== sourcePath || normalized.startsWith("../") || path.posix.isAbsolute(normalized)) {
    throw new Error(`Invalid documentation path: ${sourcePath}`);
  }
  return path.join(repository, ...normalized.split("/"));
}

export class GitHubDocumentationSource {
  private readonly maxAttempts: number;
  private readonly retryBaseMilliseconds: number;
  private readonly timeoutMilliseconds: number;
  private readonly onRetry: (message: string) => void;
  private readonly repositoryDirectory?: string;
  private readonly repositoryUrl: string;
  private readonly gitTimeoutMilliseconds: number;
  private activeRepository?: string;

  constructor(options: GitHubDocumentationSourceOptions = {}) {
    this.maxAttempts = Math.max(1, options.maxAttempts ?? 5);
    this.retryBaseMilliseconds = Math.max(0, options.retryBaseMilliseconds ?? 500);
    this.timeoutMilliseconds = Math.max(1, options.timeoutMilliseconds ?? 30_000);
    this.onRetry = options.onRetry ?? ((message) => console.error(message));
    this.repositoryDirectory = options.repositoryDirectory;
    this.repositoryUrl = options.repositoryUrl ?? DEFAULT_REPOSITORY_URL;
    this.gitTimeoutMilliseconds = Math.max(1, options.gitTimeoutMilliseconds ?? DEFAULT_GIT_TIMEOUT_MILLISECONDS);
  }

  private async git(args: string[], cwd?: string): Promise<string> {
    const result = await execFileAsync("git", args, {
      cwd,
      maxBuffer: 50 * 1024 * 1024,
      timeout: this.gitTimeoutMilliseconds,
      // The repository is public; a credential prompt would only hang a non-interactive run.
      env: { ...process.env, GIT_TERMINAL_PROMPT: "0" },
    });
    return result.stdout;
  }

  private async synchronizeRepository(branch: string): Promise<void> {
    const repositoryDirectory = this.repositoryDirectory!;
    const gitDirectory = path.join(repositoryDirectory, ".git");
    try {
      await fs.access(gitDirectory);
      this.onRetry(`Updating shallow ServiceNowDocs clone for branch '${branch}'...`);
      await this.git(["fetch", "--depth", "1", "--no-tags", "origin", branch], repositoryDirectory);
      await this.git(["checkout", "--detach", "--force", "FETCH_HEAD"], repositoryDirectory);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      const temporaryDirectory = `${repositoryDirectory}.tmp-${process.pid}-${Date.now()}`;
      await fs.mkdir(path.dirname(repositoryDirectory), { recursive: true });
      await fs.rm(temporaryDirectory, { recursive: true, force: true });
      try {
        this.onRetry(`Creating shallow ServiceNowDocs clone for branch '${branch}'. This one-time download may take several minutes...`);
        await this.git(["-c", "core.longpaths=true", "clone", "--depth", "1", "--single-branch", "--no-tags", "--branch", branch, this.repositoryUrl, temporaryDirectory]);
        await fs.rm(repositoryDirectory, { recursive: true, force: true });
        await fs.rename(temporaryDirectory, repositoryDirectory);
      } finally {
        await fs.rm(temporaryDirectory, { recursive: true, force: true });
      }
    }
    this.activeRepository = repositoryDirectory;
  }

  private async discoverFromRepository(area: DocumentationArea): Promise<SourceTree> {
    // -z keeps paths verbatim; without it git quotes and escapes non-ASCII paths.
    const output = await this.git(["ls-tree", "-r", "-z", "HEAD"], this.activeRepository);
    const entries = output.split("\0").flatMap((record) => {
      const match = record.match(/^\d+ blob ([0-9a-f]+)\t(.+)$/s);
      if (!match || !included(match[2]!, area)) return [];
      return [{ path: match[2]!, blobSha: match[1]! }];
    }).sort((left, right) => left.path.localeCompare(right.path));
    const commit = (await this.git(["rev-parse", "HEAD"], this.activeRepository)).trim();
    return { commit, entries };
  }

  /** Fetches a URL and reads its body, retrying transient failures, including ones that happen while reading the body. */
  private async checkedFetch<T>(url: string, label: string, read: (response: Response) => Promise<T>): Promise<T> {
    let lastFailure = "unknown failure";
    let attemptsUsed = 0;
    for (let attempt = 1; attempt <= this.maxAttempts; attempt += 1) {
      attemptsUsed = attempt;
      let response: Response | undefined;
      try {
        response = await fetch(url, {
          headers: {
            Accept: "application/vnd.github+json",
            "User-Agent": "nowdev-ai-toolbox-documentationsearch",
            ...(process.env.GITHUB_TOKEN ? { Authorization: `Bearer ${process.env.GITHUB_TOKEN}` } : {}),
          },
          // Covers reading the body too, so a stalled transfer is retried like a failed request.
          signal: AbortSignal.timeout(this.timeoutMilliseconds),
        });
        if (response.ok) return await read(response);
        lastFailure = `${response.status} ${response.statusText}${isRateLimited(response) ? " (rate limit exceeded; set GITHUB_TOKEN for a higher limit)" : ""}`;
        // An unread body holds its pooled connection open.
        await response.body?.cancel().catch(() => undefined);
        if (!isRetryable(response)) break;
      } catch (error) {
        lastFailure = errorDetail(error);
      }
      if (attempt < this.maxAttempts) {
        const requested = requestedDelay(response);
        if (requested !== undefined && requested > MAX_RETRY_DELAY_MILLISECONDS) {
          lastFailure += `; the server asked to wait ${Math.ceil(requested / 1_000)}s`;
          break;
        }
        const delay = requested ?? backoffDelay(attempt, this.retryBaseMilliseconds);
        this.onRetry(`GitHub request failed for ${label} (${lastFailure}); retrying ${attempt + 1}/${this.maxAttempts} in ${delay}ms...`);
        if (delay) await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
    throw new Error(`GitHub request failed for ${label} after ${attemptsUsed} ${attemptsUsed === 1 ? "attempt" : "attempts"} (${lastFailure})`);
  }

  async discover(branch: string, area: DocumentationArea = "all-docs"): Promise<SourceTree> {
    if (this.repositoryDirectory) {
      try {
        await this.synchronizeRepository(branch);
        return await this.discoverFromRepository(area);
      } catch (error) {
        this.activeRepository = undefined;
        this.onRetry(`Shallow Git synchronization failed (${errorDetail(error)}); falling back to GitHub API downloads...`);
      }
    }
    const data = await this.checkedFetch(`https://api.github.com/repos/${OWNER}/${REPOSITORY}/git/trees/${encodeURIComponent(branch)}?recursive=1`, `branch '${branch}'`, (response) => response.json() as Promise<GitTreeResponse>);
    if (data.truncated) throw new Error(`GitHub returned a truncated tree for branch '${branch}'`);
    return {
      commit: data.sha,
      entries: (data.tree ?? [])
        .filter((item) => item.type === "blob" && item.path && item.sha && included(item.path, area))
        .map((item) => ({ path: item.path!, blobSha: item.sha! }))
        .sort((left, right) => left.path.localeCompare(right.path)),
    };
  }

  async download(branch: string, entry: SourceEntry): Promise<string> {
    if (this.activeRepository) {
      if (!included(entry.path, "all-docs")) throw new Error(`Invalid documentation path: ${entry.path}`);
      return fs.readFile(repositoryFile(this.activeRepository, entry.path), "utf8");
    }
    const url = `https://raw.githubusercontent.com/${OWNER}/${REPOSITORY}/${encodeURIComponent(branch)}/${entry.path.split("/").map(encodeURIComponent).join("/")}`;
    return this.checkedFetch(url, entry.path, (response) => response.text());
  }
}