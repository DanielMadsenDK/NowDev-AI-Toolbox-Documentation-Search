import { execFile } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

const OWNER = "ServiceNow";
const REPOSITORY = "ServiceNowDocs";
const DEFAULT_REPOSITORY_URL = `https://github.com/${OWNER}/${REPOSITORY}.git`;
const execFileAsync = promisify(execFile);

export type DocumentationArea = "all-docs" | "scripting" | "server" | "client" | "scripts";

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
}

function included(sourcePath: string, area: DocumentationArea): boolean {
  if (!sourcePath.startsWith("markdown/") || !sourcePath.endsWith(".md") || path.posix.basename(sourcePath) === "index.md") return false;
  if (area === "all-docs") return true;
  const topLevelApi = /^markdown\/api-reference\/[^/]+\.md$/.test(sourcePath);
  if (area === "server") return sourcePath.startsWith("markdown/api-reference/server-api-reference/");
  if (area === "client") return topLevelApi || sourcePath.startsWith("markdown/api-reference/cllent-mobile-api-reference/");
  if (area === "scripts") return sourcePath.startsWith("markdown/api-reference/scripts/");
  return topLevelApi || [
    "markdown/api-reference/server-api-reference/",
    "markdown/api-reference/cllent-mobile-api-reference/",
    "markdown/api-reference/scripts/",
  ].some((prefix) => sourcePath.startsWith(prefix));
}

function errorDetail(error: unknown): string {
  if (!(error instanceof Error)) return String(error);
  const cause = error.cause instanceof Error ? `: ${error.cause.message}` : "";
  return `${error.message}${cause}`;
}

function retryDelay(response: Response | undefined, attempt: number, baseMilliseconds: number): number {
  const retryAfter = response?.headers.get("retry-after");
  if (retryAfter && /^\d+$/.test(retryAfter)) return Number(retryAfter) * 1_000;
  return Math.min(baseMilliseconds * 2 ** (attempt - 1), 8_000);
}

export class GitHubDocumentationSource {
  private readonly maxAttempts: number;
  private readonly retryBaseMilliseconds: number;
  private readonly timeoutMilliseconds: number;
  private readonly onRetry: (message: string) => void;
  private readonly repositoryDirectory?: string;
  private readonly repositoryUrl: string;
  private activeRepository?: string;

  constructor(options: GitHubDocumentationSourceOptions = {}) {
    this.maxAttempts = Math.max(1, options.maxAttempts ?? 5);
    this.retryBaseMilliseconds = Math.max(0, options.retryBaseMilliseconds ?? 500);
    this.timeoutMilliseconds = Math.max(1, options.timeoutMilliseconds ?? 30_000);
    this.onRetry = options.onRetry ?? ((message) => console.error(message));
    this.repositoryDirectory = options.repositoryDirectory;
    this.repositoryUrl = options.repositoryUrl ?? DEFAULT_REPOSITORY_URL;
  }

  private async git(args: string[], cwd?: string): Promise<string> {
    const result = await execFileAsync("git", args, { cwd, maxBuffer: 50 * 1024 * 1024 });
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
        await this.git(["clone", "--depth", "1", "--single-branch", "--no-tags", "--branch", branch, this.repositoryUrl, temporaryDirectory]);
        await fs.rm(repositoryDirectory, { recursive: true, force: true });
        await fs.rename(temporaryDirectory, repositoryDirectory);
      } finally {
        await fs.rm(temporaryDirectory, { recursive: true, force: true });
      }
    }
    this.activeRepository = repositoryDirectory;
  }

  private async discoverFromRepository(area: DocumentationArea): Promise<SourceTree> {
    const output = await this.git(["ls-tree", "-r", "HEAD"], this.activeRepository);
    const entries = output.split("\n").flatMap((line) => {
      const match = line.match(/^\d+\s+blob\s+([0-9a-f]+)\t(.+)$/);
      if (!match || !included(match[2]!, area)) return [];
      return [{ path: match[2]!, blobSha: match[1]! }];
    }).sort((left, right) => left.path.localeCompare(right.path));
    const commit = (await this.git(["rev-parse", "HEAD"], this.activeRepository)).trim();
    return { commit, entries };
  }

  private async checkedFetch(url: string, label: string): Promise<Response> {
    let lastFailure = "unknown failure";
    let attemptsUsed = 0;
    for (let attempt = 1; attempt <= this.maxAttempts; attempt += 1) {
      attemptsUsed = attempt;
      let response: Response | undefined;
      try {
        response = await fetch(url, {
          headers: {
            Accept: "application/vnd.github+json",
            "User-Agent": "documentationsearch",
            ...(process.env.GITHUB_TOKEN ? { Authorization: `Bearer ${process.env.GITHUB_TOKEN}` } : {}),
          },
          signal: AbortSignal.timeout(this.timeoutMilliseconds),
        });
        if (response.ok) return response;
        lastFailure = `${response.status} ${response.statusText}`;
        if (response.status !== 408 && response.status !== 429 && response.status < 500) break;
      } catch (error) {
        lastFailure = errorDetail(error);
      }
      if (attempt < this.maxAttempts) {
        const delay = retryDelay(response, attempt, this.retryBaseMilliseconds);
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
    const response = await this.checkedFetch(`https://api.github.com/repos/${OWNER}/${REPOSITORY}/git/trees/${encodeURIComponent(branch)}?recursive=1`, `branch '${branch}'`);
    const data = await response.json() as GitTreeResponse;
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
      return this.git(["show", `HEAD:${entry.path}`], this.activeRepository);
    }
    const url = `https://raw.githubusercontent.com/${OWNER}/${REPOSITORY}/${encodeURIComponent(branch)}/${entry.path.split("/").map(encodeURIComponent).join("/")}`;
    return (await this.checkedFetch(url, entry.path)).text();
  }
}