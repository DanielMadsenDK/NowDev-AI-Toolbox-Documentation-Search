import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { GitHubDocumentationSource } from "../src/github.js";

const temporaryDirectories: string[] = [];

afterEach(() => {
  vi.unstubAllGlobals();
  temporaryDirectories.splice(0).forEach((directory) => fs.rmSync(directory, { recursive: true, force: true }));
});

function git(cwd: string, ...args: string[]): string {
  return execFileSync("git", args, { cwd, encoding: "utf8" }).trim();
}

describe("GitHubDocumentationSource", () => {
  it("retries transient network failures", async () => {
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new TypeError("fetch failed", { cause: new Error("connection reset") }))
      .mockResolvedValueOnce(new Response("documentation", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const retries: string[] = [];
    const source = new GitHubDocumentationSource({ maxAttempts: 3, retryBaseMilliseconds: 0, onRetry: (message) => retries.push(message) });
    const content = await source.download("australia", { path: "markdown/test.md", blobSha: "sha" });
    expect(content).toBe("documentation");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(retries[0]).toContain("connection reset");
  });

  it("does not retry permanent client errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("missing", { status: 404, statusText: "Not Found" })));
    const source = new GitHubDocumentationSource({ maxAttempts: 3, retryBaseMilliseconds: 0, onRetry: () => undefined });
    await expect(source.download("australia", { path: "markdown/missing.md", blobSha: "sha" }))
      .rejects.toThrow("markdown/missing.md after 1 attempt (404 Not Found)");
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("clones a branch once and reads changed files from shallow fetches", async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-git-"));
    temporaryDirectories.push(root);
    const origin = path.join(root, "origin");
    const checkout = path.join(root, "checkout");
    fs.mkdirSync(path.join(origin, "markdown", "api-reference", "scripts"), { recursive: true });
    git(origin, "init", "--initial-branch", "australia");
    git(origin, "config", "user.email", "test@example.invalid");
    git(origin, "config", "user.name", "ServiceContext Test");
    const sourcePath = "markdown/api-reference/scripts/example.md";
    fs.writeFileSync(path.join(origin, sourcePath), "---\ntitle: First\n---\n# First");
    git(origin, "add", ".");
    git(origin, "commit", "-m", "first");

    const source = new GitHubDocumentationSource({ repositoryDirectory: checkout, repositoryUrl: `file://${origin}` });
    const first = await source.discover("australia", "scripting");
    expect(first.entries).toHaveLength(1);
    expect(await source.download("australia", first.entries[0]!)).toContain("# First");

    fs.writeFileSync(path.join(origin, sourcePath), "---\ntitle: Second\n---\n# Second");
    git(origin, "add", ".");
    git(origin, "commit", "-m", "second");
    const second = await source.discover("australia", "scripting");
    expect(second.entries[0]?.blobSha).not.toBe(first.entries[0]?.blobSha);
    expect(await source.download("australia", second.entries[0]!)).toContain("# Second");
  });
});