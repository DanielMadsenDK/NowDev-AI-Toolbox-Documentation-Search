import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { GitHubDocumentationSource } from "../src/github.js";

const temporaryDirectories: string[] = [];

afterEach(() => {
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

  it("retries an exhausted rate limit after the reset GitHub reports", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response("rate limited", { status: 403, statusText: "Forbidden", headers: { "x-ratelimit-remaining": "0", "x-ratelimit-reset": String(Math.floor(Date.now() / 1_000)) } }))
      .mockResolvedValueOnce(new Response("documentation", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const retries: string[] = [];
    const source = new GitHubDocumentationSource({ maxAttempts: 3, retryBaseMilliseconds: 0, onRetry: (message) => retries.push(message) });
    expect(await source.download("australia", { path: "markdown/test.md", blobSha: "sha" })).toBe("documentation");
    expect(retries[0]).toContain("rate limit exceeded");
  });

  it("honours an HTTP-date Retry-After and gives up on one that is too far away", async () => {
    const soon = new Date(Date.now() + 50).toUTCString();
    vi.stubGlobal("fetch", vi.fn()
      .mockResolvedValueOnce(new Response("busy", { status: 503, statusText: "Service Unavailable", headers: { "retry-after": soon } }))
      .mockResolvedValueOnce(new Response("documentation", { status: 200 })));
    const source = new GitHubDocumentationSource({ maxAttempts: 3, retryBaseMilliseconds: 0, onRetry: () => undefined });
    expect(await source.download("australia", { path: "markdown/test.md", blobSha: "sha" })).toBe("documentation");

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("busy", { status: 503, statusText: "Service Unavailable", headers: { "retry-after": "3600" } })));
    await expect(source.download("australia", { path: "markdown/test.md", blobSha: "sha" })).rejects.toThrow("asked to wait 3600s");
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("retries a failure while reading the response body", async () => {
    const failingBody = new ReadableStream({ start: (controller) => controller.error(new Error("connection reset while reading")) });
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(failingBody, { status: 200 }))
      .mockResolvedValueOnce(new Response("documentation", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const source = new GitHubDocumentationSource({ maxAttempts: 3, retryBaseMilliseconds: 0, onRetry: () => undefined });
    expect(await source.download("australia", { path: "markdown/test.md", blobSha: "sha" })).toBe("documentation");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("rejects clone paths that would leave the repository", async () => {
    const source = new GitHubDocumentationSource({ repositoryDirectory: "/unused" });
    (source as unknown as { activeRepository: string }).activeRepository = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-traversal-"));
    temporaryDirectories.push((source as unknown as { activeRepository: string }).activeRepository);
    await expect(source.download("australia", { path: "markdown/../../outside.md", blobSha: "sha" })).rejects.toThrow("Invalid documentation path");
  });

  it("discovers clone paths with non-ASCII characters verbatim", async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "servicecontext-git-unicode-"));
    temporaryDirectories.push(root);
    const origin = path.join(root, "origin");
    fs.mkdirSync(path.join(origin, "markdown", "api-reference", "scripts"), { recursive: true });
    git(origin, "init", "--initial-branch", "australia");
    git(origin, "config", "user.email", "test@example.invalid");
    git(origin, "config", "user.name", "ServiceContext Test");
    const sourcePath = "markdown/api-reference/scripts/r\u00e9sum\u00e9.md";
    fs.writeFileSync(path.join(origin, sourcePath), "# R\u00e9sum\u00e9");
    git(origin, "add", ".");
    git(origin, "commit", "-m", "unicode");
    const source = new GitHubDocumentationSource({ repositoryDirectory: path.join(root, "checkout"), repositoryUrl: `file://${origin}`, onRetry: () => undefined });
    const tree = await source.discover("australia", "scripting");
    expect(tree.entries.map((entry) => entry.path.normalize("NFC"))).toEqual([sourcePath]);
    expect(await source.download("australia", tree.entries[0]!)).toContain("R\u00e9sum\u00e9");
  }, 20_000);

  it("clones a branch once and reads changed files from shallow fetches", async () => {
    // Two local git clone/commit round trips are noticeably slower on Windows filesystems than the 5s default.
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
  }, 20_000);
});