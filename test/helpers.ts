import fs from "node:fs";

// libsql releases its Windows file handle asynchronously after close(), so an immediate rmSync can hit EPERM.
// Cleanup is best-effort: a leftover temp directory doesn't affect test correctness, only disk usage.
export async function removeDirectory(directory: string): Promise<void> {
  for (let attempt = 1; attempt <= 10; attempt++) {
    try {
      fs.rmSync(directory, { recursive: true, force: true });
      return;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "EPERM") throw error;
      await new Promise((resolve) => setTimeout(resolve, Math.min(100 * attempt, 300)));
    }
  }
}
