#!/usr/bin/env node
import { suppressExperimentalSqliteWarning } from "./runtime-warnings.js";

suppressExperimentalSqliteWarning();
// mcp.js only starts a server on its own when it is the script Node was launched with, so this entry must start it.
const { startMcpServer } = await import("./mcp.js");
startMcpServer().catch((error: unknown) => {
	console.error(error instanceof Error ? error.message : String(error));
	process.exitCode = 1;
});
