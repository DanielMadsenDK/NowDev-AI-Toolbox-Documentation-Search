#!/usr/bin/env node
import { suppressExperimentalSqliteWarning } from "./runtime-warnings.js";

suppressExperimentalSqliteWarning();
await import("./mcp.js");
