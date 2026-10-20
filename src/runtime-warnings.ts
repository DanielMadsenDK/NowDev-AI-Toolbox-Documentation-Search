/**
 * Hides Node's experimental-feature warning for node:sqlite, which every command would otherwise print, while leaving
 * every other warning, and the --no-warnings / NODE_NO_WARNINGS opt-outs, to Node's default handling.
 */
export function suppressExperimentalSqliteWarning(): void {
  const emitWarning = process.emitWarning.bind(process);
  process.emitWarning = ((warning: string | Error, ...rest: unknown[]) => {
    const type = typeof rest[0] === "string" ? rest[0] : (rest[0] as { type?: string } | undefined)?.type;
    const name = warning instanceof Error ? warning.name : type;
    const message = warning instanceof Error ? warning.message : warning;
    if (name === "ExperimentalWarning" && message.includes("SQLite")) return;
    (emitWarning as (...args: unknown[]) => void)(warning, ...rest);
  }) as typeof process.emitWarning;
}
