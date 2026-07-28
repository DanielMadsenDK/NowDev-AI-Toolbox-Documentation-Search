export function suppressExperimentalSqliteWarning(): void {
  process.removeAllListeners("warning");
  process.on("warning", (warning) => {
    if (warning.name === "ExperimentalWarning" && warning.message.includes("SQLite")) return;
    console.warn(warning.stack ?? `${warning.name}: ${warning.message}`);
  });
}
