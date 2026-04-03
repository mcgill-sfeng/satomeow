const vscode = require("vscode");
const path = require("path");
const fs = require("fs");
const cp = require("child_process");

function activate(context) {
    const diagnostics = vscode.languages.createDiagnosticCollection("agent");
    context.subscriptions.push(diagnostics);

    const pending = new Map();

    const scheduleValidation = (document, delayMs = 200) => {
        if (!shouldValidate(document)) {
            return;
        }

        const key = document.uri.toString();
        const existing = pending.get(key);
        if (existing) {
            clearTimeout(existing);
        }

        const timer = setTimeout(() => {
            pending.delete(key);
            validateDocument(document, diagnostics);
        }, delayMs);

        pending.set(key, timer);
    };

    context.subscriptions.push(
        vscode.workspace.onDidOpenTextDocument((document) =>
            scheduleValidation(document, 0),
        ),
        vscode.workspace.onDidChangeTextDocument((event) =>
            scheduleValidation(event.document, 250),
        ),
        vscode.workspace.onDidSaveTextDocument((document) =>
            scheduleValidation(document, 0),
        ),
        vscode.workspace.onDidCloseTextDocument((document) =>
            diagnostics.delete(document.uri),
        ),
    );

    for (const document of vscode.workspace.textDocuments) {
        scheduleValidation(document, 0);
    }
}

function shouldValidate(document) {
    return document.languageId === "agent" && document.uri.scheme === "file";
}

async function validateDocument(document, diagnosticsCollection) {
    if (!shouldValidate(document)) {
        return;
    }

    const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);
    if (!workspaceFolder) {
        return;
    }

    const workspaceRoot = workspaceFolder.uri.fsPath;
    const pythonExe = resolvePython(workspaceRoot);
    const args = [
        "-m",
        "agent.lsp_validate",
        "--stdin",
        "--path",
        document.fileName,
    ];

    let stdout = "";
    let stderr = "";

    await new Promise((resolve) => {
        const child = cp.spawn(pythonExe, args, {
            cwd: workspaceRoot,
            shell: false,
            stdio: ["pipe", "pipe", "pipe"],
        });

        child.stdout.on("data", (chunk) => {
            stdout += String(chunk);
        });

        child.stderr.on("data", (chunk) => {
            stderr += String(chunk);
        });

        child.on("error", (error) => {
            stderr += String(error);
            resolve();
        });

        child.on("close", () => {
            resolve();
        });

        child.stdin.write(document.getText());
        child.stdin.end();
    });

    const parsed = safeParseJson(stdout);
    if (!parsed || !Array.isArray(parsed.diagnostics)) {
        const fallback = new vscode.Diagnostic(
            new vscode.Range(0, 0, 0, 1),
            stderr || "Agent validator did not return valid diagnostics output.",
            vscode.DiagnosticSeverity.Error,
        );
        fallback.source = "agent";
        diagnosticsCollection.set(document.uri, [fallback]);
        return;
    }

    const vscodeDiagnostics = parsed.diagnostics.map(toVscodeDiagnostic);
    diagnosticsCollection.set(document.uri, vscodeDiagnostics);
}

function toVscodeDiagnostic(item) {
    const line = toZeroBased(item.line, 0);
    const col = toZeroBased(item.col, 0);
    const endLine = toZeroBased(item.endLine || item.line, line);
    const endCol = toZeroBased(item.endCol || item.col + 1, col + 1);

    const range = new vscode.Range(line, col, endLine, Math.max(endCol, col + 1));
    const severity =
        (item.severity || "").toLowerCase() === "warning"
            ? vscode.DiagnosticSeverity.Warning
            : vscode.DiagnosticSeverity.Error;

    const diagnostic = new vscode.Diagnostic(
        range,
        item.message || "Validation error",
        severity,
    );
    diagnostic.source = "agent";
    return diagnostic;
}

function toZeroBased(value, fallback) {
    if (typeof value !== "number" || Number.isNaN(value)) {
        return fallback;
    }
    return Math.max(0, value - 1);
}

function safeParseJson(raw) {
    try {
        return JSON.parse(raw);
    } catch {
        return null;
    }
}

function resolvePython(workspaceRoot) {
    const candidates = [
        path.join(workspaceRoot, ".venv", "Scripts", "python.exe"), // Windows
        path.join(workspaceRoot, ".venv", "bin", "python"), // macOS/Linux
        path.join(workspaceRoot, ".venv", "bin", "python3"),
    ];

    for (const candidate of candidates) {
        if (fs.existsSync(candidate)) {
            return candidate;
        }
    }

    return process.platform === "win32" ? "python" : "python3";
}

function deactivate() {
}

module.exports = {
    activate,
    deactivate,
};
