import Darwin
import Foundation

struct ProcessModelConverter: ModelConverting, Sendable {
    let runtime: WorkerRuntime

    func convert(
        checkpoint: URL,
        destination: URL,
        progress: @escaping @Sendable (ModelConversionPhase) -> Void
    ) async throws {
        guard checkpoint.isFileURL,
              destination.isFileURL,
              checkpoint.path.hasPrefix("/"),
              destination.path.hasPrefix("/") else {
            throw ModelInstallerError.unsafeInstallPath
        }
        try Task.checkCancellation()

        let box = ModelConverterProcessBox()
        let standardOutput = Pipe()
        let standardError = Pipe()
        box.process.executableURL = runtime.modelExecutable
        box.process.arguments = [
            "prepare",
            "--checkpoint", checkpoint.standardizedFileURL.path,
            "--destination", destination.standardizedFileURL.path,
        ]
        box.process.environment = Self.environment()
        box.process.standardInput = FileHandle.nullDevice
        box.process.standardOutput = standardOutput
        box.process.standardError = standardError
        box.process.currentDirectoryURL = runtime.root

        do { try box.process.run() }
        catch { throw ModelInstallerError.converterFailed("The model converter could not start.") }

        do {
            try box.adoptProcessGroup()
        } catch {
            Darwin.kill(box.process.processIdentifier, SIGKILL)
            box.process.waitUntilExit()
            throw error
        }

        let outputReader = Task.detached(priority: .utility) {
            Self.drainProgress(
                standardOutput.fileHandleForReading,
                progress: progress
            )
        }
        let diagnosticReader = Task.detached(priority: .utility) {
            Self.drainDiagnostics(standardError.fileHandleForReading)
        }
        let processWaiter = Task.detached(priority: .utility) {
            box.process.waitUntilExit()
            box.markProcessExited()
            return box.process.terminationStatus
        }

        let result = await withTaskCancellationHandler {
            let status = await processWaiter.value
            await outputReader.value
            let diagnostics = await diagnosticReader.value
            return (status, diagnostics)
        } onCancel: {
            box.requestCancellation()
        }
        if Task.isCancelled {
            await box.waitForCancellationEscalation()
            throw CancellationError()
        }
        guard result.0 == 0 else {
            let message = Self.safeDiagnostic(result.1)
            throw ModelInstallerError.converterFailed(
                message.isEmpty ? "Model conversion failed." : message
            )
        }
    }

    private static func environment() -> [String: String] {
        [
            "HOME": FileManager.default.homeDirectoryForCurrentUser.path,
            "TMPDIR": NSTemporaryDirectory(),
            "PATH": "/usr/bin:/bin",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "LC_ALL": "C",
            "LANG": "C",
        ]
    }

    private static func drainProgress(
        _ handle: FileHandle,
        progress: @escaping @Sendable (ModelConversionPhase) -> Void
    ) {
        defer { try? handle.close() }
        var line = Data()
        var discardingOversizedLine = false
        while true {
            let chunk = handle.readData(ofLength: 32 * 1_024)
            if chunk.isEmpty { break }
            for byte in chunk {
                if byte == 0x0A {
                    if !discardingOversizedLine { parseProgress(line, progress: progress) }
                    line.removeAll(keepingCapacity: true)
                    discardingOversizedLine = false
                } else if !discardingOversizedLine {
                    if line.count < 4_096 {
                        line.append(byte)
                    } else {
                        discardingOversizedLine = true
                        line.removeAll(keepingCapacity: true)
                    }
                }
            }
        }
        if !line.isEmpty, !discardingOversizedLine {
            parseProgress(line, progress: progress)
        }
    }

    private static func parseProgress(
        _ line: Data,
        progress: @escaping @Sendable (ModelConversionPhase) -> Void
    ) {
        guard let object = try? JSONSerialization.jsonObject(with: line),
              let value = object as? [String: Any],
              let rawPhase = value["phase"] as? String,
              let phase = ModelConversionPhase(rawValue: rawPhase) else {
            return
        }
        progress(phase)
    }

    private static func drainDiagnostics(_ handle: FileHandle) -> Data {
        defer { try? handle.close() }
        var retained = Data()
        while true {
            let chunk = handle.readData(ofLength: 32 * 1_024)
            if chunk.isEmpty { break }
            if retained.count < 4_096 {
                retained.append(chunk.prefix(4_096 - retained.count))
            }
        }
        return retained
    }

    private static func safeDiagnostic(_ data: Data) -> String {
        String(decoding: data.prefix(4_096), as: UTF8.self)
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

private final class ModelConverterProcessBox: @unchecked Sendable {
    private struct State {
        var processGroup: pid_t?
        var cancellationRequested = false
        var escalationFinished = true
        var escalationWaiters: [CheckedContinuation<Void, Never>] = []
    }

    let process = Process()
    private let lock = NSLock()
    private var state = State()

    func adoptProcessGroup() throws {
        let processID = process.processIdentifier
        guard processID > 0 else {
            throw ModelInstallerError.converterFailed("The model converter did not start safely.")
        }
        if Darwin.getpgid(processID) != processID {
            _ = Darwin.setpgid(processID, processID)
        }
        guard Darwin.getpgid(processID) == processID else {
            throw ModelInstallerError.converterFailed(
                "The model converter could not be isolated for safe cancellation."
            )
        }
        lock.withLock { state.processGroup = processID }
    }

    func requestCancellation() {
        let processGroup = lock.withLock { () -> pid_t? in
            guard let processGroup = state.processGroup,
                  !state.cancellationRequested else { return nil }
            state.cancellationRequested = true
            state.escalationFinished = false
            return processGroup
        }
        guard let processGroup else { return }
        _ = Darwin.kill(-processGroup, SIGTERM)
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + .milliseconds(500)) {
            _ = Darwin.kill(-processGroup, SIGKILL)
            self.finishEscalation(for: processGroup)
        }
    }

    func markProcessExited() {
        // The process-group identifier remains reserved here until escalation.
        // Children may outlive the group leader and still own inherited pipes.
    }

    func waitForCancellationEscalation() async {
        await withCheckedContinuation { continuation in
            let finishImmediately = lock.withLock {
                if !state.cancellationRequested || state.escalationFinished { return true }
                state.escalationWaiters.append(continuation)
                return false
            }
            if finishImmediately { continuation.resume() }
        }
    }

    private func finishEscalation(for processGroup: pid_t) {
        let waiters = lock.withLock { () -> [CheckedContinuation<Void, Never>] in
            guard state.processGroup == processGroup else { return [] }
            state.processGroup = nil
            state.escalationFinished = true
            let waiters = state.escalationWaiters
            state.escalationWaiters.removeAll()
            return waiters
        }
        for waiter in waiters { waiter.resume() }
    }
}
