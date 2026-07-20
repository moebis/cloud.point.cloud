import Foundation

enum MockEngineClock: Sendable {
    case immediate
}

actor MockReconstructionEngine: ReconstructionEngine {
    private enum Lifecycle {
        case idle
        case prepared
        case acceptingInput
        case inputFinished
        case completed
        case failed
        case cancelled
        case shutdown
    }

    nonisolated private let eventStream: AsyncThrowingStream<EngineEvent, Error>
    private let continuation: AsyncThrowingStream<EngineEvent, Error>.Continuation
    private let clock: MockEngineClock
    private var lifecycle: Lifecycle = .idle
    private var project: ProjectDescriptor?
    private var isPaused = false
    private var pendingFrames: [PersistedFrame] = []

    init(clock: MockEngineClock = .immediate) {
        let stream = AsyncThrowingStream.makeStream(of: EngineEvent.self)
        eventStream = stream.stream
        continuation = stream.continuation
        self.clock = clock
    }

    func prepare(configuration: EngineConfiguration) async throws {
        guard lifecycle == .idle else {
            throw ReconstructionEngineError.invalidLifecycle(operation: "prepare")
        }
        lifecycle = .prepared
        continuation.yield(.ready)
    }

    func begin(project: ProjectDescriptor) async throws {
        guard lifecycle == .prepared else {
            throw ReconstructionEngineError.invalidLifecycle(operation: "begin")
        }
        self.project = project
        lifecycle = .acceptingInput
    }

    func enqueue(_ frame: PersistedFrame) async throws {
        guard lifecycle == .acceptingInput else {
            throw ReconstructionEngineError.invalidLifecycle(operation: "enqueue")
        }
        guard UInt32(exactly: frame.index) != nil else {
            throw ReconstructionEngineError.invalidFrameIndex(frame.index)
        }

        pendingFrames.append(frame)
        try processPendingFramesIfPossible()
    }

    func finishInput() async throws {
        guard lifecycle == .acceptingInput else {
            throw ReconstructionEngineError.invalidLifecycle(operation: "finishInput")
        }
        lifecycle = .inputFinished
        try processPendingFramesIfPossible()
        finishIfInputIsComplete()
    }

    func pause() async throws {
        guard lifecycle == .acceptingInput || lifecycle == .inputFinished else {
            throw ReconstructionEngineError.invalidLifecycle(operation: "pause")
        }
        guard !isPaused else {
            throw ReconstructionEngineError.invalidLifecycle(operation: "pause")
        }
        isPaused = true
        continuation.yield(.paused)
    }

    func resume() async throws {
        guard (lifecycle == .acceptingInput || lifecycle == .inputFinished), isPaused else {
            throw ReconstructionEngineError.invalidLifecycle(operation: "resume")
        }
        isPaused = false
        try processPendingFramesIfPossible()
        finishIfInputIsComplete()
    }

    func cancel() async {
        guard lifecycle != .completed, lifecycle != .failed, lifecycle != .cancelled, lifecycle != .shutdown else { return }
        lifecycle = .cancelled
        pendingFrames.removeAll()
        continuation.yield(.cancelled)
        continuation.finish()
    }

    func shutdown() async {
        guard lifecycle != .shutdown else { return }
        lifecycle = .shutdown
        pendingFrames.removeAll()
        continuation.finish()
    }

    nonisolated func events() -> AsyncThrowingStream<EngineEvent, Error> {
        eventStream
    }

    private func processPendingFramesIfPossible() throws {
        guard !isPaused, let project else { return }

        while !pendingFrames.isEmpty {
            let frame = pendingFrames.removeFirst()
            continuation.yield(.frameStarted(frameIndex: frame.index))

            do {
                let result = try MockCPCWriter.write(frame: frame, beneath: project.packageURL)
                continuation.yield(.frameCompleted(result))
            } catch {
                lifecycle = .failed
                pendingFrames.removeAll()
                continuation.finish(throwing: error)
                throw error
            }
        }
    }

    private func finishIfInputIsComplete() {
        guard lifecycle == .inputFinished, !isPaused, pendingFrames.isEmpty else { return }
        lifecycle = .completed
        continuation.yield(.sessionCompleted)
        continuation.finish()
    }
}

private enum MockCPCWriter {
    private static let planeDimension = 64
    private static let vertexStride = 24

    static func write(frame: PersistedFrame, beneath packageURL: URL) throws -> FrameResult {
        guard let sourceFrame = UInt32(exactly: frame.index) else {
            throw ReconstructionEngineError.invalidFrameIndex(frame.index)
        }
        let pointCount = planeDimension * planeDimension
        let filename = String(format: "frame-%08u.cpc", sourceFrame)
        let relativePath = "Points/\(filename)"
        let chunkURL = try outputURL(packageURL: packageURL, filename: filename)

        var data = Data()
        data.reserveCapacity(32 + (pointCount * vertexStride))
        data.append(contentsOf: "CPC1".utf8)
        data.appendLittleEndian(UInt16(1))
        data.appendLittleEndian(UInt16(vertexStride))
        data.appendLittleEndian(UInt64(pointCount))
        data.appendLittleEndian(sourceFrame)
        data.appendLittleEndian(sourceFrame)
        data.append(contentsOf: repeatElement(0, count: 8))

        let color = color(for: frame.index)
        let z = Float(frame.index) * 0.01
        let halfDimension = Float(planeDimension - 1) / 2

        for row in 0..<planeDimension {
            for column in 0..<planeDimension {
                data.appendLittleEndian((Float(column) - halfDimension) * 0.04)
                data.appendLittleEndian((Float(row) - halfDimension) * 0.04)
                data.appendLittleEndian(z)
                data.append(contentsOf: [color.red, color.green, color.blue, 255])
                data.appendLittleEndian(Float16(1).bitPattern)
                data.appendLittleEndian(UInt16(0))
                data.appendLittleEndian(sourceFrame)
            }
        }

        try data.write(to: chunkURL, options: .atomic)
        return FrameResult(
            frameIndex: frame.index,
            pointChunkPath: relativePath,
            pointCount: pointCount
        )
    }

    private static func color(for frameIndex: Int) -> (red: UInt8, green: UInt8, blue: UInt8) {
        (
            UInt8(truncatingIfNeeded: frameIndex &* 53),
            UInt8(truncatingIfNeeded: frameIndex &* 97),
            UInt8(truncatingIfNeeded: frameIndex &* 193)
        )
    }

    private static func outputURL(packageURL: URL, filename: String) throws -> URL {
        let fileManager = FileManager.default
        let standardizedPackageURL = packageURL.standardizedFileURL
        let resolvedPackageURL = standardizedPackageURL.resolvingSymlinksInPath().standardizedFileURL
        guard standardizedPackageURL.path == resolvedPackageURL.path else {
            throw ReconstructionEngineError.unsafeOutputPath
        }

        var isDirectory = ObjCBool(false)
        guard fileManager.fileExists(atPath: resolvedPackageURL.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw ReconstructionEngineError.unsafeOutputPath
        }

        let pointsURL = resolvedPackageURL.appending(path: "Points")
        guard (try? fileManager.destinationOfSymbolicLink(atPath: pointsURL.path)) == nil else {
            throw ReconstructionEngineError.unsafeOutputPath
        }
        if fileManager.fileExists(atPath: pointsURL.path) {
            var isPointsDirectory = ObjCBool(false)
            guard fileManager.fileExists(atPath: pointsURL.path, isDirectory: &isPointsDirectory), isPointsDirectory.boolValue else {
                throw ReconstructionEngineError.unsafeOutputPath
            }
        } else {
            try fileManager.createDirectory(at: pointsURL, withIntermediateDirectories: true)
        }

        let resolvedPointsURL = pointsURL.resolvingSymlinksInPath().standardizedFileURL
        guard resolvedPointsURL.path == pointsURL.path,
              resolvedPointsURL.deletingLastPathComponent().path == resolvedPackageURL.path else {
            throw ReconstructionEngineError.unsafeOutputPath
        }

        let chunkURL = pointsURL.appending(path: filename)
        guard (try? fileManager.destinationOfSymbolicLink(atPath: chunkURL.path)) == nil else {
            throw ReconstructionEngineError.unsafeOutputPath
        }
        let resolvedChunkURL = chunkURL.resolvingSymlinksInPath().standardizedFileURL
        guard resolvedChunkURL.deletingLastPathComponent().path == resolvedPointsURL.path else {
            throw ReconstructionEngineError.unsafeOutputPath
        }
        return chunkURL
    }
}

private extension Data {
    mutating func appendLittleEndian(_ value: UInt16) {
        append(contentsOf: Swift.withUnsafeBytes(of: value.littleEndian, Array.init))
    }

    mutating func appendLittleEndian(_ value: UInt32) {
        append(contentsOf: Swift.withUnsafeBytes(of: value.littleEndian, Array.init))
    }

    mutating func appendLittleEndian(_ value: UInt64) {
        append(contentsOf: Swift.withUnsafeBytes(of: value.littleEndian, Array.init))
    }

    mutating func appendLittleEndian(_ value: Float) {
        appendLittleEndian(value.bitPattern)
    }
}
