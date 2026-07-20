#if DEBUG
import Darwin
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
    private var store: MockArtifactStore?
    private var isPaused = false
    private var pendingFrames: [PersistedFrame] = []
    private var replayFrames: [PersistedFrame] = []
    private var replayPosition = 0
    private var lastEnqueuedFrameIndex: UInt32?
    private var nextWindowIndex: UInt32 = 0
    private var windowIndexExhausted = false
    private var queuedUniqueOutputs: UInt64 = 0
    private var processedUniqueOutputs: UInt64 = 0
    private var completedWindowCount: UInt32 = 0
    private var lastCompletedWindowIndex: UInt32?

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
        do { try configuration.validate() }
        catch {
            lifecycle = .failed
            continuation.finish(throwing: error)
            throw error
        }
        lifecycle = .prepared
        continuation.yield(.ready(
            engineVersion: "mock-1.0",
            modelIdentifier: "mock-depth",
            modelRevision: "test",
            convertedWeightsSHA256: String(repeating: "0", count: 64)
        ))
    }

    func begin(project: ProjectDescriptor) async throws {
        guard lifecycle == .prepared else {
            throw ReconstructionEngineError.invalidLifecycle(operation: "begin")
        }
        do {
            let store = try MockArtifactStore(packageURL: project.packageURL)
            let manifest = try store.readManifestData().map(ProjectManifest.decode)
            if let manifest, manifest.projectID != project.projectID {
                throw ReconstructionEngineError.invalidResumeCheckpoint
            }

            let referencedPaths = Set(manifest?.completedWindows.flatMap { window in
                [window.pointChunkRelativePath] + window.frameArtifacts.flatMap {
                    [$0.depthRelativePath, $0.confidenceRelativePath, $0.geometryRelativePath]
                }
            } ?? [])
            try referencedPaths.forEach(store.validateReferencedOutput)

            if let checkpoint = project.resumeCheckpoint {
                guard checkpoint.replayFromFrameIndex <= checkpoint.lastCommittedFrameIndex,
                      let manifest,
                      try manifest.resumeCheckpoint() == checkpoint else {
                    throw ReconstructionEngineError.invalidResumeCheckpoint
                }
                let committedIndices = Set(manifest.completedWindows
                    .flatMap(\.frameArtifacts)
                    .map(\.frameIndex))
                replayFrames = manifest.frames.filter {
                    $0.index >= checkpoint.replayFromFrameIndex &&
                    $0.index <= checkpoint.lastCommittedFrameIndex &&
                    committedIndices.contains($0.index)
                }
                guard replayFrames.first?.index == checkpoint.replayFromFrameIndex,
                      replayFrames.last?.index == checkpoint.lastCommittedFrameIndex else {
                    throw ReconstructionEngineError.invalidResumeCheckpoint
                }
                nextWindowIndex = checkpoint.nextWindowIndex
            } else {
                guard manifest?.completedWindows.isEmpty ?? true else {
                    throw ReconstructionEngineError.invalidResumeCheckpoint
                }
                replayFrames = []
                nextWindowIndex = 0
            }

            try store.removeOrphans(referencedRelativePaths: referencedPaths)

            self.project = project
            self.store = store
            lifecycle = .acceptingInput
        } catch {
            lifecycle = .failed
            continuation.finish(throwing: error)
            throw error
        }
    }

    func enqueue(_ frame: PersistedFrame) async throws {
        guard lifecycle == .acceptingInput else {
            throw ReconstructionEngineError.invalidLifecycle(operation: "enqueue")
        }
        guard frame.sourceTimestamp.isFinite,
              frame.sourceTimestamp >= 0,
              ProjectRelativePath.isSafe(frame.relativePath),
              lastEnqueuedFrameIndex.map({ frame.index > $0 }) ?? true else {
            throw ReconstructionEngineError.replayOrderViolation
        }
        if replayPosition < replayFrames.count {
            guard frame == replayFrames[replayPosition] else {
                throw ReconstructionEngineError.replayOrderViolation
            }
            lastEnqueuedFrameIndex = frame.index
            replayPosition += 1
            return
        }
        if let checkpoint = project?.resumeCheckpoint,
           frame.index <= checkpoint.lastCommittedFrameIndex {
            throw ReconstructionEngineError.replayOrderViolation
        }
        guard hasCapacityForAnotherPendingWindow else {
            throw ReconstructionEngineError.windowIndexOverflow
        }

        let (queued, overflow) = queuedUniqueOutputs.addingReportingOverflow(1)
        guard !overflow else { throw ReconstructionEngineError.windowIndexOverflow }
        lastEnqueuedFrameIndex = frame.index
        queuedUniqueOutputs = queued
        pendingFrames.append(frame)
        try processPendingFramesIfPossible()
    }

    func finishInput() async throws {
        guard lifecycle == .acceptingInput else {
            throw ReconstructionEngineError.invalidLifecycle(operation: "finishInput")
        }
        guard replayPosition == replayFrames.count else {
            throw ReconstructionEngineError.replayOrderViolation
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
        continuation.yield(.paused(
            queuedFrames: queuedUniqueOutputs,
            processedFrames: processedUniqueOutputs
        ))
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
        guard lifecycle != .completed,
              lifecycle != .failed,
              lifecycle != .cancelled,
              lifecycle != .shutdown else {
            return
        }
        lifecycle = .cancelled
        pendingFrames.removeAll()
        continuation.yield(.cancelled(lastCompletedWindowIndex: lastCompletedWindowIndex))
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
        guard !isPaused, let store else { return }

        while !pendingFrames.isEmpty {
            let frame = pendingFrames.removeFirst()
            let windowIndex = nextWindowIndex
            continuation.yield(.frameStarted(frameIndex: frame.index, windowIndex: windowIndex))

            do {
                let output = try store.write(frame: frame, windowIndex: windowIndex)
                let (processed, processedOverflow) = processedUniqueOutputs.addingReportingOverflow(1)
                guard !processedOverflow else { throw ReconstructionEngineError.windowIndexOverflow }
                processedUniqueOutputs = processed
                completedWindowCount = try checkedIncrement(completedWindowCount)
                lastCompletedWindowIndex = windowIndex
                if windowIndex == .max {
                    windowIndexExhausted = true
                } else {
                    nextWindowIndex = windowIndex + 1
                }
                continuation.yield(.frameCompleted(output.artifacts))
                continuation.yield(.windowCompleted(output.window))
            } catch {
                lifecycle = .failed
                pendingFrames.removeAll()
                continuation.finish(throwing: error)
                throw error
            }
        }
    }

    private func checkedIncrement(_ value: UInt32) throws -> UInt32 {
        let (incremented, overflow) = value.addingReportingOverflow(1)
        guard !overflow else { throw ReconstructionEngineError.windowIndexOverflow }
        return incremented
    }

    private var hasCapacityForAnotherPendingWindow: Bool {
        guard !windowIndexExhausted else { return false }
        let remainingWindowCount = UInt64(UInt32.max - nextWindowIndex) + 1
        return UInt64(pendingFrames.count) < remainingWindowCount
    }

    private func finishIfInputIsComplete() {
        guard lifecycle == .inputFinished, !isPaused, pendingFrames.isEmpty else { return }
        lifecycle = .completed
        continuation.yield(.sessionCompleted(
            processedFrames: processedUniqueOutputs,
            windowCount: completedWindowCount,
            durationSeconds: 0
        ))
        continuation.finish()
    }
}

private final class MockArtifactStore: @unchecked Sendable {
    private static let planeDimension = 64
    private static let vertexStride = 24

    private let packageDescriptor: Int32
    private let predictionsDescriptor: Int32
    private let pointsDescriptor: Int32

    init(packageURL: URL) throws {
        let standardized = packageURL.standardizedFileURL
        guard standardized.path == standardized.resolvingSymlinksInPath().standardizedFileURL.path else {
            throw ReconstructionEngineError.unsafeOutputPath
        }
        let openedPackageDescriptor = Darwin.open(
            standardized.path,
            O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC
        )
        guard openedPackageDescriptor >= 0 else { throw ReconstructionEngineError.unsafeOutputPath }
        let openedPredictionsDescriptor: Int32
        do {
            openedPredictionsDescriptor = try Self.openDirectory(
                "Predictions",
                beneath: openedPackageDescriptor
            )
        } catch {
            Darwin.close(openedPackageDescriptor)
            throw error
        }
        let openedPointsDescriptor: Int32
        do {
            openedPointsDescriptor = try Self.openDirectory(
                "Points",
                beneath: openedPackageDescriptor
            )
        } catch {
            Darwin.close(openedPredictionsDescriptor)
            Darwin.close(openedPackageDescriptor)
            throw error
        }
        packageDescriptor = openedPackageDescriptor
        predictionsDescriptor = openedPredictionsDescriptor
        pointsDescriptor = openedPointsDescriptor
    }

    deinit {
        Darwin.close(pointsDescriptor)
        Darwin.close(predictionsDescriptor)
        Darwin.close(packageDescriptor)
    }

    func readManifestData() throws -> Data? {
        var status = stat()
        let statusResult = "Manifest.json".withCString {
            Darwin.fstatat(packageDescriptor, $0, &status, AT_SYMLINK_NOFOLLOW)
        }
        if statusResult != 0, errno == ENOENT { return nil }
        guard statusResult == 0, status.st_mode & S_IFMT == S_IFREG else {
            throw ReconstructionEngineError.unsafeOutputPath
        }
        let descriptor = "Manifest.json".withCString {
            Darwin.openat(packageDescriptor, $0, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        }
        guard descriptor >= 0 else { throw ReconstructionEngineError.unsafeOutputPath }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        do {
            let data = try handle.readToEnd() ?? Data()
            try handle.close()
            return data
        } catch {
            try? handle.close()
            throw error
        }
    }

    func write(frame: PersistedFrame, windowIndex: UInt32) throws -> (artifacts: FrameArtifacts, window: WindowResult) {
        try validateDirectory("Predictions", heldDescriptor: predictionsDescriptor)
        try validateDirectory("Points", heldDescriptor: pointsDescriptor)

        let depthPath = WorkerArtifactPath.depth(frameIndex: frame.index)
        let confidencePath = WorkerArtifactPath.confidence(frameIndex: frame.index)
        let geometryPath = WorkerArtifactPath.geometry(frameIndex: frame.index)
        let pointPath = WorkerArtifactPath.points(windowIndex: windowIndex)
        let files: [(descriptor: Int32, name: String, data: Data)] = [
            (predictionsDescriptor, URL(fileURLWithPath: depthPath).lastPathComponent, Self.depthData(frame.index)),
            (predictionsDescriptor, URL(fileURLWithPath: confidencePath).lastPathComponent, Self.confidenceData(frame.index)),
            (predictionsDescriptor, URL(fileURLWithPath: geometryPath).lastPathComponent, Self.geometryData(frame.index, windowIndex)),
            (pointsDescriptor, URL(fileURLWithPath: pointPath).lastPathComponent, Self.pointChunk(frame.index)),
        ]

        var committed: [(Int32, String)] = []
        do {
            for file in files {
                try writeExclusive(file.data, finalName: file.name, directoryDescriptor: file.descriptor)
                committed.append((file.descriptor, file.name))
            }
            try validateDirectory("Predictions", heldDescriptor: predictionsDescriptor)
            try validateDirectory("Points", heldDescriptor: pointsDescriptor)
        } catch {
            for (descriptor, name) in committed.reversed() {
                name.withCString { _ = Darwin.unlinkat(descriptor, $0, 0) }
                _ = Darwin.fsync(descriptor)
            }
            throw error
        }

        let artifacts = FrameArtifacts(
            frameIndex: frame.index,
            windowIndex: windowIndex,
            depthRelativePath: depthPath,
            confidenceRelativePath: confidencePath,
            geometryRelativePath: geometryPath,
            durationSeconds: 0
        )
        let window = WindowResult(
            windowIndex: windowIndex,
            inferenceFrameStart: frame.index,
            frameStart: frame.index,
            frameEnd: frame.index,
            pointChunkRelativePath: pointPath,
            alignmentRowMajor: [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
            ],
            lastProcessedFrameIndex: frame.index,
            inlierCount: UInt64(Self.planeDimension * Self.planeDimension),
            durationSeconds: 0
        )
        return (artifacts, window)
    }

    func validateReferencedOutput(_ relativePath: String) throws {
        let components = relativePath.split(separator: "/")
        guard components.count == 2 else { throw ReconstructionEngineError.unsafeOutputPath }
        let descriptor: Int32
        switch components[0] {
        case "Predictions": descriptor = predictionsDescriptor
        case "Points": descriptor = pointsDescriptor
        default: throw ReconstructionEngineError.unsafeOutputPath
        }
        var status = stat()
        let result = String(components[1]).withCString {
            Darwin.fstatat(descriptor, $0, &status, AT_SYMLINK_NOFOLLOW)
        }
        guard result == 0, status.st_mode & S_IFMT == S_IFREG else {
            throw ReconstructionEngineError.unsafeOutputPath
        }
    }

    func removeOrphans(referencedRelativePaths: Set<String>) throws {
        try validateDirectory("Predictions", heldDescriptor: predictionsDescriptor)
        try validateDirectory("Points", heldDescriptor: pointsDescriptor)
        try removeOrphans(
            directoryName: "Predictions",
            descriptor: predictionsDescriptor,
            referencedRelativePaths: referencedRelativePaths
        )
        try removeOrphans(
            directoryName: "Points",
            descriptor: pointsDescriptor,
            referencedRelativePaths: referencedRelativePaths
        )
    }

    private func removeOrphans(
        directoryName: String,
        descriptor: Int32,
        referencedRelativePaths: Set<String>
    ) throws {
        let freshDescriptor = ".".withCString {
            Darwin.openat(descriptor, $0, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        }
        guard freshDescriptor >= 0, let directory = fdopendir(freshDescriptor) else {
            if freshDescriptor >= 0 { Darwin.close(freshDescriptor) }
            throw ReconstructionEngineError.unsafeOutputPath
        }
        defer { closedir(directory) }

        while true {
            errno = 0
            guard let entry = readdir(directory) else {
                guard errno == 0 else { throw ReconstructionEngineError.unsafeOutputPath }
                break
            }
            let name = withUnsafePointer(to: entry.pointee.d_name) {
                $0.withMemoryRebound(to: CChar.self, capacity: Int(MAXNAMLEN) + 1) {
                    String(cString: $0)
                }
            }
            guard name != ".", name != ".." else { continue }
            let relativePath = "\(directoryName)/\(name)"
            let shouldRemove = Self.isCanonicalPartial(name, directory: directoryName) ||
                (Self.isCanonicalFinal(name, directory: directoryName) &&
                 !referencedRelativePaths.contains(relativePath))
            guard shouldRemove else { continue }

            var status = stat()
            let statusResult = name.withCString {
                Darwin.fstatat(descriptor, $0, &status, AT_SYMLINK_NOFOLLOW)
            }
            if statusResult != 0 {
                if errno == ENOENT { continue }
                throw ReconstructionEngineError.unsafeOutputPath
            }
            guard status.st_mode & S_IFMT == S_IFREG else { continue }
            let unlinkResult = name.withCString { Darwin.unlinkat(descriptor, $0, 0) }
            if unlinkResult != 0, errno != ENOENT {
                throw ReconstructionEngineError.unsafeOutputPath
            }
        }
        guard Darwin.fsync(descriptor) == 0 else { throw ReconstructionEngineError.unsafeOutputPath }
    }

    private static func isCanonicalFinal(_ name: String, directory: String) -> Bool {
        let pattern = directory == "Predictions"
            ? #"^[0-9]{8,10}\.(depth-f16|confidence-f16|geometry\.json)$"#
            : #"^window-[0-9]{8,10}\.cpc$"#
        guard name.range(of: pattern, options: .regularExpression) != nil else { return false }
        let digits: String
        if directory == "Predictions" {
            digits = String(name.prefix { $0.isNumber })
        } else {
            digits = String(name.dropFirst("window-".count).prefix { $0.isNumber })
        }
        guard let index = UInt32(digits) else { return false }
        if directory == "Predictions" {
            return [
                URL(fileURLWithPath: WorkerArtifactPath.depth(frameIndex: index)).lastPathComponent,
                URL(fileURLWithPath: WorkerArtifactPath.confidence(frameIndex: index)).lastPathComponent,
                URL(fileURLWithPath: WorkerArtifactPath.geometry(frameIndex: index)).lastPathComponent,
            ].contains(name)
        }
        return URL(fileURLWithPath: WorkerArtifactPath.points(windowIndex: index)).lastPathComponent == name
    }

    private static func isCanonicalPartial(_ name: String, directory: String) -> Bool {
        guard name.first == ".", name.hasSuffix(".partial") else { return false }
        let body = String(name.dropFirst().dropLast(".partial".count))
        guard let separator = body.lastIndex(of: ".") else { return false }
        let finalName = String(body[..<separator])
        let uuid = String(body[body.index(after: separator)...])
        guard uuid == uuid.lowercased(),
              let parsedUUID = UUID(uuidString: uuid),
              parsedUUID.uuidString.lowercased() == uuid else {
            return false
        }
        return isCanonicalFinal(finalName, directory: directory)
    }

    private func writeExclusive(_ data: Data, finalName: String, directoryDescriptor: Int32) throws {
        let partialName = ".\(finalName).\(UUID().uuidString.lowercased()).partial"
        let descriptor = partialName.withCString {
            Darwin.openat(
                directoryDescriptor,
                $0,
                O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                mode_t(S_IRUSR | S_IWUSR)
            )
        }
        guard descriptor >= 0 else { throw ReconstructionEngineError.unsafeOutputPath }
        var descriptorOpen = true
        do {
            try Self.writeAll(data, descriptor: descriptor)
            guard Darwin.fsync(descriptor) == 0 else { throw ReconstructionEngineError.unsafeOutputPath }
            guard Darwin.close(descriptor) == 0 else {
                descriptorOpen = false
                throw ReconstructionEngineError.unsafeOutputPath
            }
            descriptorOpen = false
            let renameResult = partialName.withCString { partial in
                finalName.withCString { final in
                    renameatx_np(
                        directoryDescriptor,
                        partial,
                        directoryDescriptor,
                        final,
                        UInt32(RENAME_EXCL)
                    )
                }
            }
            guard renameResult == 0 else {
                if errno == EEXIST { throw ReconstructionEngineError.outputAlreadyExists(finalName) }
                throw ReconstructionEngineError.unsafeOutputPath
            }
            guard Darwin.fsync(directoryDescriptor) == 0 else {
                finalName.withCString { _ = Darwin.unlinkat(directoryDescriptor, $0, 0) }
                throw ReconstructionEngineError.unsafeOutputPath
            }
        } catch {
            if descriptorOpen { Darwin.close(descriptor) }
            partialName.withCString { _ = Darwin.unlinkat(directoryDescriptor, $0, 0) }
            throw error
        }
    }

    private func validateDirectory(_ name: String, heldDescriptor: Int32) throws {
        var held = stat()
        var current = stat()
        let currentResult = name.withCString {
            Darwin.fstatat(packageDescriptor, $0, &current, AT_SYMLINK_NOFOLLOW)
        }
        guard Darwin.fstat(heldDescriptor, &held) == 0,
              currentResult == 0,
              held.st_mode & S_IFMT == S_IFDIR,
              current.st_mode & S_IFMT == S_IFDIR,
              held.st_dev == current.st_dev,
              held.st_ino == current.st_ino else {
            throw ReconstructionEngineError.unsafeOutputPath
        }
    }

    private static func openDirectory(_ name: String, beneath descriptor: Int32) throws -> Int32 {
        let opened = name.withCString {
            Darwin.openat(descriptor, $0, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        }
        guard opened >= 0 else { throw ReconstructionEngineError.unsafeOutputPath }
        return opened
    }

    private static func writeAll(_ data: Data, descriptor: Int32) throws {
        try data.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else { return }
            var offset = 0
            while offset < bytes.count {
                let written = Darwin.write(descriptor, base.advanced(by: offset), bytes.count - offset)
                if written > 0 { offset += written }
                else if written < 0, errno == EINTR { continue }
                else { throw ReconstructionEngineError.unsafeOutputPath }
            }
        }
    }

    private static func depthData(_ frameIndex: UInt32) -> Data {
        var data = Data("CPD1".utf8)
        data.appendLittleEndian(frameIndex)
        data.appendLittleEndian(Float16(1).bitPattern)
        return data
    }

    private static func confidenceData(_ frameIndex: UInt32) -> Data {
        var data = Data("CPCF".utf8)
        data.appendLittleEndian(frameIndex)
        data.appendLittleEndian(Float16(2).bitPattern)
        return data
    }

    private static func geometryData(_ frameIndex: UInt32, _ windowIndex: UInt32) -> Data {
        Data("{\"frameIndex\":\(frameIndex),\"windowIndex\":\(windowIndex)}".utf8)
    }

    private static func pointChunk(_ sourceFrame: UInt32) -> Data {
        let pointCount = planeDimension * planeDimension
        var data = Data()
        data.reserveCapacity(32 + (pointCount * vertexStride))
        data.append(contentsOf: "CPC1".utf8)
        data.appendLittleEndian(UInt16(1))
        data.appendLittleEndian(UInt16(vertexStride))
        data.appendLittleEndian(UInt64(pointCount))
        data.appendLittleEndian(sourceFrame)
        data.appendLittleEndian(sourceFrame)
        data.append(contentsOf: repeatElement(0, count: 8))

        let color = color(for: sourceFrame)
        let z = Float(sourceFrame) * 0.01
        let halfDimension = Float(planeDimension - 1) / 2
        for row in 0..<planeDimension {
            for column in 0..<planeDimension {
                data.appendLittleEndian((Float(column) - halfDimension) * 0.04)
                data.appendLittleEndian((Float(row) - halfDimension) * 0.04)
                data.appendLittleEndian(z)
                data.append(contentsOf: [color.red, color.green, color.blue, 255])
                data.appendLittleEndian(Float16(2).bitPattern)
                data.appendLittleEndian(UInt16(0))
                data.appendLittleEndian(sourceFrame)
            }
        }
        return data
    }

    private static func color(for frameIndex: UInt32) -> (red: UInt8, green: UInt8, blue: UInt8) {
        (
            UInt8(truncatingIfNeeded: frameIndex &* 53),
            UInt8(truncatingIfNeeded: frameIndex &* 97),
            UInt8(truncatingIfNeeded: frameIndex &* 193)
        )
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
#endif
