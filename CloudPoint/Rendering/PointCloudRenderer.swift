import Foundation
import Metal
import MetalKit
import simd

enum PointCloudRendererError: Error {
    case invalidDisplayLimit(Int)
    case deviceCannotDisplayPointRecord
    case pointCountOverflow
    case missingShaderFunction(String)
    case commandQueueCreationFailed
    case compactionExceededDisplayLimit(actual: Int, limit: Int)
    case bufferAllocationFailed(length: Int)
}

struct PointDrawRange: Sendable, Equatable {
    let chunkIndex: Int
    let range: Range<Int>
}

struct PointCloudCameraState: Sendable, Equatable {
    var yaw: Float
    var pitch: Float
    var distance: Float
    var target: SIMD3<Float>

    static let `default` = PointCloudCameraState(
        yaw: 0,
        pitch: 0,
        distance: 5,
        target: .zero
    )
}

@MainActor
final class PointCloudRenderer: NSObject, MTKViewDelegate {
    static let maximumDisplayPointCount = 5_000_000
    static let uniformBufferStride = MemoryLayout<PointCloudUniforms>.stride

    let device: any MTLDevice
    let displayLimit: Int

    private(set) var fullPointCount = 0
    private(set) var displayedPointCount = 0
    private(set) var displayedSourceIndices: [Int] = []
    private(set) var drawRanges: [PointDrawRange] = []
    private(set) var bufferCapacity = 0
    private(set) var pointSize: Float = 3
    private(set) var confidenceThreshold: Float = 1.5
    private(set) var cameraState: PointCloudCameraState = .default

    private let commandQueue: any MTLCommandQueue
    private let pipelineState: any MTLRenderPipelineState
    private let depthState: any MTLDepthStencilState
    private var chunks: [PointChunk] = []
    private var vertexBuffer: (any MTLBuffer)?

    init(
        device: any MTLDevice,
        displayLimit requestedDisplayLimit: Int = maximumDisplayPointCount,
        libraryBundle: Bundle = .main
    ) throws {
        guard requestedDisplayLimit > 0 else {
            throw PointCloudRendererError.invalidDisplayLimit(requestedDisplayLimit)
        }
        let contractLimit = min(requestedDisplayLimit, Self.maximumDisplayPointCount)
        let deviceLimit = device.maxBufferLength / PointChunk.vertexStride
        guard deviceLimit > 0 else {
            throw PointCloudRendererError.deviceCannotDisplayPointRecord
        }
        displayLimit = min(contractLimit, deviceLimit)
        self.device = device

        guard let commandQueue = device.makeCommandQueue() else {
            throw PointCloudRendererError.commandQueueCreationFailed
        }
        self.commandQueue = commandQueue

        let library = try device.makeDefaultLibrary(bundle: libraryBundle)
        guard let vertexFunction = library.makeFunction(name: "pointCloudVertex") else {
            throw PointCloudRendererError.missingShaderFunction("pointCloudVertex")
        }
        guard let fragmentFunction = library.makeFunction(name: "pointCloudFragment") else {
            throw PointCloudRendererError.missingShaderFunction("pointCloudFragment")
        }
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "CloudPoint point pipeline"
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
        pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)

        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.depthCompareFunction = .less
        depthDescriptor.isDepthWriteEnabled = true
        guard let depthState = device.makeDepthStencilState(descriptor: depthDescriptor) else {
            throw PointCloudRendererError.deviceCannotDisplayPointRecord
        }
        self.depthState = depthState

        super.init()
    }

    func append(_ chunk: PointChunk) throws {
        let (nextFullCount, overflow) = fullPointCount.addingReportingOverflow(chunk.pointCount)
        guard !overflow else {
            throw PointCloudRendererError.pointCountOverflow
        }

        var candidateChunks = chunks
        candidateChunks.append(chunk)
        let selection = SpatialPointCompactor.select(
            chunks: candidateChunks,
            fullPointCount: nextFullCount,
            limit: displayLimit
        )
        guard selection.references.count <= displayLimit else {
            throw PointCloudRendererError.compactionExceededDisplayLimit(
                actual: selection.references.count,
                limit: displayLimit
            )
        }
        let nextCapacity = Self.geometricCapacity(
            current: bufferCapacity,
            required: selection.references.count,
            limit: displayLimit
        )
        let newBuffer = try makeImmutableBuffer(
            capacity: nextCapacity,
            references: selection.references,
            chunks: candidateChunks
        )

        // Buffers are immutable after publication. Rebuilds allocate a fresh shared
        // buffer, so an in-flight command buffer can continue retaining and reading
        // the old allocation without racing a CPU overwrite.
        vertexBuffer = newBuffer
        bufferCapacity = nextCapacity
        chunks = candidateChunks
        fullPointCount = nextFullCount
        displayedPointCount = selection.references.count
        displayedSourceIndices = selection.references.map(\.globalIndex)
        drawRanges = Self.makeDrawRanges(from: selection.references)
    }

    func setConfidenceThreshold(_ value: Float) {
        guard value.isFinite else { return }
        confidenceThreshold = min(max(value, 0), Float(Float16.greatestFiniteMagnitude))
    }

    func setPointSize(_ value: Float) {
        guard value.isFinite else { return }
        pointSize = min(max(value, 1), 64)
    }

    func orbit(deltaX: Float, deltaY: Float) {
        let x = Self.finiteInput(deltaX)
        let y = Self.finiteInput(deltaY)
        yaw = Self.wrapped(cameraState.yaw + x * 0.005)
        cameraState.pitch = min(max(cameraState.pitch + y * 0.005, -1.55), 1.55)
    }

    func pan(deltaX: Float, deltaY: Float) {
        let x = Self.finiteInput(deltaX)
        let y = Self.finiteInput(deltaY)
        let scale = min(cameraState.distance, 10_000) * 0.001
        let right = SIMD3<Float>(cos(cameraState.yaw), 0, -sin(cameraState.yaw))
        let up = SIMD3<Float>(0, 1, 0)
        let candidate = cameraState.target + ((-x * scale) * right) + ((y * scale) * up)
        cameraState.target = SIMD3(
            min(max(candidate.x, -1_000_000), 1_000_000),
            min(max(candidate.y, -1_000_000), 1_000_000),
            min(max(candidate.z, -1_000_000), 1_000_000)
        )
    }

    func zoom(by delta: Float) {
        let bounded = Self.finiteInput(delta)
        let factor = exp(min(max(bounded * 0.002, -20), 20))
        cameraState.distance = min(max(cameraState.distance * factor, 0.05), 10_000)
    }

    func resetCamera() {
        cameraState = .default
    }

    func viewProjectionMatrix(aspectRatio: Float) -> simd_float4x4 {
        let safeAspect: Float
        if aspectRatio.isFinite, aspectRatio > 0 {
            safeAspect = min(max(aspectRatio, 0.000_1), 10_000)
        } else {
            safeAspect = 1
        }
        let cosPitch = cos(cameraState.pitch)
        let backward = simd_normalize(
            SIMD3<Float>(
                sin(cameraState.yaw) * cosPitch,
                sin(cameraState.pitch),
                cos(cameraState.yaw) * cosPitch
            )
        )
        let view = Self.lookAt(
            target: cameraState.target,
            backward: backward,
            distance: cameraState.distance
        )
        let projection = Self.perspective(
            verticalFieldOfView: .pi / 3,
            aspectRatio: safeAspect,
            near: 0.01,
            far: 100_000
        )
        return projection * view
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard displayedPointCount > 0,
              let vertexBuffer,
              view.drawableSize.height > 0,
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)
        else {
            return
        }

        var uniforms = PointCloudUniforms(
            viewProjection: viewProjectionMatrix(
                aspectRatio: Float(view.drawableSize.width / view.drawableSize.height)
            ),
            pointSize: pointSize,
            confidenceThreshold: confidenceThreshold,
            padding: .zero
        )

        encoder.label = "CloudPoint point encoder"
        encoder.setRenderPipelineState(pipelineState)
        encoder.setDepthStencilState(depthState)
        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<PointCloudUniforms>.stride, index: 1)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<PointCloudUniforms>.stride, index: 1)
        for drawRange in drawRanges where !drawRange.range.isEmpty {
            encoder.drawPrimitives(
                type: .point,
                vertexStart: drawRange.range.lowerBound,
                vertexCount: drawRange.range.count
            )
        }
        encoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    private var yaw: Float {
        get { cameraState.yaw }
        set { cameraState.yaw = newValue }
    }

    private func makeImmutableBuffer(
        capacity: Int,
        references: [PointReference],
        chunks: [PointChunk]
    ) throws -> (any MTLBuffer)? {
        guard capacity > 0 else { return nil }
        let (length, overflow) = capacity.multipliedReportingOverflow(by: PointChunk.vertexStride)
        guard !overflow, length <= device.maxBufferLength else {
            throw PointCloudRendererError.bufferAllocationFailed(length: Int.max)
        }
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw PointCloudRendererError.bufferAllocationFailed(length: length)
        }
        buffer.label = "CloudPoint immutable displayed points"

        let destination = buffer.contents()
        var destinationIndex = 0
        var referenceIndex = 0
        while referenceIndex < references.count {
            let chunkIndex = references[referenceIndex].chunkIndex
            let groupStart = referenceIndex
            while referenceIndex < references.count,
                  references[referenceIndex].chunkIndex == chunkIndex {
                referenceIndex += 1
            }
            chunks[chunkIndex].withVertexBytes { source in
                for index in groupStart..<referenceIndex {
                    let sourceOffset = references[index].pointIndex * PointChunk.vertexStride
                    destination
                        .advanced(by: destinationIndex * PointChunk.vertexStride)
                        .copyMemory(
                            from: source.baseAddress!.advanced(by: sourceOffset),
                            byteCount: PointChunk.vertexStride
                        )
                    destinationIndex += 1
                }
            }
        }
        return buffer
    }

    private static func geometricCapacity(current: Int, required: Int, limit: Int) -> Int {
        guard required > 0 else { return 0 }
        var capacity = max(current, 1)
        while capacity < required {
            if capacity > limit / 2 {
                capacity = limit
                break
            }
            capacity *= 2
        }
        return min(capacity, limit)
    }

    private static func makeDrawRanges(from references: [PointReference]) -> [PointDrawRange] {
        guard let first = references.first else { return [] }
        var result: [PointDrawRange] = []
        var currentChunk = first.chunkIndex
        var rangeStart = 0
        for index in 1..<references.count where references[index].chunkIndex != currentChunk {
            result.append(PointDrawRange(chunkIndex: currentChunk, range: rangeStart..<index))
            currentChunk = references[index].chunkIndex
            rangeStart = index
        }
        result.append(
            PointDrawRange(chunkIndex: currentChunk, range: rangeStart..<references.count)
        )
        return result
    }

    private static func finiteInput(_ value: Float) -> Float {
        guard value.isFinite else { return 0 }
        return min(max(value, -10_000), 10_000)
    }

    private static func wrapped(_ value: Float) -> Float {
        guard value.isFinite else { return 0 }
        return value.truncatingRemainder(dividingBy: 2 * .pi)
    }

    private static func lookAt(
        target: SIMD3<Float>,
        backward: SIMD3<Float>,
        distance: Float
    ) -> simd_float4x4 {
        let right = simd_normalize(simd_cross(SIMD3<Float>(0, 1, 0), backward))
        let up = simd_cross(backward, right)
        return simd_float4x4(
            SIMD4(right.x, up.x, backward.x, 0),
            SIMD4(right.y, up.y, backward.y, 0),
            SIMD4(right.z, up.z, backward.z, 0),
            SIMD4(
                -simd_dot(right, target),
                -simd_dot(up, target),
                -simd_dot(backward, target) - distance,
                1
            )
        )
    }

    private static func perspective(
        verticalFieldOfView: Float,
        aspectRatio: Float,
        near: Float,
        far: Float
    ) -> simd_float4x4 {
        let yScale = 1 / tan(verticalFieldOfView * 0.5)
        let xScale = yScale / aspectRatio
        let zScale = far / (near - far)
        return simd_float4x4(
            SIMD4(xScale, 0, 0, 0),
            SIMD4(0, yScale, 0, 0),
            SIMD4(0, 0, zScale, -1),
            SIMD4(0, 0, near * zScale, 0)
        )
    }
}

private struct PointCloudUniforms {
    var viewProjection: simd_float4x4
    var pointSize: Float
    var confidenceThreshold: Float
    var padding: SIMD2<Float>
}

private struct PointReference: Sendable, Equatable {
    let chunkIndex: Int
    let pointIndex: Int
    let globalIndex: Int
}

private struct SpatialSelection {
    let references: [PointReference]
}

private enum SpatialPointCompactor {
    private struct Bounds {
        var minimum = SIMD3<Double>(repeating: .greatestFiniteMagnitude)
        var maximum = SIMD3<Double>(repeating: -.greatestFiniteMagnitude)

        mutating func include(_ point: SIMD3<Float>) {
            let widened = SIMD3<Double>(
                Double(point.x),
                Double(point.y),
                Double(point.z)
            )
            minimum = simd_min(minimum, widened)
            maximum = simd_max(maximum, widened)
        }
    }

    private struct Voxel: Hashable {
        let x: Int
        let y: Int
        let z: Int
    }

    static func select(
        chunks: [PointChunk],
        fullPointCount: Int,
        limit: Int
    ) -> SpatialSelection {
        if fullPointCount <= limit {
            var references: [PointReference] = []
            references.reserveCapacity(fullPointCount)
            var globalIndex = 0
            for (chunkIndex, chunk) in chunks.enumerated() {
                for pointIndex in 0..<chunk.pointCount {
                    references.append(
                        PointReference(
                            chunkIndex: chunkIndex,
                            pointIndex: pointIndex,
                            globalIndex: globalIndex
                        )
                    )
                    globalIndex += 1
                }
            }
            return SpatialSelection(references: references)
        }

        var bounds = Bounds()
        for chunk in chunks {
            chunk.forEachVertex { _, vertex in
                bounds.include(vertex.position)
            }
        }
        let extents = bounds.maximum - bounds.minimum
        let largestExtent = max(extents.x, max(extents.y, extents.z))
        let activeThreshold = max(
            largestExtent * Double.ulpOfOne * 8,
            Double.leastNormalMagnitude
        )
        let activeAxes = [extents.x, extents.y, extents.z].filter { $0 > activeThreshold }.count
        var side: Int
        if activeAxes == 0 {
            side = 1
        } else {
            side = max(1, Int(floor(pow(Double(limit), 1 / Double(activeAxes)))))
        }
        while voxelCapacity(side: side, dimensions: activeAxes) > limit, side > 1 {
            side -= 1
        }
        let bins = SIMD3<Int>(
            extents.x > activeThreshold ? side : 1,
            extents.y > activeThreshold ? side : 1,
            extents.z > activeThreshold ? side : 1
        )

        var representatives: [Voxel: PointReference] = [:]
        representatives.reserveCapacity(min(limit, fullPointCount))
        var globalIndex = 0
        for (chunkIndex, chunk) in chunks.enumerated() {
            chunk.forEachVertex { pointIndex, vertex in
                let voxel = Voxel(
                    x: bin(Double(vertex.position.x), minimum: bounds.minimum.x, extent: extents.x, count: bins.x),
                    y: bin(Double(vertex.position.y), minimum: bounds.minimum.y, extent: extents.y, count: bins.y),
                    z: bin(Double(vertex.position.z), minimum: bounds.minimum.z, extent: extents.z, count: bins.z)
                )
                if representatives[voxel] == nil {
                    representatives[voxel] = PointReference(
                        chunkIndex: chunkIndex,
                        pointIndex: pointIndex,
                        globalIndex: globalIndex
                    )
                }
                globalIndex += 1
            }
        }
        let references = representatives.values.sorted { $0.globalIndex < $1.globalIndex }
        return SpatialSelection(references: references)
    }

    private static func bin(_ value: Double, minimum: Double, extent: Double, count: Int) -> Int {
        guard count > 1, extent > 0 else { return 0 }
        let normalized = (value - minimum) / extent
        return min(count - 1, max(0, Int(floor(normalized * Double(count)))))
    }

    private static func voxelCapacity(side: Int, dimensions: Int) -> Int {
        guard dimensions > 0 else { return 1 }
        var result = 1
        for _ in 0..<dimensions {
            let (next, overflow) = result.multipliedReportingOverflow(by: side)
            if overflow { return .max }
            result = next
        }
        return result
    }
}
