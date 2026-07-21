@preconcurrency import MetalKit
import MetalSplatter
import SplatIO
import SwiftUI
import simd

enum GaussianViewerState: Sendable, Equatable {
    case empty
    case loading
    case ready(count: Int)
    case failed(String)
}

@MainActor
final class GaussianSplatRenderer: NSObject, ObservableObject, MTKViewDelegate {
    static let openCVToMetal = simd_float4x4(diagonal: SIMD4<Float>(1, -1, -1, 1))

    let device: MTLDevice
    @Published private(set) var state: GaussianViewerState = .empty

    private let commandQueue: MTLCommandQueue
    private let splatRenderer: SplatRenderer
    private let inFlight = DispatchSemaphore(value: 3)
    private var loadTask: Task<Void, Never>?
    private var loadedURL: URL?
    private var drawableSize = CGSize(width: 1, height: 1)
    private var target = SIMD3<Float>(0, 0, -1)
    private var sceneRadius: Float = 1
    private var distance: Float = 2.5
    private var yaw: Float = 0
    private var pitch: Float = 0
    private var panOffset = SIMD2<Float>.zero
    private var mirrorDisplay = false

    init(device: MTLDevice) throws {
        guard let commandQueue = device.makeCommandQueue() else {
            throw GaussianViewerError.metalUnavailable
        }
        self.device = device
        self.commandQueue = commandQueue
        splatRenderer = try SplatRenderer(
            device: device,
            colorFormat: .bgra8Unorm_srgb,
            depthFormat: .depth32Float,
            sampleCount: 1,
            maxViewCount: 1,
            maxSimultaneousRenders: 3,
            highQualityDepth: false,
            clearColor: MTLClearColor(red: 0.018, green: 0.025, blue: 0.04, alpha: 1)
        )
        super.init()
    }

    deinit { loadTask?.cancel() }

    func load(_ url: URL?) {
        guard loadedURL?.standardizedFileURL != url?.standardizedFileURL else { return }
        loadedURL = url?.standardizedFileURL
        loadTask?.cancel()
        guard let url else {
            state = .empty
            return
        }
        state = .loading
        loadTask = Task { [weak self, device] in
            do {
                let points = try await Task.detached(priority: .userInitiated) {
                    let reader = try AutodetectSceneReader(url)
                    return try await reader.readAll()
                }.value
                try Task.checkCancellation()
                guard !points.isEmpty else { throw GaussianViewerError.emptyScene }
                let bounds = try Self.bounds(of: points)
                let chunk = try SplatChunk(device: device, from: points)
                try Task.checkCancellation()
                guard let self else { return }
                await self.splatRenderer.addChunk(chunk)
                guard !Task.isCancelled, self.loadedURL == url.standardizedFileURL else { return }
                self.target = bounds.center
                self.sceneRadius = bounds.radius
                self.resetCamera()
                self.state = .ready(count: points.count)
            } catch is CancellationError {
            } catch {
                guard let self else { return }
                self.state = .failed(error.localizedDescription)
            }
        }
    }

    func resetCamera() {
        yaw = 0
        pitch = 0
        panOffset = .zero
        distance = max(sceneRadius * 2.4, 0.25)
    }

    func orbit(deltaX: Float, deltaY: Float) {
        yaw += deltaX * 0.006
        pitch = min(max(pitch + deltaY * 0.006, -.pi * 0.49), .pi * 0.49)
    }

    func pan(deltaX: Float, deltaY: Float) {
        let scale = max(distance, 0.1) * 0.0015
        panOffset.x += deltaX * scale
        panOffset.y += deltaY * scale
    }

    func zoom(by delta: Float) {
        distance = min(max(distance * exp(delta * 0.008), sceneRadius * 0.03), sceneRadius * 30)
    }

    func setMirrorDisplay(_ enabled: Bool) { mirrorDisplay = enabled }

    func draw(in view: MTKView) {
        guard case .ready = state,
              splatRenderer.isReadyToRender,
              let drawable = view.currentDrawable,
              let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        guard inFlight.wait(timeout: .now()) == .success else { return }
        commandBuffer.addCompletedHandler { [inFlight] _ in inFlight.signal() }

        let descriptor = viewportDescriptor()
        do {
            let rendered = try splatRenderer.render(
                viewports: [descriptor],
                colorTexture: drawable.texture,
                colorStoreAction: .store,
                depthTexture: view.depthStencilTexture,
                rasterizationRateMap: nil,
                renderTargetArrayLength: 0,
                to: commandBuffer
            )
            if rendered { commandBuffer.present(drawable) }
        } catch {
            state = .failed(error.localizedDescription)
        }
        commandBuffer.commit()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        drawableSize = size
    }

    private func viewportDescriptor() -> SplatRenderer.ViewportDescriptor {
        let width = max(drawableSize.width, 1)
        let height = max(drawableSize.height, 1)
        let nearZ: Float = max(sceneRadius * 0.002, 0.001)
        let farZ: Float = max(distance + sceneRadius * 10, 100)
        let projection = Self.perspective(
            verticalFOV: .pi / 3,
            aspect: Float(width / height),
            nearZ: nearZ,
            farZ: farZ
        )
        let view = Self.modelViewMatrix(
            target: target,
            distance: distance,
            yaw: yaw,
            pitch: pitch,
            panOffset: panOffset,
            mirrorDisplay: mirrorDisplay
        )
        return SplatRenderer.ViewportDescriptor(
            viewport: MTLViewport(
                originX: 0,
                originY: 0,
                width: width,
                height: height,
                znear: 0,
                zfar: 1
            ),
            projectionMatrix: projection,
            viewMatrix: view,
            screenSize: SIMD2(Int(width), Int(height))
        )
    }

    private nonisolated static func bounds(
        of points: [SplatPoint]
    ) throws -> (center: SIMD3<Float>, radius: Float) {
        var lower = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var upper = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for point in points {
            guard allFinite(point.position) else { throw GaussianViewerError.invalidScene }
            let converted = SIMD3<Float>(point.position.x, -point.position.y, -point.position.z)
            lower = simd_min(lower, converted)
            upper = simd_max(upper, converted)
        }
        let center = (lower + upper) * 0.5
        let radius = max(simd_length(upper - lower) * 0.5, 0.01)
        guard allFinite(center), radius.isFinite else { throw GaussianViewerError.invalidScene }
        return (center, radius)
    }

    private nonisolated static func allFinite(_ value: SIMD3<Float>) -> Bool {
        value.x.isFinite && value.y.isFinite && value.z.isFinite
    }

    static func modelViewMatrix(
        target: SIMD3<Float>,
        distance: Float,
        yaw: Float,
        pitch: Float,
        panOffset: SIMD2<Float>,
        mirrorDisplay: Bool
    ) -> simd_float4x4 {
        let mirror = simd_float4x4(
            diagonal: SIMD4<Float>(mirrorDisplay ? -1 : 1, 1, 1, 1)
        )
        return translation(panOffset.x, panOffset.y, -distance)
            * rotationX(pitch)
            * rotationY(yaw)
            * mirror
            * translation(-target.x, -target.y, -target.z)
            * openCVToMetal
    }

    private static func perspective(
        verticalFOV: Float,
        aspect: Float,
        nearZ: Float,
        farZ: Float
    ) -> simd_float4x4 {
        let y = 1 / tan(verticalFOV * 0.5)
        let x = y / aspect
        let z = farZ / (nearZ - farZ)
        return simd_float4x4(columns: (
            SIMD4<Float>(x, 0, 0, 0),
            SIMD4<Float>(0, y, 0, 0),
            SIMD4<Float>(0, 0, z, -1),
            SIMD4<Float>(0, 0, nearZ * z, 0)
        ))
    }

    private static func translation(_ x: Float, _ y: Float, _ z: Float) -> simd_float4x4 {
        var matrix = matrix_identity_float4x4
        matrix.columns.3 = SIMD4<Float>(x, y, z, 1)
        return matrix
    }

    private static func rotationX(_ angle: Float) -> simd_float4x4 {
        let c = cos(angle)
        let s = sin(angle)
        return simd_float4x4(columns: (
            SIMD4<Float>(1, 0, 0, 0),
            SIMD4<Float>(0, c, s, 0),
            SIMD4<Float>(0, -s, c, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
    }

    private static func rotationY(_ angle: Float) -> simd_float4x4 {
        let c = cos(angle)
        let s = sin(angle)
        return simd_float4x4(columns: (
            SIMD4<Float>(c, 0, -s, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(s, 0, c, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
    }
}

enum GaussianViewerError: Error, LocalizedError {
    case metalUnavailable
    case emptyScene
    case invalidScene

    var errorDescription: String? {
        switch self {
        case .metalUnavailable: "Metal is unavailable on this Mac."
        case .emptyScene: "The Gaussian scene contains no splats."
        case .invalidScene: "The Gaussian scene contains invalid geometry."
        }
    }
}
