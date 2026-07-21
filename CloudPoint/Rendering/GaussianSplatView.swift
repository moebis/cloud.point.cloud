import AppKit
import MetalKit
import SwiftUI

struct GaussianSplatView: NSViewRepresentable {
    @ObservedObject var renderer: GaussianSplatRenderer

    func makeNSView(context: Context) -> MTKView {
        let view = InteractiveGaussianSplatView(frame: .zero, device: renderer.device)
        view.gaussianRenderer = renderer
        view.delegate = renderer
        view.colorPixelFormat = .bgra8Unorm_srgb
        view.depthStencilPixelFormat = .depth32Float
        view.clearColor = MTLClearColor(red: 0.018, green: 0.025, blue: 0.04, alpha: 1)
        view.preferredFramesPerSecond = 60
        view.enableSetNeedsDisplay = false
        view.isPaused = false
        return view
    }

    func updateNSView(_ view: MTKView, context: Context) {
        view.delegate = renderer
        (view as? InteractiveGaussianSplatView)?.gaussianRenderer = renderer
    }
}

@MainActor
final class InteractiveGaussianSplatView: MTKView {
    weak var gaussianRenderer: GaussianSplatRenderer?
    private var lastDragLocation: CGPoint?

    override var acceptsFirstResponder: Bool { true }

    override func mouseDown(with event: NSEvent) {
        lastDragLocation = convert(event.locationInWindow, from: nil)
    }

    override func mouseDragged(with event: NSEvent) {
        applyDrag(event, pan: event.modifierFlags.contains(.shift))
    }

    override func rightMouseDown(with event: NSEvent) {
        lastDragLocation = convert(event.locationInWindow, from: nil)
    }

    override func rightMouseDragged(with event: NSEvent) { applyDrag(event, pan: true) }
    override func mouseUp(with event: NSEvent) { lastDragLocation = nil }
    override func rightMouseUp(with event: NSEvent) { lastDragLocation = nil }

    override func scrollWheel(with event: NSEvent) {
        gaussianRenderer?.zoom(by: Float(event.scrollingDeltaY))
    }

    private func applyDrag(_ event: NSEvent, pan: Bool) {
        let location = convert(event.locationInWindow, from: nil)
        defer { lastDragLocation = location }
        guard let previous = lastDragLocation else { return }
        let deltaX = Float(location.x - previous.x)
        let deltaY = Float(location.y - previous.y)
        if pan { gaussianRenderer?.pan(deltaX: deltaX, deltaY: deltaY) }
        else { gaussianRenderer?.orbit(deltaX: deltaX, deltaY: deltaY) }
    }
}
