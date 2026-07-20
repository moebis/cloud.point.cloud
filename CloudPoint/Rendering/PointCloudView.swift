import AppKit
import MetalKit
import SwiftUI

struct PointCloudView: NSViewRepresentable {
    let renderer: PointCloudRenderer

    func makeNSView(context: Context) -> MTKView {
        let view = InteractivePointCloudView(frame: .zero, device: renderer.device)
        view.pointRenderer = renderer
        view.delegate = renderer
        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        view.clearColor = MTLClearColor(red: 0.025, green: 0.03, blue: 0.04, alpha: 1)
        view.preferredFramesPerSecond = 60
        view.enableSetNeedsDisplay = false
        view.isPaused = false
        return view
    }

    func updateNSView(_ view: MTKView, context: Context) {
        view.delegate = renderer
    }
}

@MainActor
private final class InteractivePointCloudView: MTKView {
    weak var pointRenderer: PointCloudRenderer?
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

    override func rightMouseDragged(with event: NSEvent) {
        applyDrag(event, pan: true)
    }

    override func mouseUp(with event: NSEvent) {
        lastDragLocation = nil
    }

    override func rightMouseUp(with event: NSEvent) {
        lastDragLocation = nil
    }

    override func scrollWheel(with event: NSEvent) {
        pointRenderer?.zoom(by: Float(event.scrollingDeltaY))
    }

    private func applyDrag(_ event: NSEvent, pan: Bool) {
        let location = convert(event.locationInWindow, from: nil)
        defer { lastDragLocation = location }
        guard let previous = lastDragLocation else { return }
        let deltaX = Float(location.x - previous.x)
        let deltaY = Float(location.y - previous.y)
        if pan {
            pointRenderer?.pan(deltaX: deltaX, deltaY: deltaY)
        } else {
            pointRenderer?.orbit(deltaX: deltaX, deltaY: deltaY)
        }
    }
}
