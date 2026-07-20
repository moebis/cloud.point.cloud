@preconcurrency import AVFoundation
import AppKit
import SwiftUI

struct CameraPreviewView: NSViewRepresentable {
    let session: AVCaptureSession?

    func makeNSView(context: Context) -> CameraPreviewNSView {
        CameraPreviewNSView(session: session)
    }

    func updateNSView(_ nsView: CameraPreviewNSView, context: Context) {
        nsView.setSession(session)
    }

    static func dismantleNSView(_ nsView: CameraPreviewNSView, coordinator: ()) {
        nsView.tearDown()
    }
}

@MainActor
final class CameraPreviewNSView: NSView {
    let previewLayer = AVCaptureVideoPreviewLayer()

    init(session: AVCaptureSession?) {
        super.init(frame: .zero)
        wantsLayer = true
        layer = previewLayer
        previewLayer.videoGravity = .resizeAspect
        previewLayer.session = session
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) is unavailable")
    }

    override func layout() {
        super.layout()
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        previewLayer.frame = bounds
        CATransaction.commit()
    }

    func setSession(_ session: AVCaptureSession?) {
        guard previewLayer.session !== session else { return }
        previewLayer.session = session
    }

    func tearDown() {
        previewLayer.session = nil
        previewLayer.removeFromSuperlayer()
        layer = nil
    }
}
