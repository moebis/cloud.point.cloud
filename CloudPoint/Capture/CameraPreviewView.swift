@preconcurrency import AVFoundation
import AppKit
import SwiftUI

enum CameraConnectionRole: Sendable, Equatable {
    case preview
    case frameOutput
}

struct CameraDisplayPolicy: Sendable, Equatable {
    var mirrorDisplay: Bool

    func shouldMirrorVideo(for role: CameraConnectionRole) -> Bool {
        role == .preview && mirrorDisplay
    }

    func configure(_ connection: AVCaptureConnection?, for role: CameraConnectionRole) {
        guard let connection, connection.isVideoMirroringSupported else { return }
        connection.automaticallyAdjustsVideoMirroring = false
        connection.isVideoMirrored = shouldMirrorVideo(for: role)
    }
}

struct CameraPreviewView: NSViewRepresentable {
    let session: AVCaptureSession?
    var mirrorDisplay = true

    func makeNSView(context: Context) -> CameraPreviewNSView {
        CameraPreviewNSView(session: session, mirrorDisplay: mirrorDisplay)
    }

    func updateNSView(_ nsView: CameraPreviewNSView, context: Context) {
        nsView.setSession(session)
        nsView.setMirrorDisplay(mirrorDisplay)
    }

    static func dismantleNSView(_ nsView: CameraPreviewNSView, coordinator: ()) {
        nsView.tearDown()
    }
}

@MainActor
final class CameraPreviewNSView: NSView {
    let previewLayer = AVCaptureVideoPreviewLayer()
    private(set) var mirrorDisplay: Bool

    init(session: AVCaptureSession?, mirrorDisplay: Bool = true) {
        self.mirrorDisplay = mirrorDisplay
        super.init(frame: .zero)
        wantsLayer = true
        layer = previewLayer
        previewLayer.videoGravity = .resizeAspect
        previewLayer.session = session
        configureMirroring()
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
        configureMirroring()
    }

    func setSession(_ session: AVCaptureSession?) {
        if previewLayer.session !== session {
            previewLayer.session = session
        }
        configureMirroring()
    }

    func setMirrorDisplay(_ mirrorDisplay: Bool) {
        self.mirrorDisplay = mirrorDisplay
        configureMirroring()
    }

    func tearDown() {
        previewLayer.session = nil
        previewLayer.removeFromSuperlayer()
        layer = nil
    }

    private func configureMirroring() {
        CameraDisplayPolicy(mirrorDisplay: mirrorDisplay).configure(
            previewLayer.connection,
            for: .preview
        )
    }
}
