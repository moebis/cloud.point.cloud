import AppKit
import SwiftUI
import XCTest
@testable import CloudPoint
import UniformTypeIdentifiers

@MainActor
final class WorkspaceViewModelTests: XCTestCase {
    func testRecordingProgressUsesHumanStagesAndStableExpectedTotal() {
        let reading = WorkspaceProgressPresentation.make(
            phase: .importing,
            source: .recording,
            sampledCount: 6,
            queuedCount: 5,
            processedCount: 3,
            expectedCount: 22
        )
        let reconstructing = WorkspaceProgressPresentation.make(
            phase: .processing,
            source: .recording,
            sampledCount: 22,
            queuedCount: 22,
            processedCount: 9,
            expectedCount: 22
        )

        XCTAssertEqual(reading.title, "Reading video")
        XCTAssertEqual(reading.completedCount, 6)
        XCTAssertEqual(reading.totalCount, 22)
        XCTAssertEqual(reconstructing.title, "Reconstructing scene")
        XCTAssertEqual(reconstructing.completedCount, 9)
        XCTAssertEqual(reconstructing.totalCount, 22)
    }

    func testCameraProgressNeverUsesRecordingControlsOrLabels() {
        let preflight = WorkspaceProgressPresentation.make(
            phase: .ready,
            source: .cameraPreflight,
            sampledCount: 0,
            queuedCount: 0,
            processedCount: 0,
            expectedCount: nil
        )
        let draining = WorkspaceProgressPresentation.make(
            phase: .processing,
            source: .camera,
            sampledCount: 18,
            queuedCount: 18,
            processedCount: 11,
            expectedCount: nil
        )

        XCTAssertEqual(preflight.title, "Ready to capture")
        XCTAssertEqual(draining.title, "Processing remaining camera frames")
        XCTAssertEqual(draining.totalCount, 18)
    }

    func testCameraInitialSourceSelectsPreflightDeviceWithoutStartingCapture() throws {
        let package = try TemporaryProjectPackage.make()
        let manifest = ProjectManifest(
            cameraSource: CameraSourceReference(
                deviceID: "camera-7",
                deviceName: "Studio Camera"
            )
        )
        try manifest.writeAtomically(to: package.url)

        let viewModel = WorkspaceViewModel(
            document: CloudPointDocument(manifest: manifest),
            packageURL: package.url,
            initialSource: .camera(deviceID: "camera-7", deviceName: "Studio Camera"),
            arguments: []
        )

        XCTAssertEqual(viewModel.sourceMode, .cameraPreflight)
        XCTAssertEqual(viewModel.selectedCameraID, "camera-7")
        XCTAssertFalse(viewModel.snapshot.isCapturing)
        XCTAssertTrue(viewModel.requiresCloseConfirmation == false)
    }

    func testMissingRecordingBookmarkOffersLocateOriginalRecovery() async throws {
        let package = try TemporaryProjectPackage.make()
        let source = RecordingSourceReference(
            bookmarkData: Data("missing".utf8),
            originalFilename: "Missing.mov",
            fingerprint: RecordingSourceFingerprint(
                byteCount: 42,
                sha256: String(repeating: "c", count: 64)
            ),
            durationSeconds: 2,
            framesPerSecond: 2,
            expectedSampleCount: 4,
            nextSampleOrdinal: 0
        )
        let manifest = ProjectManifest(recordingSource: source)
        try manifest.writeAtomically(to: package.url)
        let viewModel = WorkspaceViewModel(
            document: CloudPointDocument(manifest: manifest),
            packageURL: package.url,
            recordingSources: FailingRecordingSourceManager(),
            arguments: ["CloudPoint", "--mock-engine"]
        )

        viewModel.start()
        let didOfferRecovery = await waitUntil {
            viewModel.recoveryAction == .locateOriginal
        }

        XCTAssertTrue(didOfferRecovery)
        XCTAssertEqual(viewModel.recoveryAction?.title, "Locate Original…")
        XCTAssertEqual(
            viewModel.presentedErrorText,
            RecordingSourceAccessError.unavailable.localizedDescription
        )
    }

    func testUnavailableEngineOffersRepairModelActionHook() async throws {
        let package = try TemporaryProjectPackage.make()
        let manifest = ProjectManifest()
        try manifest.writeAtomically(to: package.url)
        let tracker = RepairActionTracker()
        let viewModel = WorkspaceViewModel(
            document: CloudPointDocument(manifest: manifest),
            packageURL: package.url,
            onRepairModel: { tracker.callCount += 1 },
            arguments: []
        )

        viewModel.start()
        let didOfferRepair = await waitUntil {
            viewModel.recoveryAction == .repairModel
        }
        viewModel.performRecoveryAction()

        XCTAssertTrue(didOfferRepair)
        XCTAssertEqual(tracker.callCount, 1)
    }

    func testManagedWorkspaceHasNoSavePrerequisite() throws {
        let package = try TemporaryProjectPackage.make()
        let manifest = ProjectManifest()
        try manifest.writeAtomically(to: package.url)
        let viewModel = WorkspaceViewModel(
            document: CloudPointDocument(manifest: manifest),
            packageURL: package.url,
            arguments: []
        )

        XCTAssertNil(viewModel.snapshot.setupText)
    }

    func testMockEngineRequiresExactDebugFlag() {
#if DEBUG
        XCTAssertTrue(WorkspaceViewModel.shouldUseMockEngine(arguments: ["CloudPoint", "--mock-engine"]))
#else
        XCTAssertFalse(WorkspaceViewModel.shouldUseMockEngine(arguments: ["CloudPoint", "--mock-engine"]))
#endif
        XCTAssertFalse(WorkspaceViewModel.shouldUseMockEngine(arguments: ["CloudPoint"]))
        XCTAssertFalse(WorkspaceViewModel.shouldUseMockEngine(arguments: ["CloudPoint", "--mock-engine-extra"]))
    }

    func testWindowCloseGuardDoesNotAdvertisePreviousDelegateOptionalSelectors() async throws {
        let package = try TemporaryProjectPackage.make()
        var manifest = ProjectManifest()
        manifest.sessionState = SessionState(phase: .capturing, isCapturing: true)
        try manifest.writeAtomically(to: package.url)
        let viewModel = WorkspaceViewModel(
            document: CloudPointDocument(manifest: manifest),
            packageURL: package.url,
            arguments: []
        )
        let previousDelegate = OptionalWindowDelegate()
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 900, height: 650),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window.delegate = previousDelegate
        window.contentView = NSHostingView(rootView: WorkspaceView(viewModel: viewModel))
        window.orderFront(nil)
        defer {
            window.delegate = previousDelegate
            window.orderOut(nil)
            window.contentView = nil
        }

        let didAttachGuard = await waitUntil { window.delegate !== previousDelegate }
        let optionalSelector = #selector(NSWindowDelegate.windowDidResize(_:))

        XCTAssertTrue(didAttachGuard)
        XCTAssertTrue(previousDelegate.responds(to: optionalSelector))
        XCTAssertFalse(window.delegate?.responds(to: optionalSelector) == true)
        await viewModel.close()
    }

    func testWindowCloseGuardRestoresPreviousDelegateWhenDisabled() async throws {
        let activePackage = try TemporaryProjectPackage.make()
        var activeManifest = ProjectManifest()
        activeManifest.sessionState = SessionState(phase: .capturing, isCapturing: true)
        try activeManifest.writeAtomically(to: activePackage.url)
        let activeViewModel = WorkspaceViewModel(
            document: CloudPointDocument(manifest: activeManifest),
            packageURL: activePackage.url,
            arguments: []
        )
        let completedPackage = try TemporaryProjectPackage.make()
        var completedManifest = ProjectManifest()
        completedManifest.sessionState = SessionState(phase: .completed)
        try completedManifest.writeAtomically(to: completedPackage.url)
        let completedViewModel = WorkspaceViewModel(
            document: CloudPointDocument(manifest: completedManifest),
            packageURL: completedPackage.url,
            arguments: []
        )
        let previousDelegate = OptionalWindowDelegate()
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 900, height: 650),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        let host = NSHostingView(rootView: WorkspaceView(viewModel: activeViewModel))
        window.delegate = previousDelegate
        window.contentView = host
        window.orderFront(nil)
        defer {
            window.delegate = previousDelegate
            window.orderOut(nil)
            window.contentView = nil
        }

        let didAttachGuard = await waitUntil { window.delegate !== previousDelegate }
        XCTAssertTrue(didAttachGuard)

        host.rootView = WorkspaceView(viewModel: completedViewModel)

        let didRestorePreviousDelegate = await waitUntil {
            window.delegate === previousDelegate
        }
        XCTAssertTrue(didRestorePreviousDelegate)
        await activeViewModel.close()
        await completedViewModel.close()
    }

    func testConfirmedWindowCloseStillClosesAfterGuardDisables() async {
        let previousDelegate = OptionalWindowDelegate()
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 900, height: 650),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window.isReleasedWhenClosed = false
        var confirmedClose: (() -> Void)?
        let host = NSHostingView(rootView: WorkspaceWindowCloseGuard(isEnabled: true) {
            confirmedClose = $0
        })
        window.delegate = previousDelegate
        window.contentView = host
        window.orderFront(nil)
        host.rootView = WorkspaceWindowCloseGuard(isEnabled: true) {
            confirmedClose = $0
        }
        defer {
            window.delegate = previousDelegate
            window.orderOut(nil)
            window.contentView = nil
        }

        let didAttachGuard = await waitUntil { window.delegate !== previousDelegate }
        guard didAttachGuard else {
            return XCTFail("Expected the active close guard to attach")
        }

        window.performClose(nil)

        XCTAssertTrue(window.isVisible)
        XCTAssertNotNil(confirmedClose)

        host.rootView = WorkspaceWindowCloseGuard(isEnabled: false) { _ in }
        let didRestorePreviousDelegate = await waitUntil {
            window.delegate === previousDelegate
        }
        XCTAssertTrue(didRestorePreviousDelegate)

        confirmedClose?()

        let didClose = await waitUntil { !window.isVisible }
        XCTAssertTrue(didClose)
    }

}

@MainActor
private func waitUntil(
    timeout: Duration = .seconds(2),
    condition: @escaping @MainActor () -> Bool
) async -> Bool {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: timeout)
    while clock.now < deadline {
        if condition() { return true }
        try? await Task.sleep(for: .milliseconds(10))
    }
    return condition()
}

private struct FailingRecordingSourceManager: RecordingSourceManaging {
    func makeReference(
        for url: URL,
        probe: VideoProbeResult,
        framesPerSecond: Int
    ) async throws -> RecordingSourceReference {
        throw RecordingSourceAccessError.unavailable
    }

    func resolve(_ reference: RecordingSourceReference) async throws -> URL {
        throw RecordingSourceAccessError.unavailable
    }

    func replacement(
        for url: URL,
        preserving reference: RecordingSourceReference
    ) async throws -> RecordingSourceReference {
        throw RecordingSourceAccessError.unavailable
    }
}

@MainActor
private final class RepairActionTracker {
    var callCount = 0
}

@MainActor
private final class OptionalWindowDelegate: NSObject, NSWindowDelegate {
    func windowDidResize(_ notification: Notification) {}
}
