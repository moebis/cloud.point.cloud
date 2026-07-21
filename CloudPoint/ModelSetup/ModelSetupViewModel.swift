import Foundation

@MainActor
final class ModelSetupViewModel: ObservableObject {
    @Published private(set) var health: ModelHealth = .absent
    @Published private(set) var isPreparing = false
    @Published private(set) var errorMessage: String?

    let installer: any ModelInstalling

    init(installer: any ModelInstalling) {
        self.installer = installer
    }

    func refresh() async { health = await installer.health() }

    func prepare() async {
        guard !isPreparing else { return }
        isPreparing = true
        errorMessage = nil
        defer { isPreparing = false }
        do {
            let stream = await installer.prepare()
            for try await event in stream { health = .preparing(event) }
            health = await installer.health()
        } catch is CancellationError {
            health = await installer.health()
        } catch {
            let message = Self.message(for: error)
            errorMessage = message
            health = .invalid(.operationFailed(message))
        }
    }

    func cancel() { Task { await installer.cancel() } }

    var stageTitle: String {
        switch health {
        case .absent: "Reconstruction model required"
        case let .preparing(event):
            switch event {
            case .downloading: "Downloading model"
            case .verifying: "Verifying download"
            case let .converting(phase):
                switch phase {
                case .verifying: "Verifying trusted checkpoint"
                case .restrictedLoading, .trustedArtifactLoading: "Loading trusted checkpoint"
                case .converting: "Converting for Apple MLX"
                case .validating: "Validating converted model"
                case .ready: "Finishing model setup"
                }
            case .validating: "Validating converted model"
            case .ready: "Model ready"
            }
        case .ready: "Model ready"
        case .invalid: "Model repair required"
        }
    }

    var progressFraction: Double? {
        guard case let .preparing(event) = health else { return nil }
        switch event {
        case let .downloading(received, expected), let .verifying(received, expected):
            guard expected > 0 else { return nil }
            return min(max(Double(received) / Double(expected), 0), 1)
        case .converting, .validating, .ready: return nil
        }
    }

    var isReady: Bool {
        if case .ready = health { true } else { false }
    }

    private static func message(for error: Error) -> String {
        switch error {
        case ModelInstallerError.sourceChecksumMismatch:
            "The downloaded checkpoint did not match the pinned Lingbot Map release. Try again."
        case let ModelInstallerError.converterFailed(message): message
        default: "Model setup could not finish. Check your connection and available disk space, then retry."
        }
    }
}

@MainActor
final class SharpModelSetupViewModel: ObservableObject {
    @Published private(set) var health: SharpModelHealth = .absent
    @Published private(set) var isPreparing = false
    @Published private(set) var errorMessage: String?
    @Published var hasAcceptedLicense = false

    let installer: any SharpModelInstalling
    let licenseText: String

    init(installer: any SharpModelInstalling, licenseText: String) {
        self.installer = installer
        self.licenseText = licenseText
    }

    func refresh() async { health = await installer.health() }

    func prepare() async {
        guard !isPreparing else { return }
        isPreparing = true
        errorMessage = nil
        defer { isPreparing = false }
        do {
            let stream = await installer.prepare(acceptingLicense: hasAcceptedLicense)
            for try await event in stream { health = .preparing(event) }
            health = await installer.health()
        } catch is CancellationError {
            health = await installer.health()
        } catch {
            errorMessage = error.localizedDescription
            health = .invalid
        }
    }

    func cancel() { Task { await installer.cancel() } }

    var stageTitle: String {
        switch health {
        case .absent: "Apple SHARP model required"
        case let .preparing(event):
            switch event {
            case .downloading: "Downloading SHARP"
            case .verifying: "Verifying Apple checkpoint"
            case .publishing: "Publishing verified model"
            case .ready: "SHARP ready"
            }
        case .ready: "SHARP ready"
        case .invalid: "SHARP model repair required"
        }
    }

    var progressFraction: Double? {
        guard case let .preparing(event) = health else { return nil }
        switch event {
        case let .downloading(received, expected), let .verifying(received, expected):
            guard expected > 0 else { return nil }
            return min(max(Double(received) / Double(expected), 0), 1)
        case .publishing, .ready: return nil
        }
    }

    var isReady: Bool {
        if case .ready = health { true } else { false }
    }
}
