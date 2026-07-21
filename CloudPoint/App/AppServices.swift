import Foundation

struct ProductionReconstructionContext: Sendable, Equatable {
    let runtime: WorkerRuntime
    let modelDirectory: URL

    func makeEngine() throws -> any ReconstructionEngine {
        try PythonMLXEngineFactory(runtime: runtime).makeEngine(
            modelDirectory: modelDirectory
        )
    }
}

struct SharpProductionReconstructionContext: Sendable, Equatable {
    let runtime: WorkerRuntime
    let installation: SharpModelInstallation

    func makeEngine() -> any ReconstructionEngine {
        SharpReconstructionEngineFactory(
            runtime: runtime,
            installation: installation
        ).makeEngine()
    }
}
