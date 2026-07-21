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
