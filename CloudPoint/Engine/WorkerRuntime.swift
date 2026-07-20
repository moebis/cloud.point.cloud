import Foundation

enum WorkerRuntimeError: Error, Equatable, Sendable {
    case buildSettingMissing
    case runtimeMustBeAbsolute(String)
    case executableMissing(String)
}

struct WorkerRuntime: Sendable, Equatable {
    let root: URL
    let workerExecutable: URL
    let modelExecutable: URL

    static func resolve(
        bundleValue: String?,
        environment: [String: String],
        validateFiles: Bool = true,
        fileManager: FileManager = .default
    ) throws -> Self {
        // Deliberately ignore PATH and every inherited environment lookup. The
        // app must launch the one runtime selected at build/setup time.
        _ = environment
        guard let raw = bundleValue?.trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else {
            throw WorkerRuntimeError.buildSettingMissing
        }
        guard raw.hasPrefix("/") else {
            throw WorkerRuntimeError.runtimeMustBeAbsolute(raw)
        }

        let root = URL(filePath: raw, directoryHint: .isDirectory).standardizedFileURL
        let worker = root.appending(path: "bin/cloudpoint-worker")
        let model = root.appending(path: "bin/cloudpoint-model")
        if validateFiles {
            for executable in [worker, model]
            where !fileManager.isExecutableFile(atPath: executable.path) {
                throw WorkerRuntimeError.executableMissing(executable.path)
            }
        }
        return Self(root: root, workerExecutable: worker, modelExecutable: model)
    }

    static func unchecked(root: URL) -> Self {
        let root = root.standardizedFileURL
        return Self(
            root: root,
            workerExecutable: root.appending(path: "bin/cloudpoint-worker"),
            modelExecutable: root.appending(path: "bin/cloudpoint-model")
        )
    }
}

struct WorkerLaunch: Sendable, Equatable {
    let executable: URL
    let arguments: [String]

    init(runtime: WorkerRuntime, project: URL, model: URL) throws {
        for url in [project, model] {
            guard url.isFileURL, url.path.hasPrefix("/") else {
                throw WorkerRuntimeError.runtimeMustBeAbsolute(
                    url.isFileURL ? url.path : url.absoluteString
                )
            }
        }
        executable = runtime.workerExecutable
        arguments = [
            "serve",
            "--project", project.standardizedFileURL.path,
            "--model", model.standardizedFileURL.path,
        ]
    }
}
