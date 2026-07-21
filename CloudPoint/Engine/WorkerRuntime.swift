import Foundation

enum WorkerRuntimeError: Error, Equatable, Sendable {
    case buildSettingMissing
    case runtimeMustBeAbsolute(String)
    case executableMissing(String)
}

struct WorkerRuntime: Sendable, Equatable {
    let root: URL
    let pythonExecutable: URL
    let workerExecutable: URL
    let modelExecutable: URL

    static func resolve(
        bundleValue: String?,
        environment: [String: String],
        bundledResourcesURL: URL? = Bundle.main.resourceURL,
        validateFiles: Bool = true,
        fileManager: FileManager = .default
    ) throws -> Self {
        // Deliberately ignore PATH and every inherited environment lookup. The
        // app must launch its bundled runtime or the exact development runtime
        // selected at build/setup time.
        _ = environment

        if let bundledResourcesURL {
            let bundledRoot = bundledResourcesURL
                .appending(path: "WorkerRuntime", directoryHint: .isDirectory)
                .standardizedFileURL
            if fileManager.fileExists(atPath: bundledRoot.path) {
                return try make(
                    root: bundledRoot,
                    validateFiles: validateFiles,
                    fileManager: fileManager
                )
            }
        }

        guard let raw = bundleValue?.trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else {
            throw WorkerRuntimeError.buildSettingMissing
        }
        guard raw.hasPrefix("/") else {
            throw WorkerRuntimeError.runtimeMustBeAbsolute(raw)
        }

        let root = URL(filePath: raw, directoryHint: .isDirectory).standardizedFileURL
        return try make(
            root: root,
            validateFiles: validateFiles,
            fileManager: fileManager
        )
    }

    static func unchecked(root: URL) -> Self {
        let root = root.standardizedFileURL
        return Self(
            root: root,
            pythonExecutable: root.appending(path: "bin/python3.12"),
            workerExecutable: root.appending(path: "bin/cloudpoint-worker"),
            modelExecutable: root.appending(path: "bin/cloudpoint-model")
        )
    }

    private static func make(
        root: URL,
        validateFiles: Bool,
        fileManager: FileManager
    ) throws -> Self {
        let runtime = unchecked(root: root)
        if validateFiles {
            for executable in [
                runtime.pythonExecutable,
                runtime.workerExecutable,
                runtime.modelExecutable,
            ] {
                var isDirectory: ObjCBool = false
                guard fileManager.fileExists(
                    atPath: executable.path,
                    isDirectory: &isDirectory
                ), !isDirectory.boolValue,
                      fileManager.isExecutableFile(atPath: executable.path) else {
                    throw WorkerRuntimeError.executableMissing(executable.path)
                }
            }
        }
        return runtime
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
