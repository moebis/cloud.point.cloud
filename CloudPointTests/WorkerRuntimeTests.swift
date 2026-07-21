import Foundation
import XCTest
@testable import CloudPoint

final class WorkerRuntimeTests: XCTestCase {
    func testBundledRuntimeTakesPrecedenceOverDeveloperBuildSetting() throws {
        let resources = try makeResourcesWithRuntime()

        let runtime = try WorkerRuntime.resolve(
            bundleValue: "/Users/developer/cloudpoint/worker/.venv",
            environment: ["PATH": "/usr/local/bin:/usr/bin"],
            bundledResourcesURL: resources
        )

        XCTAssertEqual(
            runtime.root,
            resources.appending(path: "WorkerRuntime", directoryHint: .isDirectory)
                .standardizedFileURL
        )
        XCTAssertEqual(
            runtime.pythonExecutable,
            runtime.root.appending(path: "bin/python3.12")
        )
    }

    func testMalformedBundledRuntimeDoesNotFallBackToDeveloperBuildSetting() throws {
        let resources = try makeResourcesWithRuntime(excluding: "cloudpoint-model")
        let missingExecutable = resources
            .appending(path: "WorkerRuntime/bin/cloudpoint-model")

        XCTAssertThrowsError(try WorkerRuntime.resolve(
            bundleValue: "/tmp/otherwise-valid-developer-runtime",
            environment: [:],
            bundledResourcesURL: resources
        )) { error in
            XCTAssertEqual(
                error as? WorkerRuntimeError,
                .executableMissing(missingExecutable.path)
            )
        }
    }

    func testRuntimeRejectsMissingPackagedPython() throws {
        let resources = try makeResourcesWithRuntime(excluding: "python3.12")
        let missingExecutable = resources
            .appending(path: "WorkerRuntime/bin/python3.12")

        XCTAssertThrowsError(try WorkerRuntime.resolve(
            bundleValue: nil,
            environment: [:],
            bundledResourcesURL: resources
        )) { error in
            XCTAssertEqual(
                error as? WorkerRuntimeError,
                .executableMissing(missingExecutable.path)
            )
        }
    }

    func testMissingBuildSettingNeverFallsBackToPATH() {
        XCTAssertThrowsError(try WorkerRuntime.resolve(
            bundleValue: nil,
            environment: ["PATH": "/usr/local/bin:/usr/bin"],
            validateFiles: false
        )) { error in
            XCTAssertEqual(error as? WorkerRuntimeError, .buildSettingMissing)
        }
    }

    func testRelativeBuildSettingIsRejectedBeforeURLResolution() {
        XCTAssertThrowsError(try WorkerRuntime.resolve(
            bundleValue: "worker/.venv",
            environment: [:],
            validateFiles: false
        )) { error in
            XCTAssertEqual(
                error as? WorkerRuntimeError,
                .runtimeMustBeAbsolute("worker/.venv")
            )
        }
    }

    func testRuntimeUsesExactManagedExecutables() throws {
        let root = URL(filePath: "/tmp/cloudpoint-worker-runtime", directoryHint: .isDirectory)
        let runtime = try WorkerRuntime.resolve(
            bundleValue: root.path,
            environment: [:],
            validateFiles: false
        )

        XCTAssertEqual(runtime.root, root.standardizedFileURL)
        XCTAssertEqual(
            runtime.workerExecutable,
            root.appending(path: "bin/cloudpoint-worker")
        )
        XCTAssertEqual(
            runtime.modelExecutable,
            root.appending(path: "bin/cloudpoint-model")
        )
    }

    func testServeArgumentsContainOnlyResolvedAbsoluteFilePaths() throws {
        let launch = try WorkerLaunch(
            runtime: .unchecked(
                root: URL(filePath: "/opt/cloudpoint/.venv", directoryHint: .isDirectory)
            ),
            project: URL(filePath: "/tmp/Test.cloudpoint", directoryHint: .isDirectory),
            model: URL(filePath: "/tmp/model", directoryHint: .isDirectory)
        )

        XCTAssertEqual(launch.executable.path, "/opt/cloudpoint/.venv/bin/cloudpoint-worker")
        XCTAssertEqual(launch.arguments, [
            "serve", "--project", "/tmp/Test.cloudpoint",
            "--model", "/tmp/model",
        ])
    }

    func testServeLaunchRejectsNonFileURLs() {
        XCTAssertThrowsError(try WorkerLaunch(
            runtime: .unchecked(
                root: URL(filePath: "/opt/cloudpoint/.venv", directoryHint: .isDirectory)
            ),
            project: URL(string: "https://example.com/project.cloudpoint")!,
            model: URL(filePath: "/tmp/model", directoryHint: .isDirectory)
        )) { error in
            XCTAssertEqual(
                error as? WorkerRuntimeError,
                .runtimeMustBeAbsolute("https://example.com/project.cloudpoint")
            )
        }
    }

    private func makeResourcesWithRuntime(excluding excludedName: String? = nil) throws -> URL {
        let resources = FileManager.default.temporaryDirectory
            .appending(path: "CloudPoint-WorkerRuntimeTests-\(UUID().uuidString)", directoryHint: .isDirectory)
        let bin = resources.appending(path: "WorkerRuntime/bin", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: resources)
        }

        for name in ["python3.12", "cloudpoint-worker", "cloudpoint-model"]
        where name != excludedName {
            let executable = bin.appending(path: name)
            XCTAssertTrue(FileManager.default.createFile(
                atPath: executable.path,
                contents: Data("#!/bin/sh\nexit 0\n".utf8)
            ))
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o755],
                ofItemAtPath: executable.path
            )
        }
        return resources
    }
}
