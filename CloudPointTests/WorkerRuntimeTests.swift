import Foundation
import XCTest
@testable import CloudPoint

final class WorkerRuntimeTests: XCTestCase {
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
}
