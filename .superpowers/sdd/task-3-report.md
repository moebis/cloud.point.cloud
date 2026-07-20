# Task 3: Reconstruction Engine Contract and Deterministic Mock

## RED

1. Added `MockReconstructionEngineTests.testMockEmitsOneDeterministicChunkPerFrameThenCompletes` before either engine source file existed. The test creates a real `TemporaryProjectPackage`, starts the actor, obtains its synchronous event stream, submits frame 7, and reads the written CPC file.
2. The first focused `xcodebuild` selection executed zero tests because the generated Xcode project predated the new source file. Regenerated from `project.yml` with `scripts/generate-project` and reran the exact selection.
3. The regenerated test build failed as intended because `EngineConfiguration` and `ProjectDescriptor` did not exist. This established the missing engine boundary rather than a fixture or test-harness failure.

## GREEN

1. Added the approved `ReconstructionEngine` protocol with the synchronous `events() -> AsyncThrowingStream<EngineEvent, Error>` requirement, value payloads, and explicit lifecycle errors.
2. Implemented `MockReconstructionEngine` as an actor. Its event stream is an immutable Sendable nonisolated value, while its continuation, lifecycle, queue, pause state, and project state remain actor-owned.
3. Added a private `MockCPCWriter` that emits the exact 32-byte little-endian CPC1 header and 24-byte `<fff4BeHI` vertex records. The mock writes a deterministic 64-by-64 plane below the project package's `Points` directory. Frame index deterministically selects the RGBA color and Z offset; the emitted `pointChunkPath` stays package-relative.
4. Expanded the focused suite to verify real package files and actor behavior:
   - CPC path, header, full size, first vertex coordinates, color, Float16 confidence, flags, and source frame.
   - Paused frames do not write before resume and preserve completion order after resume.
   - Cancellation emits `.cancelled`, never completes the session, and closes the stream.
   - Invalid begin/enqueue lifecycle calls return the rejected operation.
   - Empty input emits `.sessionCompleted` and closes the stream.
5. While adding the final empty-input assertion, the test compiler correctly rejected an `await` inside XCTest's synchronous autoclosure. The test was fixed by awaiting before `XCTAssertEqual`; no production behavior was altered.

## Verification

- Focused `MockReconstructionEngineTests`: 5 tests passed.
- `git diff --check`: passed.
- `scripts/test-native`: passed all three build schemes and the complete test suite: 16 tests, 0 failures.

## Self-review

- Confirmed the protocol preserves the exact approved method list and the event method is not `async`.
- Confirmed mutable stream continuation is actor-isolated; only the immutable stream is exposed nonisolated.
- Confirmed header fields are explicit little-endian integers/bit patterns: `CPC1`, version 1, stride 24, 4,096 points, source frame range, and eight zero reserved bytes.
- Confirmed tests use no test-only production API and exercise actual temporary files rather than mocking the engine or writer.

## Review follow-up: failure semantics and containment

### RED

1. Added focused tests for a `Points` write failure, a `Points` symlink to an external directory, and `UInt32.max` filename generation.
2. Ran `scripts/generate-project && xcodebuild test -project CloudPoint.xcodeproj -scheme CloudPoint -destination 'platform=macOS,arch=arm64' -only-testing:CloudPointTests/MockReconstructionEngineTests CODE_SIGNING_ALLOWED=NO` before the implementation. It failed with 8 tests and 7 failures: a write failure closed normally after `finishInput` and emitted `sessionCompleted`; a `Points` symlink wrote outside the package; and the maximum unsigned frame generated `frame--0000001.cpc`.

### GREEN

1. The writer now transitions the actor to a terminal failed lifecycle, clears pending frames, and finishes the stream by throwing the original write error.
2. The writer validates the physical package, `Points` directory, and output path after symlink resolution, rejecting package, directory, and output symlinks or escapes.
3. Frame filenames now use `UInt32(exactly:)` with `%08u`, producing the expected unsigned filename for `UInt32.max`.
4. The focused command above passed: 8 tests, 0 failures.
5. `scripts/test-native` passed all three build schemes and the complete suite: 19 tests, 0 failures.

### Follow-up self-review

- Confirmed that a failed write finishes the stream with the same error before returning it to the caller, never emits `sessionCompleted`, and rejects subsequent `finishInput`, `enqueue`, and `resume` calls through the existing lifecycle guards.
- Confirmed output is written only after the real package and `Points` paths resolve to their expected physical locations; a symlinked or non-directory `Points` path is rejected before writing.
- Confirmed `%08u` preserves ordinary zero-padded names and represents the largest accepted UInt32 frame without signed overflow.
