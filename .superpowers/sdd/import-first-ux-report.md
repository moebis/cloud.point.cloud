# Import-first native UX report

Date: 2026-07-21

## Outcome

CloudPoint now launches as a native `WindowGroup` into a branded welcome screen instead of creating an untitled document. `AppCoordinator` owns every MOV, MP4, M4V, and `.cloudpoint` route. Opening a movie validates it with AVFoundation, creates a durable managed project under Application Support, records it in recents, and passes the initial source into the workspace without a Save dialog.

The managed store stages complete packages before an exclusive atomic rename. Each package contains `Manifest.json`, `Frames`, `Predictions`, `Points`, and `Logs`; the recent-project index is also replaced atomically. Existing packages can be reopened and their committed manifest/state is reflected in the welcome screen.

Movie Finder registrations use `LSHandlerRank=Alternate`, while `CloudPointDocument.readableContentTypes` remains limited to the CloudPoint package type. The mock engine remains reachable only from a DEBUG build with the exact `--mock-engine` argument; normal and release launches do not select it.

## Test-driven evidence

The first focused run failed at compile time because the new tests required managed-project construction and routing APIs that did not exist, including `CloudPointDocument(manifest:)`. After implementation:

- Focused import-first suite: 27 tests passed, 0 failures.
- Full `scripts/test-native`: 252 tests passed, 2 intentional manual/opt-in skips, 0 failures.
- `scripts/test-recording-smoke /Users/moebis/Downloads/IMG_2285.MOV`: passed. The supplied 10.6-second 4K HEVC recording decoded through AVFoundation, sampled frames, and completed the portable artifact-validation pipeline without changing the source file.
- Manual app route: selecting `IMG_2285.MOV` from the welcome screen created an `IMG_2285-<uuid>.cloudpoint` package under the sandboxed Application Support `CloudPoint/Projects` directory and opened the workspace with no save prerequisite.

## UI validation

The welcome screen was inspected at the normal 1080x760 window and near its compact minimum. The generated CloudPoint application icon is used in the header; button contrast, compact spacing, recent-project empty state, engine status, and the drag/drop affordance remain legible without clipping. The supplied MOV was also opened through the native Open Video panel and produced the expected `IMG_2285` workspace title.

## Scope boundary

This milestone establishes the import-first shell and durable project lifecycle. Production MLX model acquisition, conversion, and worker-engine readiness are separate model/runtime work. With that runtime absent, the production workspace fails closed with the existing engine-unavailable message and never substitutes the deterministic mock.
