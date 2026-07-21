import AppKit
import SwiftUI

struct ModelSetupView: View {
    @ObservedObject var model: ModelSetupViewModel
    let onContinue: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 24) {
            HStack(spacing: 16) {
                Image(nsImage: NSApplication.shared.applicationIconImage)
                    .resizable()
                    .frame(width: 64, height: 64)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Prepare CloudPoint")
                        .font(.title2.weight(.semibold))
                    Text("One-time setup for real 3D reconstruction on Apple Silicon")
                        .foregroundStyle(.secondary)
                }
            }

            VStack(alignment: .leading, spacing: 10) {
                Label("Pinned Lingbot Map model", systemImage: "checkmark.shield")
                    .font(.headline)
                LabeledContent("Source", value: "robbyant/lingbot-map")
                LabeledContent("Download", value: "4.32 GiB")
                LabeledContent("Revision", value: "204754b7…")
            }
            .padding(16)
            .background(.quaternary.opacity(0.45), in: RoundedRectangle(cornerRadius: 14))

            Text("CloudPoint verifies the exact upstream checkpoint, then converts it locally into Apple MLX weights. The checkpoint and converted weights are not redistributed. Allow roughly 8 GiB of temporary free space during setup.")
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            VStack(alignment: .leading, spacing: 8) {
                Text(model.stageTitle).font(.headline)
                if let progress = model.progressFraction {
                    ProgressView(value: progress)
                    Text(progress.formatted(.percent.precision(.fractionLength(0))))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                } else if model.isPreparing {
                    ProgressView().controlSize(.small)
                }
                if let error = model.errorMessage {
                    Label(error, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            HStack {
                if model.isPreparing {
                    Button("Cancel") { model.cancel() }
                }
                Spacer()
                if model.isReady {
                    Button("Continue", action: onContinue)
                        .buttonStyle(.borderedProminent)
                        .keyboardShortcut(.defaultAction)
                } else {
                    Button(model.errorMessage == nil ? "Download and Prepare" : "Retry") {
                        Task { await model.prepare() }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.isPreparing)
                }
            }
        }
        .padding(28)
        .frame(width: 560)
        .task { await model.refresh() }
    }
}
