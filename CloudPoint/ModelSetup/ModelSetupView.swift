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

struct SharpModelSetupView: View {
    @ObservedObject var model: SharpModelSetupViewModel
    let onContinue: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack(spacing: 16) {
                Image(systemName: "sparkles.rectangle.stack.fill")
                    .font(.system(size: 36, weight: .semibold))
                    .foregroundStyle(.purple)
                    .frame(width: 58, height: 58)
                    .background(.purple.opacity(0.12), in: RoundedRectangle(cornerRadius: 14))
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Prepare Apple SHARP")
                        .font(.title2.weight(.semibold))
                    Text("Experimental single-image Gaussian reconstruction on Apple silicon")
                        .foregroundStyle(.secondary)
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                Label("Research use only", systemImage: "flask.fill")
                    .font(.headline)
                    .foregroundStyle(.purple)
                Text("The 2.62 GiB checkpoint is downloaded directly from Apple, verified byte-for-byte, and stored only on this Mac. It is not included with CloudPoint.")
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(14)
            .background(.purple.opacity(0.08), in: RoundedRectangle(cornerRadius: 14))

            if !model.isReady {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Apple Machine Learning Research Model License Agreement")
                        .font(.headline)
                    ScrollView {
                        Text(model.licenseText)
                            .font(.caption)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(10)
                    }
                    .frame(height: 180)
                    .background(.background.opacity(0.55), in: RoundedRectangle(cornerRadius: 10))

                    Toggle(
                        "I accept Apple's research-only model license for this personal experiment.",
                        isOn: $model.hasAcceptedLicense
                    )
                    .toggleStyle(.checkbox)
                }
            }

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
                    Button("Create Gaussian Scene", action: onContinue)
                        .buttonStyle(.borderedProminent)
                        .tint(.purple)
                        .keyboardShortcut(.defaultAction)
                } else {
                    Button(model.errorMessage == nil ? "Accept and Download" : "Retry") {
                        Task { await model.prepare() }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.purple)
                    .disabled(model.isPreparing || !model.hasAcceptedLicense)
                }
            }
        }
        .padding(28)
        .frame(width: 640)
        .task { await model.refresh() }
    }
}
