import SwiftUI

@main
struct CloudPointApp: App {
    @StateObject private var coordinator = AppCoordinator.live()

    var body: some Scene {
        WindowGroup("CloudPoint") {
            CloudPointRootView(coordinator: coordinator)
        }
        .defaultSize(width: 1_080, height: 760)
        .restorationBehavior(.disabled)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open Video…") { coordinator.chooseVideo() }
                    .keyboardShortcut("o", modifiers: .command)
                Button("Use Camera") { Task { await coordinator.useCamera() } }
                Divider()
                Button("Open CloudPoint Project…") { coordinator.chooseProject() }
                    .keyboardShortcut("o", modifiers: [.command, .option])
            }
        }
    }
}

private struct CloudPointRootView: View {
    @ObservedObject var coordinator: AppCoordinator

    var body: some View {
        Group {
            switch coordinator.destination {
            case .welcome:
                WelcomeView(coordinator: coordinator)

            case let .workspace(launch):
                let viewModel = coordinator.workspaceViewModel(for: launch)
                WorkspaceView(
                    viewModel: viewModel,
                    sourceTitle: launch.sourceTitle,
                    onOpenVideo: coordinator.chooseVideo,
                    onShowWelcome: { Task { await coordinator.showWelcome() } }
                )
                .id(launch.id)
            }
        }
        .task { await coordinator.start() }
        .onOpenURL { url in
            Task { await coordinator.openExternalURL(url) }
        }
        .handlesExternalEvents(preferring: ["*"], allowing: ["*"])
        .sheet(isPresented: $coordinator.isModelSetupPresented) {
            if let model = coordinator.modelSetupViewModel {
                ModelSetupView(model: model) {
                    Task { await coordinator.continueAfterModelSetup() }
                }
            }
        }
        .sheet(item: Binding(
            get: { coordinator.pendingReconstruction },
            set: { value in
                if value == nil { coordinator.cancelPendingReconstruction() }
            }
        )) { request in
            NewReconstructionView(coordinator: coordinator, request: request)
        }
    }
}
