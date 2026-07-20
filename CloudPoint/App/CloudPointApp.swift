import SwiftUI

@main
struct CloudPointApp: App {
    @StateObject private var coordinator = AppCoordinator.live()

    var body: some Scene {
        WindowGroup("CloudPoint") {
            CloudPointRootView(coordinator: coordinator)
        }
        .defaultSize(width: 1_080, height: 760)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open Video…") { coordinator.chooseVideo() }
                    .keyboardShortcut("o", modifiers: .command)
                Button("Use Camera") { coordinator.useCamera() }
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
                WorkspaceView(
                    document: CloudPointDocument(manifest: launch.manifest),
                    packageURL: launch.packageURL,
                    initialSource: launch.initialSource,
                    sourceTitle: launch.sourceTitle,
                    onOpenVideo: coordinator.chooseVideo,
                    onShowWelcome: coordinator.showWelcome
                )
                .id(launch.id)
            }
        }
        .task { await coordinator.start() }
        .onOpenURL { url in
            Task { await coordinator.openExternalURL(url) }
        }
    }
}
