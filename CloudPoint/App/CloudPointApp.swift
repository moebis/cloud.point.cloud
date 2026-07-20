import SwiftUI

@main
struct CloudPointApp: App {
    var body: some Scene {
        DocumentGroup(newDocument: { CloudPointDocument() }) { configuration in
            WorkspaceView(
                document: configuration.document,
                packageURL: configuration.fileURL
            )
        }
    }
}
