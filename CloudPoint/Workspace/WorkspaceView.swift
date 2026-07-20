import SwiftUI

struct WorkspaceView: View {
    var body: some View {
        ContentUnavailableView(
            "Create a 3D map",
            systemImage: "point.3.connected.trianglepath.dotted",
            description: Text("Open a recording or use a camera.")
        )
        .frame(minWidth: 960, minHeight: 640)
    }
}
