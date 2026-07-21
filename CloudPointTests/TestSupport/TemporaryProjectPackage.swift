import Foundation

final class TemporaryProjectPackage {
    let url: URL

    private init(url: URL) {
        self.url = url
    }

    deinit {
        try? FileManager.default.removeItem(at: url)
    }

    static func make() throws -> TemporaryProjectPackage {
        let url = FileManager.default.temporaryDirectory
            .appending(path: UUID().uuidString)
            .appendingPathExtension("cloudpoint")
        let fileManager = FileManager.default

        try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
        for directory in ["Frames", "Predictions", "Points", "Outputs/Gaussians", "Logs"] {
            try fileManager.createDirectory(
                at: url.appending(path: directory),
                withIntermediateDirectories: true
            )
        }

        return TemporaryProjectPackage(url: url)
    }
}
