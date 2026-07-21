import Darwin
import Foundation

enum ModelDownloadError: Error, Sendable, Equatable {
    case alreadyRunning
    case invalidResponse
    case unsafeRedirect
    case missingTemporaryFile
}

final class URLSessionModelDownloader: NSObject, ModelDownloading,
    URLSessionDownloadDelegate, @unchecked Sendable {
    private struct State {
        var task: URLSessionDownloadTask?
        var continuation: CheckedContinuation<URL, Error>?
        var cancelContinuation: CheckedContinuation<Data?, Never>?
        var progress: (@Sendable (Int64, Int64) -> Void)?
        var completedURL: URL?
        var rejectedRedirect = false
    }

    private let lock = NSLock()
    private var state = State()
    private lazy var session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.waitsForConnectivity = true
        configuration.httpCookieStorage = nil
        configuration.httpShouldSetCookies = false
        configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        configuration.timeoutIntervalForResource = 86_400
        let queue = OperationQueue()
        queue.name = "cloud.point.cloud.model-download"
        queue.maxConcurrentOperationCount = 1
        return URLSession(configuration: configuration, delegate: self, delegateQueue: queue)
    }()

    func download(
        request: URLRequest,
        resumeData: Data?,
        progress: @escaping @Sendable (Int64, Int64) -> Void
    ) async throws -> URL {
        guard Self.isAllowed(request.url) else { throw ModelDownloadError.unsafeRedirect }
        return try await withCheckedThrowingContinuation { continuation in
            let task: URLSessionDownloadTask? = lock.withLock {
                guard state.task == nil else { return nil }
                let task = if let resumeData, !resumeData.isEmpty {
                    session.downloadTask(withResumeData: resumeData)
                } else {
                    session.downloadTask(with: request)
                }
                state.task = task
                state.continuation = continuation
                state.progress = progress
                state.completedURL = nil
                state.rejectedRedirect = false
                return task
            }
            guard let task else {
                continuation.resume(throwing: ModelDownloadError.alreadyRunning)
                return
            }
            task.resume()
        }
    }

    func cancel() async -> Data? {
        let task = lock.withLock { state.task }
        guard let task else { return nil }
        return await withCheckedContinuation { continuation in
            let shouldCancel = lock.withLock {
                guard state.task === task else { return false }
                state.cancelContinuation = continuation
                return true
            }
            guard shouldCancel else {
                continuation.resume(returning: nil)
                return
            }
            task.cancel(byProducingResumeData: { [weak self] data in
                self?.finishCancellation(with: data)
            })
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        _ = session
        _ = response
        let allowed = Self.isAllowed(request.url)
        if !allowed {
            lock.withLock {
                if state.task === task { state.rejectedRedirect = true }
            }
        }
        completionHandler(allowed ? request : nil)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        _ = session
        _ = bytesWritten
        let callback = lock.withLock { () -> (@Sendable (Int64, Int64) -> Void)? in
            guard state.task === downloadTask else { return nil }
            return state.progress
        }
        callback?(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        _ = session
        guard let response = downloadTask.response as? HTTPURLResponse,
              [200, 206].contains(response.statusCode),
              Self.isAllowed(response.url) else {
            complete(task: downloadTask, result: .failure(ModelDownloadError.invalidResponse))
            return
        }
        let destination = FileManager.default.temporaryDirectory.appending(
            path: "cloudpoint-model-download-\(UUID().uuidString.lowercased())"
        )
        do {
            try FileManager.default.moveItem(at: location, to: destination)
            lock.withLock {
                if state.task === downloadTask { state.completedURL = destination }
            }
        } catch {
            complete(task: downloadTask, result: .failure(error))
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        _ = session
        let resume = (error as NSError?)?.userInfo[NSURLSessionDownloadTaskResumeData] as? Data
        if let resume { finishCancellation(with: resume) }
        if let error {
            let rejectedRedirect = lock.withLock {
                state.task === task && state.rejectedRedirect
            }
            if rejectedRedirect {
                complete(task: task, result: .failure(ModelDownloadError.unsafeRedirect))
            } else if Self.isCancellation(error) {
                complete(task: task, result: .failure(CancellationError()))
            } else {
                complete(task: task, result: .failure(error))
            }
            return
        }
        let completed = lock.withLock { state.task === task ? state.completedURL : nil }
        guard let completed else {
            complete(task: task, result: .failure(ModelDownloadError.missingTemporaryFile))
            return
        }
        complete(task: task, result: .success(completed))
    }

    private func complete(task: URLSessionTask, result: Result<URL, Error>) {
        let completion = lock.withLock {
            () -> (CheckedContinuation<URL, Error>?, URL?)? in
            guard state.task === task else { return nil }
            let continuation = state.continuation
            let temporary: URL? = if case .failure = result {
                state.completedURL
            } else {
                nil
            }
            state.task = nil
            state.continuation = nil
            state.progress = nil
            state.completedURL = nil
            state.rejectedRedirect = false
            return (continuation, temporary)
        }
        if let temporary = completion?.1 { Self.removeOwnedTemporary(temporary) }
        completion?.0?.resume(with: result)
    }

    private func finishCancellation(with data: Data?) {
        let continuation = lock.withLock { () -> CheckedContinuation<Data?, Never>? in
            let continuation = state.cancelContinuation
            state.cancelContinuation = nil
            return continuation
        }
        continuation?.resume(returning: data)
    }

    static func isAllowed(_ url: URL?) -> Bool {
        guard let url,
              url.scheme?.lowercased() == "https",
              let host = url.host?.lowercased() else { return false }
        return host == "huggingface.co"
            || host.hasSuffix(".huggingface.co")
            || host == "hf.co"
            || host.hasSuffix(".hf.co")
    }

    private static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        return (error as? URLError)?.code == .cancelled
            || (error as NSError).domain == NSURLErrorDomain
                && (error as NSError).code == URLError.cancelled.rawValue
    }

    private static func removeOwnedTemporary(_ url: URL) {
        let standardized = url.standardizedFileURL
        let temporaryRoot = FileManager.default.temporaryDirectory.standardizedFileURL
        guard standardized.deletingLastPathComponent() == temporaryRoot,
              standardized.lastPathComponent.hasPrefix("cloudpoint-model-download-") else {
            return
        }
        var status = stat()
        guard Darwin.lstat(standardized.path, &status) == 0,
              status.st_mode & S_IFMT == S_IFREG else { return }
        try? FileManager.default.removeItem(at: standardized)
    }
}
