import Darwin
import Foundation

guard CommandLine.arguments.count >= 2 else {
    FileHandle.standardError.write(Data("launcher requires an executable\n".utf8))
    exit(64)
}

if setpgid(0, 0) != 0, getpgrp() != getpid() {
    FileHandle.standardError.write(Data("launcher setpgid failed: \(errno)\n".utf8))
    exit(71)
}
guard getpgrp() == getpid() else {
    FileHandle.standardError.write(Data("launcher process group mismatch\n".utf8))
    exit(71)
}

FileHandle.standardError.write(Data("CLOUDPOINT_LAUNCHER_READY:\(getpid())\n".utf8))

let targetArguments = Array(CommandLine.arguments.dropFirst())
let executable = strdup(targetArguments[0])!
var arguments: [UnsafeMutablePointer<CChar>?] = targetArguments.map { strdup($0) }
arguments.append(nil)
var environment: [UnsafeMutablePointer<CChar>?] = ProcessInfo.processInfo.environment.map {
    strdup("\($0.key)=\($0.value)")
}
environment.append(nil)

arguments.withUnsafeMutableBufferPointer { argumentsBuffer in
    environment.withUnsafeMutableBufferPointer { environmentBuffer in
        execve(executable, argumentsBuffer.baseAddress!, environmentBuffer.baseAddress!)
    }
}

let failure = errno
free(executable)
arguments.dropLast().forEach { free($0) }
environment.dropLast().forEach { free($0) }
FileHandle.standardError.write(Data("launcher exec failed: \(failure)\n".utf8))
exit(126)
