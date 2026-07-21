import Foundation
import XCTest
@testable import CloudPoint

final class SharpWorkerProtocolTests: XCTestCase {
    func testEveryEventDecodesFromCanonicalStrictJSONLine() throws {
        let lines: [(String, SharpWorkerEvent)] = [
            (
                #"{"fraction":0.25,"protocolVersion":1,"stage":"inference","type":"progress"}"#,
                .progress(stage: .inference, fraction: 0.25)
            ),
            (
                #"{"monotonicSeconds":12.5,"protocolVersion":1,"type":"heartbeat"}"#,
                .heartbeat(monotonicSeconds: 12.5)
            ),
            (
                #"{"code":"MPS_FALLBACK","message":"Retrying on CPU","protocolVersion":1,"recoverable":true,"type":"warning"}"#,
                .warning(code: "MPS_FALLBACK", message: "Retrying on CPU", recoverable: true)
            ),
            (
                #"{"device":"mps","durationSeconds":3.5,"gaussianCount":1179648,"plyRelativePath":"Outputs/Gaussians/00000000.ply","protocolVersion":1,"provenanceRelativePath":"Outputs/Gaussians/00000000.json","sourceFrameIndex":0,"type":"completed","usedCPUFallback":false}"#,
                .completed(SharpWorkerCompletion(
                    sourceFrameIndex: 0,
                    plyRelativePath: "Outputs/Gaussians/00000000.ply",
                    provenanceRelativePath: "Outputs/Gaussians/00000000.json",
                    gaussianCount: 1_179_648,
                    durationSeconds: 3.5,
                    device: "mps",
                    usedCPUFallback: false
                ))
            ),
            (
                #"{"code":"SHARP_INFERENCE_FAILED","message":"failed","protocolVersion":1,"recoverable":true,"type":"failed"}"#,
                .failed(code: "SHARP_INFERENCE_FAILED", message: "failed", recoverable: true)
            ),
            (
                #"{"protocolVersion":1,"type":"cancelled"}"#,
                .cancelled
            ),
        ]

        for (line, expected) in lines {
            XCTAssertEqual(try SharpWorkerLineCodec.decode(Data(line.utf8)), expected)
        }
    }

    func testUnknownDuplicateMalformedAndNonFiniteFieldsFailClosed() {
        let invalid = [
            #"{"fraction":0.2,"protocolVersion":1,"stage":"inference","type":"progress","unknown":1}"#,
            #"{"protocolVersion":2,"type":"cancelled"}"#,
            #"{"fraction":1e999,"protocolVersion":1,"stage":"inference","type":"progress"}"#,
            #"{"protocolVersion":1,"type":"cancelled","type":"cancelled"}"#,
            #"{"device":"mps","durationSeconds":1,"gaussianCount":0,"plyRelativePath":"../bad.ply","protocolVersion":1,"provenanceRelativePath":"bad.json","sourceFrameIndex":0,"type":"completed","usedCPUFallback":false}"#,
        ]

        for line in invalid {
            XCTAssertThrowsError(try SharpWorkerLineCodec.decode(Data(line.utf8)), line)
        }
    }

    func testLineSizeIsBounded() {
        XCTAssertThrowsError(
            try SharpWorkerLineCodec.decode(Data(repeating: 0x20, count: 1_048_577))
        )
    }
}
