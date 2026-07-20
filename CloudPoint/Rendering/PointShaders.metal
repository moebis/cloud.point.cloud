#include <metal_stdlib>
using namespace metal;

struct PackedPoint {
    packed_float3 position;
    uchar4 rgba;
    half confidence;
    ushort flags;
    uint sourceFrame;
};

static_assert(sizeof(PackedPoint) == 24, "CPC1 point records must remain packed at 24 bytes");

struct PointCloudUniforms {
    float4x4 viewProjection;
    float pointSize;
    float confidenceThreshold;
    float2 padding;
};

static_assert(sizeof(PointCloudUniforms) == 80, "Swift and Metal uniforms must remain 80 bytes");

struct PointRasterizerData {
    float4 position [[position]];
    float pointSize [[point_size]];
    float4 color;
    float confidence;
};

vertex PointRasterizerData pointCloudVertex(
    device const PackedPoint *points [[buffer(0)]],
    constant PointCloudUniforms &uniforms [[buffer(1)]],
    uint vertexID [[vertex_id]]
) {
    PackedPoint point = points[vertexID];
    PointRasterizerData output;
    output.position = uniforms.viewProjection * float4(float3(point.position), 1.0);
    output.pointSize = uniforms.pointSize;
    output.color = float4(point.rgba) / 255.0;
    output.confidence = float(point.confidence);
    return output;
}

fragment float4 pointCloudFragment(
    PointRasterizerData input [[stage_in]],
    float2 pointCoordinate [[point_coord]],
    constant PointCloudUniforms &uniforms [[buffer(1)]]
) {
    if (input.confidence < uniforms.confidenceThreshold) {
        discard_fragment();
    }
    float2 centered = pointCoordinate * 2.0 - 1.0;
    if (dot(centered, centered) > 1.0) {
        discard_fragment();
    }
    return input.color;
}
