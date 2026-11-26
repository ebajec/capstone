#ifndef POINT_VIZ_GLSL
#define POINT_VIZ_GLSL

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require

layout (buffer_reference, std430, buffer_reference_align = 16) readonly buffer U32Buffer
{
	uint data[];
};

// must match COVQTrainingVisualization::ColorMode
const uint COLOR_MODE_VALUE     = 0;
const uint COLOR_MODE_CELL_HASH = 1;
const uint COLOR_MODE_CODEPOINT = 2;

// must match COVQTrainingVisualization::FlagBits
const uint JITTER_BIT            = 0x1;
const uint SNAP_TO_CODEPOINT_BIT = 0x2;

// must match COVQTrainingVisualization::PushConstants
layout (push_constant) uniform PC {
	// unorm16 dataset values
	U32Buffer points;
	// channel output index for each point in the dataset
	U32Buffer codewords;
	// unorm16 reconstruction vector for each channel output
	U32Buffer codepoints;
	uint count;
	uint stride;
	uint num_codepoints;
	uint color_mode;
	uint flags;
	// number of unorm16 components per point / codepoint
	uint dim;
	float w;

	// maps data into R^3.  To support dim > 3, would want to move to a separate
	// buffer b/c of push constant size limits
	mat3 basis;
} pc;

uint pcg_hash(uint v) {
	uint state = v * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

#endif // POINT_VIZ_GLSL
