#ifndef POINT_VIZ_GLSL
#define POINT_VIZ_GLSL

#include "cap/shader/covq.glsl"

// must match COVQTrainingVisualization::ColorMode
const uint COLOR_MODE_VALUE     = 0;
const uint COLOR_MODE_CELL_HASH = 1;
const uint COLOR_MODE_CODEPOINT = 2;

// must match COVQTrainingVisualization::FlagBits
const uint JITTER_BIT            = 0x1;
const uint SNAP_TO_CODEPOINT_BIT = 0x2;

// must match COVQTrainingVisualization::PushConstants
layout (push_constant) uniform PC {
	// rgba8 dataset values
	Points points;
	// channel output index for each point in the dataset
	Mapping codewords;
	// rgba8 reconstruction vector for each channel output
	Centroids codepoints;
	uint count;
	uint stride;
	uint num_codepoints;
	uint color_mode;
	uint flags;
	float w;
} pc;

uint pcg_hash(uint v) {
	uint state = v * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

#endif // POINT_VIZ_GLSL
