#version 430 core
#extension GL_GOOGLE_include_directive : require

#include "cap/shader/covq.glsl"

layout (location = 0) flat in uint in_idx;
layout (location = 1) flat in vec3 in_orig;
layout (location = 0) out vec4 out_color;

uint pcg_hash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

void main()
{
	//vec3 rgb = in_orig;
	vec3 rgb = unpackUnorm4x8(pcg_hash(in_idx)).rgb;
	//vec3 rgb = unpackUnorm4x8(pc.centroids.data[in_idx]).rgb;

	out_color = vec4(rgb,1);
}
