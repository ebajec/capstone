#version 430 core
#extension GL_GOOGLE_include_directive : require

#include "core/shader/frame.glsl"
#include "cap/shader/covq.glsl"

layout (set = PER_DRAW_SET, binding = 0) uniform sampler2D u_tex;

layout (location = 0) in vec2 in_pos;
layout (location = 1) in vec2 in_uv;
layout (location = 0) out vec4 out_color;

uint pcg_hash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint get_mapping(uint idx)
{
	return pc.mapping.data[pc.image_start + idx];
}

void main()
{
	ivec2 size = pc.size; 
	ivec2 texc = ivec2(in_uv * vec2(size) + vec2(0.5)); 

	uint idx = uint(texc.x + texc.y * pc.size.x);
	uint map = get_mapping(idx);

	uint Y = pc.centroids.data[min(map, 255)];

	vec3 rgb = bool(pc.flags & USE_HASH_COLORS_BIT) ? 
		unpackUnorm4x8(Y).rgb :
		texture(u_tex, in_uv).rgb;
	out_color = vec4(rgb, 1);
}
