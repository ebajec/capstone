#version 430 core
#extension GL_GOOGLE_include_directive : require

#include "core/shader/frame.glsl"
#include "cap/shader/covq.glsl"

layout (location = 0) flat out uint out_idx;
layout (location = 1) flat out vec3 out_orig;

// useful for debugging
vec3 get_pos_grid(uint idx)
{
	int y = int(idx) / pc.size.x;
	int x = int(idx) - y * pc.size.x;

	vec2 g = vec2(x, y)/ 1000;
	return vec3(g, 0); 
}

vec3 get_value(uint idx)
{
	return unpackUnorm4x8(pc.points.data[idx]).rgb;
}

void main()
{
	const vec2 corners[4] = {
		vec2(-1,-1),
		vec2(1,-1),
		vec2(-1,1),
		vec2(1,1),
	};

	uint idx = gl_InstanceIndex;

	if (pc.stride > 0) {
		idx = min(idx * uint(pc.stride), pc.count - 1);
	}

	vec3 value = get_value(idx);

	vec3 pos = value;
	uint mapping = pc.mapping.data[idx];

	float aspect = u_view.p[0][0]/u_view.p[1][1];
	vec2 scale = vec2(pc.w*aspect, pc.w);

	vec4 cs_pos = u_view.pv * vec4(pos, 1);
	cs_pos.xy += scale * corners[gl_VertexIndex & 0x3];

	out_idx = mapping;
	out_orig = value;

	gl_Position = cs_pos;
}
