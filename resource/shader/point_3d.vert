#version 430 core
#extension GL_GOOGLE_include_directive : require

#include "core/shader/core.glsl"
#include "cap/shader/point_viz.glsl"

layout (location = 0) flat out vec3 out_color;

vec3 get_value(uint idx)
{
	return unpackUnorm4x8(pc.points.data[idx]).rgb;
}

// falls back to `value` if there is no codepoint for this codeword yet
vec3 get_codepoint(uint codeword, vec3 value)
{
	if (codeword >= pc.num_codepoints)
		return value;
	return unpackUnorm4x8(pc.codepoints.data[codeword]).rgb;
}

void main()
{
	const vec2 corners[4] = {
		vec2(-1,-1),
		vec2(1,-1),
		vec2(-1,1),
		vec2(1,1),
	};

	uint idx = min(uint(gl_InstanceIndex) * max(pc.stride, 1u), pc.count - 1);

	vec3 value = get_value(idx);
	uint codeword = pc.codewords.data[idx];

	vec3 codepoint = get_codepoint(codeword, value);

	vec3 color;
	switch (pc.color_mode) {
	case COLOR_MODE_CELL_HASH:
		color = unpackUnorm4x8(pcg_hash(codeword)).rgb;
		break;
	case COLOR_MODE_CODEPOINT:
		color = codepoint;
		break;
	case COLOR_MODE_VALUE:
	default:
		color = value;
		break;
	}

	vec3 pos = bool(pc.flags & SNAP_TO_CODEPOINT_BIT) ? codepoint : value;

	if (bool(pc.flags & JITTER_BIT))
		pos += (vec3(1.f) - 2.f * unpackUnorm4x8(pcg_hash(idx)).rgb) / 255.f;

	float aspect = u_pass.p[0][0]/u_pass.p[1][1];
	vec2 scale = vec2(pc.w*aspect, pc.w);

	vec4 cs_pos = u_pass.pv * vec4(pos, 1);
	cs_pos.xy += scale * corners[gl_VertexIndex & 0x3];

	out_color = color;

	gl_Position = cs_pos;
}
