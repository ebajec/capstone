#version 430 core
#extension GL_GOOGLE_include_directive : require

#include "core/shader/core.glsl"
#include "cap/shader/point_viz.glsl"

layout (location = 0) flat out vec3 out_color;

// i-th unorm16 component of a tightly packed buffer
float load_unorm16(U32Buffer buf, uint i)
{
	return unpackUnorm2x16(buf.data[i >> 1])[i & 0x1u];
}

// idx-th vector of `buf` mapped to display space
vec3 get_vector(U32Buffer buf, uint idx)
{
	uint base = idx * pc.dim;

	vec3 v = vec3(0);
	for (uint i = 0; i < min(pc.dim, 3u); ++i)
		v += pc.basis[i] * load_unorm16(buf, base + i);

	return v;
}

vec3 get_codepoint(uint codeword, vec3 value)
{
	if (codeword >= pc.num_codepoints)
		return value;
	return get_vector(pc.codepoints, codeword);
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

	vec3 value = get_vector(pc.points, idx);
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
