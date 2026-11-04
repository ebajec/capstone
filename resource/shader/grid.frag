
#version 430 core
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "core/shader/frame.glsl"

layout (location = 0) in vec2 in_pos;
layout (location = 1) in vec2 in_uv;
layout (location = 0) out vec4 out_color;

layout (push_constant) uniform PC {
	uint tex;
} pc;

vec3 palette8(int i)
{
        const vec3 colors[9] = vec3[9](
                vec3(0, 0, 0),  // #440154
                vec3(0.267, 0.004, 0.329),  // #440154
                vec3(0.275, 0.196, 0.494),  // #46327e
                vec3(0.212, 0.361, 0.553),  // #365c8d
                vec3(0.153, 0.498, 0.557),  // #277f8e
                vec3(0.122, 0.631, 0.529),  // #1fa187
                vec3(0.290, 0.757, 0.427),  // #4ac16d
                vec3(0.627, 0.855, 0.224),  // #a0da39
                vec3(0.992, 0.906, 0.145)   // #fde725
        );
        return colors[clamp(i, 0, 8)];
}

void main()
{
	vec2 res = u_pass.resolution;
	vec2 px = vec2(gl_FragCoord.x, res.y - gl_FragCoord.y);
	vec2 uv = px / min(res.x, res.y);

	if (any(greaterThanEqual(uv, vec2(1)))){
		out_color = vec4(0);
		return;
	}

	int b = int(255 * texture(u_textures[pc.tex], uv).r);
	out_color = vec4(palette8(b), 1);
}
