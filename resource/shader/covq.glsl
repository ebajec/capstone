#ifndef COVQ_GLSL
#define COVQ_GLSL

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require

layout (buffer_reference, std430, buffer_reference_align = 16) readonly buffer Points
{
	uint data[];
};

layout (buffer_reference, std430, buffer_reference_align = 16) readonly buffer Mapping
{
	uint data[];
};

layout (buffer_reference, std430, buffer_reference_align = 16) readonly buffer Centroids
{
	uint data[];
};

#endif // COVQ_GLSL
