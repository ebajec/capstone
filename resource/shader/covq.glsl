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

layout (push_constant) uniform PC {
	Points points;
	Mapping mapping;
	Centroids centroids;
	uint count;
	float w;
	int stride;
	ivec2 size;
} pc;

