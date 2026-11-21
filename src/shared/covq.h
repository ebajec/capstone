#ifndef COVQ_H
#define COVQ_H

#include <Eigen/Dense>

#include <cstdint>
#include <cfloat>
#include <mutex>
#include <functional>

struct COVQEncoder
{
	// For mapping inputs to sent indices, but only for
	// partitions with nonzero probability
	
	Eigen::MatrixXd E; // (dim x M)
	Eigen::VectorXd v; // (M x 1)

	// Maps indices of E and v to an index to send over the channel
	std::vector<uint32_t> index_map; 
};

struct COVQCodec
{
	uint32_t dim;
	uint32_t N; // channel output alphabet size
	uint32_t M; // channel input alphabet size
	uint32_t word_length;

	Eigen::MatrixXd codebook;
	Eigen::MatrixXd centroids;
	Eigen::MatrixXd probs;

	// channel matrix
	Eigen::MatrixXd P;

	COVQEncoder encoder;
};

struct COVQTrainingContext
{
	uint32_t count;
	uint32_t dim;
	uint32_t N;
	uint32_t M;

	uint32_t word_length;

	// map an encoder index to the uncorrupted output.
	// used for a channel with erasures, where the encoder
	// codebook differs in size from the decoder one
	uint32_t (*output_index_mapping)(uint32_t w, int word_length) = nullptr;

	Eigen::MatrixXd P;
	Eigen::MatrixXd Y;
	Eigen::MatrixXd E;
	Eigen::VectorXd v;

	Eigen::MatrixXd probs;
	Eigen::MatrixXd centroids;

	// number of reconstruction vectors with positive probability
	uint32_t num_possible_codepoints;
	double expected_distortion = 0;

	std::vector<uint32_t> mapping;

	std::atomic_uint64_t epoch;

	static COVQTrainingContext *create(
		const Eigen::MatrixXd& X,
		const Eigen::MatrixXd &P
	);

	void train(const Eigen::MatrixXd& X, bool log = true);
	COVQCodec *create_codec();
};

static constexpr uint32_t covq_index_mapping(const COVQTrainingContext &ctx, uint32_t i)
{
	return ctx.output_index_mapping ? ctx.output_index_mapping(i, ctx.word_length) : i;
}

extern size_t covq_serialized_size(const COVQCodec &codec);
extern int covq_serialize(const COVQCodec &codec, void *buf);
extern int covq_deserialize(COVQCodec *codec, const void *buf);

extern int covq_save(const COVQCodec *codec, const char *path);
extern int covq_load(COVQCodec *codec, const char *path);

#endif // COVQ_H
