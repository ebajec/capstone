#ifndef COVQ_H
#define COVQ_H

#include <Eigen/Dense>

#include <cstdint>
#include <cfloat>
#include <chrono>
#include <atomic>
#include <mutex>
#include <functional>

#define TIMER_BEGIN(name)\
std::atomic_thread_fence(std::memory_order_acquire);\
std::chrono::time_point name##_timer_start = std::chrono::high_resolution_clock::now();

#define TIMER_END(name)\
std::atomic_thread_fence(std::memory_order_release);\
std::chrono::time_point name##_timer_end = std::chrono::high_resolution_clock::now();\
printf("%s: %4f ms\n", #name, double((name##_timer_end - name##_timer_start).count()) / 1e6); 

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

	std::vector<uint32_t> mapping;

	std::atomic_uint64_t epoch;
	std::mutex mutex;
};

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

// @brief map an input vector to an index to send over the channel
static constexpr uint32_t covq_encode(const COVQCodec *codec, const double *in)
{
	const int partitions = codec->encoder.E.cols();
	const int dim = codec->dim;

	const Eigen::MatrixXd &E = codec->encoder.E;
	const Eigen::VectorXd &v = codec->encoder.v;

	uint32_t argmin = 0;
	double d_min = DBL_MAX;

	for (uint32_t i = 0; i < partitions; ++i) {
		double Ex_i = 0.0;
		for (uint32_t j = 0; j < dim; ++j)
			Ex_i += E(j, i) * in[j];
		double d = v(i) - 2.0 * Ex_i;

		if (d < d_min) {
			argmin = i;
			d_min = d;
		}
	}

	return codec->encoder.index_map[argmin];
}

static constexpr uint32_t covq_index_mapping(const COVQTrainingContext &ctx, uint32_t i)
{
	return ctx.output_index_mapping ? ctx.output_index_mapping(i, ctx.word_length) : i;
}

// @brief map a channel output to a reconstruction vector.
static constexpr void covq_decode(const COVQCodec *codec, uint32_t in, double *out)
{
	const double *src = codec->codebook.col(in).data(); 
	std::copy(src, src + codec->dim, out);
}

extern COVQTrainingContext *covq_init(
	const Eigen::MatrixXd& X,
	const Eigen::MatrixXd &P
);

extern void covq_training_it(
	COVQTrainingContext &ctx,
	const Eigen::MatrixXd &X
);

extern COVQCodec *covq_create_codec(const COVQTrainingContext &ctx);

extern size_t covq_serialized_size(const COVQCodec &codec);
extern int covq_serialize(const COVQCodec &codec, void *buf);
extern int covq_deserialize(COVQCodec *codec, const void *buf);

extern int covq_save(const COVQCodec *codec, const char *path);
extern int covq_load(COVQCodec *codec, const char *path);


#endif // COVQ_H
