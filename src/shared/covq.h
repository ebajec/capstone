#ifndef COVQ_H
#define COVQ_H

#include <Eigen/Dense>

#include <cstdint>
#include <chrono>
#include <atomic>
#include <mutex>

#define TIMER_BEGIN(name)\
std::atomic_thread_fence(std::memory_order_acquire);\
std::chrono::time_point name##_timer_start = std::chrono::high_resolution_clock::now();

#define TIMER_END(name)\
std::atomic_thread_fence(std::memory_order_release);\
std::chrono::time_point name##_timer_end = std::chrono::high_resolution_clock::now();\
printf("%s: %4f ms\n", #name, double((name##_timer_end - name##_timer_start).count()) / 1e6); 

struct COVQTrainingDump
{
	std::mutex sync;
	std::atomic_uint32_t generation = 0;
	std::vector<uint32_t> codewords;
	Eigen::MatrixXd codepoints;
};

struct COVQTrainingContext
{
	uint32_t count;
	uint32_t dim;
	uint32_t N;
	uint32_t M;

	uint32_t word_length;

	// map an encoder index to the uncorrupted output.
	// used for training a channel with erasures, where the set of
	// outputs differs from the set of inputs
	uint32_t (*index_mapping)(uint32_t w, int word_length) = nullptr;

	Eigen::MatrixXd P;
	Eigen::MatrixXd Y;
	Eigen::MatrixXd E;
	Eigen::VectorXd v;

	Eigen::MatrixXd S_probs;
	Eigen::MatrixXd S_centers;

	std::vector<uint32_t> mapping;
};


extern COVQTrainingContext *covq_init(
	const Eigen::MatrixXd& X,
	const Eigen::MatrixXd &P
);

extern void covq_training_it(
	COVQTrainingContext &ctx,
	const Eigen::MatrixXd &X,
	COVQTrainingDump *dump = nullptr
);


#endif // COVQ_H
