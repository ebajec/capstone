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
	std::vector<uint32_t> mapping;
	Eigen::MatrixXd codepoints;
};

struct COVQTrainingContext
{
	uint32_t count;
	uint32_t dim;
	uint32_t N;

	Eigen::MatrixXd P;
	Eigen::MatrixXd Y;
	Eigen::MatrixXd E;
	Eigen::VectorXd v;

	Eigen::MatrixXd S_wts;
	Eigen::MatrixXd S_centers;

	// one for each thread
	Eigen::VectorXd d;

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
