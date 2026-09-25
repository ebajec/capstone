#ifndef COVQ_H
#define COVQ_H

#include <Eigen/Eigen>

#include <cstdint>
#include <chrono>
#include <atomic>

#define TIMER_BEGIN(name)\
std::atomic_thread_fence(std::memory_order_acquire);\
std::chrono::time_point name##_timer_start = std::chrono::high_resolution_clock::now();

#define TIMER_END(name)\
std::atomic_thread_fence(std::memory_order_release);\
std::chrono::time_point name##_timer_end = std::chrono::high_resolution_clock::now();\
printf("%s: %4f ms\n", #name, double((name##_timer_end - name##_timer_start).count()) / 1e6); 

struct COVQBook
{
	uint32_t dim; // dimension of data
	uint32_t N; // codebook size

	Eigen::MatrixXd Y; // the codebook; y_j for j = 0...N-1

 	// For distorion computation.
	// E[i] = p(0|i) * y_0 + ... + p(N-1|i) * y_{N-1}   
	Eigen::MatrixXd E;

 	// For distorion computation.
	// v[i] = p(0|i) * |y_0|^2 + ... + p(N-1|i) * |y_{N-1}|^2   
	Eigen::VectorXd v;
};

extern COVQBook *covq_train(
	Eigen::MatrixXd &X,
	const Eigen::MatrixXd &P,
	int its
);


#endif // COVQ_H
