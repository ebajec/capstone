#include "covq.h"

#include <cassert>

static void split_heuristic_rec(
	Eigen::Ref<Eigen::MatrixXd> X,
	Eigen::Ref<Eigen::MatrixXd> out,
	unsigned its, unsigned K, unsigned *idx, int depth)
{
	Eigen::Index n = X.cols();

	if (n < 2 || depth == 0)
		return;

    Eigen::VectorXd mu = X.rowwise().mean();

    Eigen::MatrixXd Xc = X.colwise() - mu;
    Eigen::VectorXd mu_test = Xc.rowwise().minCoeff();
    Eigen::VectorXd u = Eigen::VectorXd::Random(X.rows()).normalized();

	Eigen::MatrixXd A = Xc * Xc.transpose();

    for (int t = 0; t < its; ++t) {
        Eigen::VectorXd w = A * u;
        u = w.normalized();
    }

	if (!n) {
		fprintf(stderr, "X is empty!\n");
		return;
	}

	Eigen::Index i = 0;
	Eigen::Index j = n - 1;

	while (i <= j) {
		double prod = Xc.col(i).dot(u);

		if (prod < 0) {
			++i;
		} else {
			Xc.col(i).swap(Xc.col(j));
			--j;
		}
	}

	// don't need this anymore
	Xc.resize(0,0);

	if (depth == 1) {
		unsigned dst = (*idx)++;
		if (dst >= K) {
			return;
		}
		out.col(dst) = mu;
	}

	if (i == 0 || i == n)
		return;

	split_heuristic_rec(X.leftCols(i), out, its, K, idx, depth - 1);
	split_heuristic_rec(X.rightCols(n - i), out, its, K, idx, depth - 1);
}

static Eigen::MatrixXd split_heuristic_internal(Eigen::MatrixXd &X, unsigned K)
{
	int dims = X.rows();
	Eigen::MatrixXd out = Eigen::MatrixXd(dims, K); 

	int depth = std::bit_width(K);

	unsigned idx = 0;
	split_heuristic_rec(X, out, 20, K, &idx, depth);

	return out;
}

static Eigen::MatrixXd split_heuristic(
	const Eigen::MatrixXd& X,
	int N
)
{
	// gets rearranged in place
	Eigen::MatrixXd X2 = X;
	return split_heuristic_internal(X2, N);
}

constexpr uint32_t bits_to_base_3(uint32_t w, int k)
{
	uint64_t v = 0;
	for (int i = k - 1; i >= 0;  --i) {
		v = 3 * v + ((w >> i) & 1);
	}
	return v;
}

COVQTrainingContext *covq_init(
	const Eigen::MatrixXd& X,
	const Eigen::MatrixXd &P
)
{
	COVQTrainingContext *ctx = new COVQTrainingContext{};

	uint32_t count = X.cols();
	uint32_t dim = X.rows();
	uint32_t M = P.rows();
	uint32_t N = P.cols();

	uint32_t numthread = omp_get_num_threads();

	*ctx = COVQTrainingContext{
		.count = count,
		.dim = dim,
		.N = N,
		.M = M,
		.word_length = (uint32_t)(std::bit_width(M) - 1),
		.index_mapping = N == M ? nullptr : &bits_to_base_3,

		.P = P,
		.E = Eigen::MatrixXd::Zero(dim, M), 
		.v = Eigen::VectorXd::Zero(M), 

		.S_probs = Eigen::MatrixXd::Zero(1, M),

		// dims * M
		.S_centers = split_heuristic(X, M),

		.mapping = std::vector<uint32_t>(count, 0),
	};

	Eigen::VectorXd d = Eigen::VectorXd::Zero(M);
	for (Eigen::Index idx = 0; idx < X.cols(); ++idx) {
		d.noalias() = (ctx->S_centers.colwise() - X.col(idx)).colwise().squaredNorm(); 
		uint32_t argmin = 0;
		d.minCoeff(&argmin);
		ctx->S_probs(0, argmin) += 1;
	}

	const double p_x = 1.0/(double)X.cols();
	
	ctx->S_probs *= p_x;
	ctx->S_centers *= p_x;

	return ctx;
}

static uint32_t covq_index_mapping(const COVQTrainingContext &ctx, uint32_t i)
{
	return ctx.index_mapping ? ctx.index_mapping(i, ctx.word_length) : i;
}

// @param X - (dim , count) input dataset
// @param P - (N , N) channel transition probabilities; P(i, j) = p(j|i)
void covq_training_it(
	COVQTrainingContext &ctx,
	const Eigen::MatrixXd &X,
	COVQTrainingDump *dump
)
{
#undef EIGEN_USE_AVX512_GEMM_KERNELS
#define EIGEN_USE_AVX512_GEMM_KERNELS 1

	TIMER_BEGIN(TrainingIteration);

	ctx.Y.setZero();
	ctx.Y.noalias() = ctx.S_centers * ctx.P;

	for (uint32_t j = 0; j < ctx.N; ++j) {
		double prob_j = (ctx.S_probs * ctx.P.col(j))(0,0);

		// TODO: Ignore when probability is near zero
		if (fabs(prob_j) < 1e-8) {
			ctx.Y.col(j).setZero();
		} else {
			ctx.Y.col(j) /= prob_j;
		}
	}

	ctx.E.noalias() = ctx.Y * ctx.P.transpose();
	ctx.v.noalias() = ctx.P * ctx.Y.colwise().squaredNorm().transpose();

	TIMER_BEGIN(Centroids);
	#pragma omp parallel
	{
		Eigen::VectorXd d_tls = Eigen::VectorXd::Zero(ctx.M);

		#pragma omp for schedule(dynamic)
		for (uint32_t c = 0; c < ctx.count; ++c) {
			uint32_t argmin = 0;

			d_tls.noalias() = ctx.v - 2.0 * (ctx.E.transpose() * X.col(c));
			d_tls.minCoeff(&argmin);

			uint32_t idx = argmin;
			ctx.mapping[c] = idx;
		}
	}
	TIMER_END(Centroids);

	// accumulate partition centroids
	ctx.S_centers.setZero();
	ctx.S_probs.setZero();

	for (uint32_t c = 0; c < ctx.count; ++c) {
		uint32_t idx = ctx.mapping[c];
		ctx.S_probs(0, idx) += 1.0;
		ctx.S_centers.col(idx).noalias() += X.col(c);
	}

	const double p_x = 1.0 / X.cols(); 

	ctx.S_probs *= p_x;
	ctx.S_centers *= p_x;

	TIMER_END(TrainingIteration);

	if (dump) {
		std::unique_lock<std::mutex> lock(dump->sync);
		TIMER_BEGIN(TrainingDump);

		dump->codewords.clear();
		dump->codewords.reserve(ctx.mapping.size());

		uint32_t k = std::bit_width(ctx.M) - 1;

		for (uint32_t idx : ctx.mapping) {
			uint32_t codeword = covq_index_mapping(ctx, idx);
			dump->codewords.push_back(codeword);
		}
		dump->codepoints = ctx.Y;
		++dump->generation;
		TIMER_END(TrainingDump);
	}
#undef EIGEN_USE_AVX512_GEMM_KERNELS
#define EIGEN_USE_AVX512_GEMM_KERNELS 0
}

