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

	unsigned dst = (*idx)++;
	if (dst >= K) {
		return;
	}

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

	out.col(dst) = mu;

	if (i == 0 || i == n)
		return;

	split_heuristic_rec(X.leftCols(i), out, its, K, idx, depth - 1);
	split_heuristic_rec(X.rightCols(n - i), out, its, K, idx, depth - 1);
}

static Eigen::MatrixXd split_heuristic(Eigen::MatrixXd &X, unsigned K)
{
	int dims = X.rows();
	Eigen::MatrixXd out = Eigen::MatrixXd(dims, K); 

	int depth = std::bit_width(K);

	unsigned idx = 0;
	split_heuristic_rec(X, out, 20, K, &idx, depth);

	return out;
}

Eigen::MatrixXd covq_init_centers(
	const Eigen::MatrixXd& X,
	const Eigen::MatrixXd &P
)
{
	assert(P.rows() == P.cols());
	int N = P.rows();

	// gets rearranged in place
	Eigen::MatrixXd X2 = X;
	return split_heuristic(X2, N);
}

COVQTrainingContext *covq_init(
	const Eigen::MatrixXd& X,
	const Eigen::MatrixXd &P
)
{
	COVQTrainingContext *ctx = new COVQTrainingContext{};

	uint32_t count = X.cols();
	uint32_t dim = X.rows();
	uint32_t N = P.rows();

	*ctx = COVQTrainingContext{
		.count = count,
		.dim = dim,
		.N = N,

		.E = Eigen::MatrixXd::Zero(dim, N), 
		.v = Eigen::VectorXd::Zero(N), 

		.S_wts = Eigen::MatrixXd::Zero(1, N),
		.S_centers = Eigen::MatrixXd::Zero(dim, N),

		.d = Eigen::VectorXd::Zero(N),

		.mapping = std::vector<uint32_t>(count),

	};

	ctx->Y = covq_init_centers(X, P);

	return ctx;
}

// @param X - (dim , count) input dataset
// @param P - (N , N) channel transition probabilities; P(i, j) = p(j|i)
void covq_training_it(
	COVQTrainingContext &ctx,
	const Eigen::MatrixXd &X,
	const Eigen::MatrixXd &P, 
	int its, 
	COVQTrainingDump *dump
)
{
	TIMER_BEGIN(TrainingIteration);

	// compute E and v based on the codepoints from the previous iteration
	
	ctx.E.noalias() = ctx.Y * P.transpose();
	ctx.v.noalias() = P * ctx.Y.colwise().squaredNorm().transpose();

	// TODO: derive this from the data per element
	const double p_x = 1.0 / X.cols(); 

	uint32_t largest = 0;

	TIMER_BEGIN(Centroids);
    //#pragma omp parallel for
	for (uint32_t c = 0; c < ctx.count; ++c) {
		uint32_t argmin = 0;

		ctx.d.noalias() = ctx.v - 2.0 * (ctx.E.transpose() * X.col(c));
		ctx.d.minCoeff(&argmin);

		uint32_t idx = argmin;
		ctx.mapping[c] = argmin;

		largest = std::max(largest, argmin);
	}
	TIMER_END(Centroids);

	printf("Largest: %d\n", largest);

	// accumulate partition centroids
	ctx.S_centers.setZero();
	ctx.S_wts.setZero();

	for (uint32_t c = 0; c < ctx.count; ++c) {
		uint32_t idx = ctx.mapping[c];
		ctx.S_wts(0, idx) += p_x;
		ctx.S_centers.col(idx).noalias() += p_x * X.col(c);
	}

	ctx.Y.setZero();
	ctx.Y.noalias() = ctx.S_centers * P;

	double smallest = 0;

	for (uint32_t j = 0; j < ctx.N; ++j) {
		double den = (ctx.S_wts * P.col(j))(0,0);

		if (fabs(den) < 1e-6) {
			ctx.Y.col(j).setZero();
		} else {
			ctx.Y.col(j) /= den;
		}

		smallest = std::min(den, smallest);
	}
	printf("smallest: %f\n", smallest);

	TIMER_END(TrainingIteration);

	if (dump) {
		std::unique_lock<std::mutex> lock(dump->sync);
		dump->mapping = ctx.mapping;
		dump->codepoints = ctx.Y;
		++dump->generation;
	}
	//}
}

