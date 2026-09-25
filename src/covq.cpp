#include "covq.h"

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
			X.col(i).swap(X.col(j));
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

// @param X - (dim , count) input dataset
// @param P - (N , N) channel transition probabilities; P(i, j) = p(j|i)
COVQBook *covq_train(Eigen::MatrixXd &X, const Eigen::MatrixXd &P, int its)
{
	uint32_t count = X.cols();
	uint32_t dim = X.rows();
	uint32_t N = P.rows();

	Eigen::MatrixXd Y = split_heuristic(X, N); 

	// initialize codebook

	Eigen::MatrixXd E = Eigen::MatrixXd::Zero(dim, N); 
	Eigen::VectorXd v = Eigen::VectorXd::Zero(N); 

	Eigen::MatrixXd S_wts = Eigen::MatrixXd::Zero(1, N);
	Eigen::MatrixXd S_centers = Eigen::MatrixXd::Zero(dim, N);

	for (int it = 0; it < its; ++it) {
		TIMER_BEGIN(TrainingIteration);

		// compute E and v based on the codepoints from the previous iteration
		
		E.noalias() = Y * P.transpose();
		v.noalias() = P * Y.colwise().squaredNorm().transpose();

		//for (uint32_t i = 0; i < N; ++i) {
		//	for (uint32_t j = 0; j < N; ++j) {
		//		E.col(i) += P(i, j) * Y.col(j);
		//		v(i) += P(i, j) * Y.col(j).squaredNorm();
		//	}
		//}

		// accumulate partition centroids
		S_centers.setZero();
		S_wts.setZero();

		// TODO: derive this from the data per element
		const double p_x = 1.0 / X.cols(); 

		TIMER_BEGIN(Centroids);
		for (uint32_t c = 0; c < count; ++c) {
			uint32_t argmin = 0;
			double min_d = std::numeric_limits<double>::max();

			for (uint32_t i = 0; i < N; ++i) {
				double d = v(i) - 2.0 * (X.col(c).dot(E.col(i)));

				if (d < min_d) {
					argmin = i;
					min_d = d;
				}
			}

			S_wts(0, argmin) += p_x;
			S_centers.col(argmin) += p_x * X.col(c);
		}
		TIMER_END(Centroids);

		Y.setZero();
		Y.noalias() = S_centers * P;

		for (uint32_t j = 0; j < N; ++j) {
			double den = (S_wts * P.col(j))(0,0);

			//for (uint32_t i = 0; i < N; ++i) {
			//	Y.col(j).noalias() += P(i, j) * S_centers.col(i);
			//	//den += P(i, j) * S_wts(0, i);
			//}

			Y.col(j) /= den;
		}

		TIMER_END(TrainingIteration);
	}

	COVQBook * book = new COVQBook{
		.dim = dim,
		.N = N,
		.Y = Y,
		.E = Y * P.transpose(),
		.v = P * Y.colwise().squaredNorm().transpose(),
	};

	return book;
}

