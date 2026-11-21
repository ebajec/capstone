#include "covq.h"

#include <cassert>
#include "timers.h"

#define ZERO_PROB_THRESHOLD 1e-14

static void split_heuristic_rec(
	Eigen::Ref<Eigen::MatrixXd> X,
	Eigen::Ref<Eigen::MatrixXd> out,
	unsigned K, unsigned *idx, int depth)
{
	Eigen::Index n = X.cols();

	if (!n)
		return;

    Eigen::VectorXd mu = X.rowwise().mean();

	const uint32_t leaf_depth = std::bit_width(K) - 1;

	if (depth == leaf_depth) {
		unsigned dst = (*idx)++;
		if (dst >= K) {
			printf("Done\n");
			return;
		}
		out.col(dst) = mu;
		return;
	}

    Eigen::MatrixXd Xc = X.colwise() - mu;

	Eigen::MatrixXd A = Xc * Xc.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
    Eigen::VectorXd u = es.eigenvectors().col(X.rows() - 1);   // largest eigenvalue
	
	// canonicalize u so that largest compoenent is always positive
	Eigen::Index k;
    u.cwiseAbs().maxCoeff(&k);
    if (u(k) < 0)
		u = -u;

	if (!n) {
		fprintf(stderr, "X is empty!\n");
		return;
	}

	Eigen::VectorXd proj = Xc.transpose() * u;
	Xc.resize(0,0);

	Eigen::Index i = 0;
	Eigen::Index j = n - 1;

	while (i <= j) {
		if (proj(i) < 0) {
			++i;
		} else {
			X.col(i).swap(X.col(j));
			std::swap(proj(i), proj(j));
			--j;
		}
	}

	if (i == 0 || i == n) {
        const unsigned leaves = 1u << (leaf_depth - depth);
        for (unsigned l = 0; l < leaves && *idx < K; ++l)
                out.col((*idx)++) = mu;
        return;
	}

	Eigen::Ref<Eigen::MatrixXd> left = X.leftCols(i);
	Eigen::Ref<Eigen::MatrixXd> right = X.rightCols(n - i);

	split_heuristic_rec(left, out, K, idx, depth + 1);
	split_heuristic_rec(right, out, K, idx, depth + 1);
}

static Eigen::MatrixXd split_heuristic_internal(Eigen::MatrixXd &X, unsigned K)
{
	int dims = X.rows();
	Eigen::MatrixXd out = Eigen::MatrixXd(dims, K); 

	int depth = 0;

	unsigned idx = 0;
	split_heuristic_rec(X, out, K, &idx, depth);

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

COVQTrainingContext *COVQTrainingContext::create(
	const Eigen::MatrixXd& X,
	const Eigen::MatrixXd &P
)
{
	uint32_t count = X.cols();
	uint32_t dim = X.rows();
	uint32_t M = P.rows();
	uint32_t N = P.cols();

	uint32_t numthread = omp_get_num_threads();

	// constructed in place, since the mutex/atomic members make it non-copyable
	COVQTrainingContext *ctx = new COVQTrainingContext{
		.count = count,
		.dim = dim,
		.N = N,
		.M = M,
		.word_length = (uint32_t)(std::bit_width(M) - 1),
		.output_index_mapping = N == M ? nullptr : &bits_to_base_3,

		.P = P,
		.E = Eigen::MatrixXd::Zero(dim, M), 
		.v = Eigen::VectorXd::Zero(M), 

		.probs = Eigen::MatrixXd::Zero(1, M),

		// dims * M
		.centroids = Eigen::MatrixXd::Zero(dim, M),

		.mapping = std::vector<uint32_t>(count, 0),
	};

	Eigen::MatrixXd init = split_heuristic(X, M); 

	Eigen::VectorXd d = Eigen::VectorXd::Zero(M);
	for (Eigen::Index idx = 0; idx < X.cols(); ++idx) {
		d.noalias() = (init.colwise() - X.col(idx)).colwise().squaredNorm(); 
		uint32_t argmin = 0;
		d.minCoeff(&argmin);
		ctx->probs(0, argmin) += 1;
		ctx->centroids.col(argmin) += X.col(idx);
	}

	const double p_x = 1.0/(double)X.cols();
	
	ctx->probs *= p_x;
	ctx->centroids *= p_x;

	return ctx;
}

//#define LOG_COVQ_TRAINING

// @param X - (dim , count) input dataset
// @param P - (N , N) channel transition probabilities; P(i, j) = p(j|i)
void COVQTrainingContext::train(const Eigen::MatrixXd& X, bool log)
{
#undef EIGEN_USE_AVX512_GEMM_KERNELS
#define EIGEN_USE_AVX512_GEMM_KERNELS 1

#ifdef LOG_COVQ_TRAINING
	printf("------------------------------------------------\n");
	TIMER_BEGIN(TrainingIteration);
#endif

	Y.setZero();
	Y.noalias() = centroids * P;

	uint32_t num_nonzero_prob = 0;

	for (uint32_t j = 0; j < N; ++j) {
		double prob_j = (probs * P.col(j))(0,0);

		// TODO: Ignore when probability is near zero
		if (fabs(prob_j) < ZERO_PROB_THRESHOLD) {
			Y.col(j).setZero();
		} else {
			Y.col(j) /= prob_j;
			++num_nonzero_prob;
		}
	}

	num_possible_codepoints = num_nonzero_prob;

	E.noalias() = Y * P.transpose();
	v.noalias() = P * Y.colwise().squaredNorm().transpose();


#ifdef LOG_COVQ_TRAINING
	TIMER_BEGIN(Centroids);
#endif

	double d_sum = 0;
	#pragma omp parallel reduction(+:d_sum)
	{
		Eigen::VectorXd d_tls = Eigen::VectorXd::Zero(M);

		double d_expected_tls = 0;

		#pragma omp for schedule(dynamic)
		for (uint32_t c = 0; c < count; ++c) {
			uint32_t argmin = 0;

			d_tls.noalias() = v - 2.0 * (E.transpose() * X.col(c));
			double d = d_tls.minCoeff(&argmin);
			uint32_t idx = argmin;

			d_sum += d + X.col(c).squaredNorm();
			mapping[c] = idx;
		}
	}

#ifdef LOG_COVQ_TRAINING
		TIMER_END(Centroids);
#endif

	double d_expected = d_sum / (double)count;

#ifdef LOG_COVQ_TRAINING
	printf("Expected distortion: %.6f\n", d_expected);
#endif

	expected_distortion = d_expected;

	// accumulate partition centroids
	centroids.setZero();
	probs.setZero();

	for (uint32_t c = 0; c < count; ++c) {
		uint32_t idx = mapping[c];
		probs(0, idx) += 1.0;
		centroids.col(idx).noalias() += X.col(c);
	}

	const double p_x = 1.0 / X.cols(); 

	probs *= p_x;
	centroids *= p_x;

	++epoch;

#ifdef LOG_COVQ_TRAINING
		TIMER_END(TrainingIteration);
#endif

#undef EIGEN_USE_AVX512_GEMM_KERNELS
#define EIGEN_USE_AVX512_GEMM_KERNELS 0
}

static void covq_init_encoder(COVQCodec *codec)
{
	uint32_t nb_nonzero = 0;
	for (uint32_t i = 0; i < codec->M; ++i) {
		if (codec->probs(0, i) > 0.0)
			++nb_nonzero;
	}

	Eigen::MatrixXd E = codec->codebook * codec->P.transpose();
	Eigen::VectorXd v = codec->P * codec->codebook.colwise().squaredNorm().transpose();

	std::vector<uint32_t> index_map (nb_nonzero);
	Eigen::MatrixXd E_out(codec->dim, nb_nonzero);
	Eigen::VectorXd v_out(nb_nonzero);

	for (uint32_t i = 0, idx = 0; i < codec->M; ++i) {
		if (codec->probs(0, i) > 0.0) {
			index_map[idx] = i;
			E_out.col(idx) = E.col(i);
			v_out(idx) = v(i);
			++idx;
		}
	}

	codec->encoder.E = std::move(E_out);
	codec->encoder.v = std::move(v_out);
	codec->encoder.index_map = std::move(index_map);
}

COVQCodec *COVQTrainingContext::create_codec()
{
	COVQCodec *codec = new COVQCodec{COVQCodec{
		.dim = dim,
		.N = N,
		.M = M,
		.word_length = word_length,

		.codebook = Y,
		.centroids = centroids,
		.probs = probs,

		.P = P
	}};

	covq_init_encoder(codec);
	return codec;
}

size_t covq_serialized_size(const COVQCodec &codec)
{
}

int covq_serialize(const COVQCodec &codec, void *buf)
{
}

int covq_deserialize(COVQCodec *codec, const void *buf)
{
}

int covq_save(const COVQCodec *codec, const char *path)
{
}

int covq_load(COVQCodec *codec, const char *path)
{
}

