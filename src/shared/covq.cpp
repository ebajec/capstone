#include "covq.h"

#include <cassert>

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

COVQTrainingContext *covq_init(
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

// @param X - (dim , count) input dataset
// @param P - (N , N) channel transition probabilities; P(i, j) = p(j|i)
void covq_training_it(
	COVQTrainingContext &ctx,
	const Eigen::MatrixXd &X
)
{
	std::unique_lock<std::mutex> lock(ctx.mutex);
#undef EIGEN_USE_AVX512_GEMM_KERNELS
#define EIGEN_USE_AVX512_GEMM_KERNELS 1

	TIMER_BEGIN(TrainingIteration);

	ctx.Y.setZero();
	ctx.Y.noalias() = ctx.centroids * ctx.P;

	uint32_t num_nonzero_prob = 0;

	for (uint32_t j = 0; j < ctx.N; ++j) {
		double prob_j = (ctx.probs * ctx.P.col(j))(0,0);

		// TODO: Ignore when probability is near zero
		if (fabs(prob_j) < ZERO_PROB_THRESHOLD) {
			ctx.Y.col(j).setZero();
		} else {
			ctx.Y.col(j) /= prob_j;
			++num_nonzero_prob;
		}
	}

	ctx.num_possible_codepoints = num_nonzero_prob;

	ctx.E.noalias() = ctx.Y * ctx.P.transpose();
	ctx.v.noalias() = ctx.P * ctx.Y.colwise().squaredNorm().transpose();


	TIMER_BEGIN(Centroids);
	double d_sum = 0;
	#pragma omp parallel reduction(+:d_sum)
	{
		Eigen::VectorXd d_tls = Eigen::VectorXd::Zero(ctx.M);

		double d_expected_tls = 0;

		#pragma omp for schedule(dynamic)
		for (uint32_t c = 0; c < ctx.count; ++c) {
			uint32_t argmin = 0;

			d_tls.noalias() = ctx.v - 2.0 * (ctx.E.transpose() * X.col(c));
			double d = d_tls.minCoeff(&argmin);
			uint32_t idx = argmin;

			d_sum += d + X.col(c).squaredNorm();
			ctx.mapping[c] = idx;
		}
	}
	TIMER_END(Centroids);
	double d_expected = d_sum / (double)ctx.count;

	printf("Expected distortion: %.6f\n", d_expected);

	// accumulate partition centroids
	ctx.centroids.setZero();
	ctx.probs.setZero();

	for (uint32_t c = 0; c < ctx.count; ++c) {
		uint32_t idx = ctx.mapping[c];
		ctx.probs(0, idx) += 1.0;
		ctx.centroids.col(idx).noalias() += X.col(c);
	}

	const double p_x = 1.0 / X.cols(); 

	ctx.probs *= p_x;
	ctx.centroids *= p_x;

	++ctx.epoch;

	TIMER_END(TrainingIteration);

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

COVQCodec *covq_create_codec(const COVQTrainingContext &ctx)
{
	COVQCodec *codec = new COVQCodec{COVQCodec{
		.dim = ctx.dim,
		.N = ctx.N,
		.M = ctx.M,
		.word_length = ctx.word_length,

		.codebook = ctx.Y,
		.centroids = ctx.centroids,
		.probs = ctx.probs,

		.P = ctx.P
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

