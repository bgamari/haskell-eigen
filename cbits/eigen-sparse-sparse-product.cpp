#include <Eigen/Sparse>
#include <iostream>

typedef Eigen::SparseMatrix<float> M;

// Compute A * B^T
M* a_bt(double thresh, M& a, M& b) {
  return new M((a * b.transpose()).pruned(thresh).triangularView<Eigen::Lower>());
}

extern "C" {
  const char* eigen_sparse_a_bt(float thresh, void* a, void* b, void** out) {
  M *aa = (M *) a;
  M *bb = (M *) b;
  *(M**)out = a_bt(thresh, *aa, *bb);
  return NULL;
}
}

// Compute A^T * A
template<typename T, typename ResultType>
void at_a_naive(const double thresh, const Eigen::SparseMatrix<T>& a, ResultType& res) {
  typedef typename Eigen::SparseMatrix<T> Mat;
  typedef typename Eigen::SparseMatrix<T>::StorageIndex Index;

  using namespace Eigen;
  Index n = a.outerSize();
  res.resize(n, n);
  res.reserve(2*a.nonZeros());

  for (Index i = 0; i < n; ++i) {
    res.startVec(i);
    for (Index j = 0; j < n; ++j) {
      T accum = 0;
      typename Mat::InnerIterator ii(a, i);
      typename Mat::InnerIterator jj(a, j);

      while (ii && jj) {
        if (ii.index() < jj.index()) {
          ++ii;
        } else if (jj.index() < ii.index()) {
          ++jj;
        } else {
          accum += ii.value() * jj.value();
          ++ii; ++jj;
        }
      }

      if (accum > thresh) {
        res.insertBackByOuterInner(i, j) = accum;
      }
    }
  }

  res.finalize();
}

#define TEST 0
#if TEST

#include <random>
#include <chrono>

// construct a random, square, sparse, positive semidefinite matrix
// of the given density and size
template<typename T>
Eigen::SparseMatrix<T> random_sparse(float density, int n) {
  std::mt19937 gen;
  std::uniform_int_distribution<> idxDist(0, n-1);
  std::uniform_real_distribution<T> valDist(0, 5);

  Eigen::SparseMatrix<T> m(n,n);
  const int nonzeros = density * n * n;
  std::vector<Eigen::Triplet<float>> triplets;
  triplets.reserve(nonzeros);
  for (int idx=0; idx < nonzeros; idx++) {
    int i = idxDist(gen);
    int j = idxDist(gen);
    int val = valDist(gen);
    Eigen::Triplet<float> t(i,j,val);
    triplets.push_back(t);
  }
  m.setFromTriplets(triplets.begin(), triplets.end());
  return m;
}

template<typename Return>
Return time_it(std::string description, std::function<Return()> func) {
  using namespace std::chrono;
  milliseconds start = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
  Return ret = func();
  milliseconds end = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
  std::cout << description << " took " << (end - start).count() << " ms" << std::endl;
  return ret;
}

void at_a_test() {
  const int n = 2;
  Eigen::MatrixXf test(n,n);
  //test << 1, 2, 2, 1;
  test = random_sparse<float>(0.1, 2000);
  Eigen::SparseMatrix<float> sparse(test.sparseView());

  Eigen::MatrixXf dense;
  Eigen::SparseMatrix<float> truth, result;
  time_it<int>("dense", [&] {dense = test.transpose() * test; return 0;});
  time_it<int>("eigen", [&] {truth = sparse.transpose() * sparse; return 0;});
  time_it<int>("naive", [&] {at_a_naive(1e-14, sparse, result); return 0;});

  std::cout << dense.sum() << std::endl;
  std::cout << truth.sum() << "\t" << (truth - dense.sparseView()).squaredNorm() << std::endl;
  std::cout << result.sum() << "\t" << (result - dense.sparseView()).squaredNorm() << std::endl;
}

void simple_test() {
  const int n = 2;
  Eigen::MatrixXf test(n,n);
  test <<
    1, 2,
    2, 1;

  M sparse(n,n);
  sparse = test.sparseView();
  M* res = a_bt(0, sparse, sparse);

  std::cout << test << std::endl << std::endl;
  std::cout << Eigen::MatrixXf(*res) << std::endl;
}

int main() {
  //simple_test();
  at_a_test();
  return 0;
}
#endif
