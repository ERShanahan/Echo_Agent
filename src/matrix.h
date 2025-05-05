#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

class matrix {
public:
    int rows, cols;
    std::vector<double> data;

    matrix() : rows(0), cols(0) {}

    matrix(int r, int c);

    double& operator()(int i, int j);
    const double& operator()(int i, int j) const;

    void zero();

    matrix transpose() const;

    template<typename Func>
    matrix apply(Func func) const {
        matrix result(*this);
        for (auto& x : result.data){
            x = func(x);
        }
        return result;
    }

    double fNorm() const;

    void randomize(double mean, double stddev);

    matrix softmax() const;

    double maxElement() const;

    double average() const;

    double abs_average() const;

    matrix shift(int delay) const;

    void copy(const matrix& src);

    void serializeMatrix(std::ostream &out) const;

    void deserializeMatrix(std::istream &in);

    matrix hadamard(const matrix &other) const;

    // Overload operator* for matrix-matrix multiplication
    matrix operator*(const matrix& rhs) const;

    // Overload operator* for matrix-scalar multiplication
    matrix operator*(double scalar) const;

    // Overload operator+ for matrix-matrix addition
    matrix operator+(const matrix& other) const;

    // Overload operator+ for matrix-vector addition
    matrix operator+(const std::vector<double>& vec) const;

    // Overload operator+ for matrix-double addition
    matrix operator+(double scalar) const;

    // Overload operator/ for matrix-scalar division
    matrix operator/(double scalar) const;
};

void print_matrix(const matrix& m);

#endif // MATRIX_H