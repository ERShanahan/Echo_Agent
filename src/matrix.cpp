#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <cmath>

#include "matrix.h"
#include "cblas.h"

matrix::matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}

double& matrix::operator()(int i, int j) { return data[i * cols + j]; }
const double& matrix::operator()(int i, int j) const { return data[i * cols + j]; }

void matrix::zero(){
    int n = data.size();
    cblas_dscal(n, 0.0, data.data(), 1);
}

matrix matrix::transpose() const{
    matrix result(cols, rows);
    cblas_domatcopy(CblasRowMajor, CblasTrans, rows, cols,
                    1.0, this->data.data(), cols,
                    result.data.data(), rows);
    return result;
}

matrix matrix::apply(double (*func)(double)) const {
    matrix result(*this);
    for (auto& x : result.data){
        x = func(x);
    }
    return result;
}

double matrix::fNorm() const {
    return cblas_dnrm2(rows * cols, data.data(), 1);
}

void matrix::randomize(double mean, double stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(mean, stddev);
    for (auto& x : data) {
        x = dist(gen);
    }
}

matrix matrix::softmax() const{
    matrix result(rows, cols);
    
    // Process each row independently.
    for (int i = 0; i < rows; i++) {
        // Find maximum value in the row for numerical stability.
        double max_val = (*this)(i, 0);
        for (int j = 1; j < cols; j++) {
            if ((*this)(i,j) > max_val) {
                max_val = (*this)(i,j);
            }
        }
        
        // Compute the exponentials and sum for the row.
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            result(i, j) = std::exp((*this)(i, j) - max_val);
            sum += result(i,j);
        }
        
        // Normalize each element in the row.
        for (int j = 0; j < cols; j++) {
            result(i, j) /= sum;
        }
    }
    return result;
}

double matrix::maxElement() const {
    if (data.empty()) {
        throw std::runtime_error("Matrix is empty");
    }
    // Get the total number of elements.
    int n = data.size();
    // cblas_idamax returns the index of the element with maximum absolute value.
    int maxIndex = cblas_idamax(n, data.data(), 1);
    return data[maxIndex];
}

double matrix::average() const {
    if (data.empty()) {
        throw std::runtime_error("Matrix is empty");
    }
    int n = data.size();
    // Create a vector of ones of the same size as data
    std::vector<double> ones(n, 1.0);
    // Use BLAS ddot to compute the dot product of data and ones (i.e., the sum of data)
    double sum = cblas_ddot(n, data.data(), 1, ones.data(), 1);
    return sum / n;
}

double matrix::abs_average() const {
    if (data.empty()) {
        throw std::runtime_error("Matrix is empty");
    }
    int n = data.size();
    // Use cblas_dasum to compute the sum of absolute values of the data array.
    double abs_sum = cblas_dasum(n, data.data(), 1);
    return abs_sum / n;
}

matrix matrix::shift(int delay) const {
    matrix shifted(rows, cols);
    // Fill the first 'delay' rows with zeros (or a designated start token value).
    for (int i = 0; i < delay && i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            shifted(i, j) = 0.0;
        }
    }
    // Copy the remaining rows: each row i of the original becomes row i+delay.
    for (int i = delay; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            shifted(i, j) = (*this)(i - delay, j);
        }
    }
    return shifted;
}

void matrix::copy(const matrix& src) {
    if (this == &src) return;

    // If dimensions differ, update rows, cols, and resize data vector
    if (rows != src.rows || cols != src.cols) {
        rows = src.rows;
        cols = src.cols;
        data.resize(rows * cols);
    }

    // Copy all elements from src.data into this object's data
    std::copy(src.data.begin(), src.data.end(), data.begin());
}

void matrix::serializeMatrix(std::ostream &out) const {
    out << this->rows << " " << this->cols << "\n";
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            out << (*this)(i, j) << " ";
        }
        out << "\n";
    }
}

void matrix::deserializeMatrix(std::istream &in) {
    int rows, cols;
    in >> rows >> cols;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            in >> (*this)(i, j);
        }
    }
}

matrix matrix::operator*(const matrix& rhs) const {
    if (cols != rhs.rows) {
        std::cout << "Attempted: (" << rows << "," << cols << ") * (" << rhs.rows << "," << rhs.cols << ")" << std::endl;
        throw std::runtime_error("Dimension Problems.");
    }
    matrix result(rows, rhs.cols);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, rhs.cols, cols,
                1.0, this->data.data(), cols,
                     rhs.data.data(), rhs.cols,
                0.0, result.data.data(), rhs.cols);

    return result;
}

matrix matrix::operator*(double scalar) const {
    matrix result(*this);
    int n = result.data.size();
    cblas_dscal(n, scalar, result.data.data(), 1);
    return result;
}

matrix matrix::operator+(const matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions mismatch for addition");
    }
    matrix result(*this);
    int n = rows * cols;
    cblas_daxpy(n, 1.0, other.data.data(), 1, result.data.data(), 1);
    return result;
}

matrix matrix::operator+(const std::vector<double>& vec) const {
    if (vec.size() != static_cast<size_t>(rows)) {
        throw std::invalid_argument("Vector size mismatch for matrix addition");
    }
    matrix result(*this);
    for (int j = 0; j < cols; j++){
        cblas_daxpy(rows, 1.0, vec.data(), 1, &result.data[j], cols);
    }
    return result;
}

matrix matrix::operator+(double scalar) const {
    matrix result(*this);
    int n = rows * cols;
    std::vector<double> ones(n, 1.0);
    cblas_daxpy(n, scalar, ones.data(), 1, result.data.data(), 1);
    return result;
}

matrix matrix::operator/(double scalar) const {
    if (scalar == 0.0) {
        throw std::invalid_argument("Division by zero in matrix operator/.");
    }
    matrix result(*this);
    double inv = 1.0 / scalar;
    int n = result.data.size();
    cblas_dscal(n, inv, result.data.data(), 1);
    return result;
}

void print_matrix(const matrix& m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            std::cout << m(i, j) << " ";
        }
        std::cout << std::endl;
    }
}