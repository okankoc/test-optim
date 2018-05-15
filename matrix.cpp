/*
 * matrix.cpp
 *
 *  Created on: May 13, 2018
 *      Author: okoc
 */

#include <iostream>
#include <random>
#include "optim.h"

namespace unconstr_optim {

Vector::Vector(int m_): m(m_), vec(m) {}

Vector::Vector(std::vector<double> & vals): m(vals.size()), vec(vals) {}

void Vector::set_seed(const int seed_) {
    engine.seed(seed_);
}

void Vector::randu() {
    std::uniform_real_distribution<double> distribution(0.0,MAX_RANDU);
    for (int i = 0; i < m; i++) {
        vec[i] = distribution(engine);
    }
}

void Vector::randn() {
    std::normal_distribution<double> distr(0.0,VAR_RANDN);
    for (int i = 0; i < m; i++) {
        vec[i] = distr(engine);
    }
}

void Vector::assign(const int idx, const double & val) {
    vec[idx] = val;
}

bool Vector::compare(const raw_vector & vec_) const {

    return (vec_ == this->vec);
}

void Vector::print() const {

    using namespace std;
    cout << "vec = [";
    for (int i = 0; i < m-1; i++) {
        cout << vec[i] << " ";
    }
    cout << vec[m-1] << "]" << endl;
}

void Vector::print(const std::string & mes) const {

    using namespace std;
    cout << mes << " = [";
    for (int i = 0; i < m-1; i++) {
        cout << vec[i] << " ";
    }
    cout << vec[m-1] << "]" << endl;
}

double Vector::inner_prod(const Vector & vec_two) const {

    double out = 0.0;
    for (int i = 0; i < m; i++)
        out += vec[i] * vec_two.vec[i];
    return out;
}

double Vector::inner_prod(const std::vector<double> & vec_two) const {

    double out = 0.0;
    for (int i = 0; i < m; i++)
        out += vec[i] * vec_two[i];
    return out;
}

Matrix::Matrix(int m_, int n_) : m(m_), n(n_), mat(m,std::vector<double>(n)) {}

void Matrix::print() const {

    using namespace std;
    cout << "mat = [\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n-1; j++) {
            cout << mat[i][j] << " ";
        }
        cout << mat[i][n-1] << endl;
    }
    cout << "]\n";
}

void Matrix::print(const std::string & mes) const {

    using namespace std;
    cout << mes << " = [\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n-1; j++) {
            cout << mat[i][j] << " ";
        }
        cout << mat[i][n-1] << endl;
    }
    cout << "]\n";
}

void Matrix::set_seed(const int seed_) {
    engine.seed(seed_);
}

void Matrix::randu() {
    std::uniform_real_distribution<double> distribution(0.0,MAX_RANDU);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat[i][j] = distribution(engine);
        }
    }
}

void Matrix::randn() {
    std::normal_distribution<double> distr(0.0,VAR_RANDN);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat[i][j] = distr(engine);
        }
    }
}

Vector Matrix::operator*(const Vector & in) const {
    Vector out(m);
    mult_vec(in,out);
    return out;
}

void Matrix::mult_vec(const Vector & in, Vector & out) const {

    for (int i = 0; i < m; i++) {
        out.assign(i,in.inner_prod(mat[i]));
    }
}

}
