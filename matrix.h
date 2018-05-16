/*
 * optim.h
 *
 *  Created on: May 14, 2018
 *      Author: okoc
 */

#ifndef EX_MATRIX_H_
#define EX_MATRIX_H_

#include <vector>
#include <random>

typedef std::vector<double> raw_vector;
typedef std::vector<std::vector<double> > raw_matrix;

namespace unconstr_optim {

const double MAX_RANDU = 1.0; //!< uniform random number generation limit
const double VAR_RANDN = 1.0; //!< gaussian r.v. variance

class Vector {

private:
    int m; //!< number of elements
    raw_vector vec;
    std::default_random_engine engine = std::default_random_engine();
    double inner_prod(const Vector & v) const;

public:
    Vector(int m);
    Vector(std::vector<double> & vals);
    void assign(const int idx, const double & val);
    void print() const;
    void print(const std::string & message = "") const;
    void set_seed(const int seed);
    void randu();
    void randn();
    bool compare(const raw_vector & vec_, const double & max_diff) const;
    double inner_prod(const raw_vector & vec_two) const;
    double operator*(const Vector & vec_two) const;
    Vector operator*(const double & val) const;
    Vector operator-(const Vector & vec_two) const;
    Vector operator-(const double & val) const;
    Vector operator+(const double & val) const;
    Vector & operator+=(const double & val);
    Vector & operator+=(const Vector & vec_two);
    Vector & operator-=(const double & val);
    Vector & operator-=(const Vector & vec_two);
    Vector & operator*=(const double & val);
};

class Matrix {

private:
    int m; //!< number of rows
    int n; //!< number of columns
    raw_matrix mat;
    std::default_random_engine engine = std::default_random_engine();
    void mult_vec(const Vector & in, Vector & out) const;

public:
    Matrix(int m, int n);
    void print() const;
    void print(const std::string & message) const;
    void set_seed(const int seed);
    void randu();
    void randn();
    Vector operator*(const Vector & in) const;
};

double norm(const Vector & vec);

}

#endif /* EX_OPTIM_H_ */
