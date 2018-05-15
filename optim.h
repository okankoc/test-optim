/*
 * optim.h
 *
 *  Created on: May 14, 2018
 *      Author: okoc
 */

#ifndef EX_OPTIM_H_
#define EX_OPTIM_H_

#include <vector>

typedef std::vector<double> raw_vector;
typedef std::vector<std::vector<double> > raw_matrix;

namespace unconstr_optim {

class Vector {

private:
	int m;
	int MAX_RAND = 10;
	raw_vector vec;

public:
	Vector(int m);
	Vector(std::vector<double> & vals);
	void assign(const int idx, const double & val);
	void print() const;
	void print(const std::string & message = "") const;
	void init_rand();
	double inner_prod(const Vector & v) const;
	double inner_prod(const std::vector<double> & vec_two) const;
};

class Matrix {

private:
	int m;
	int n;
	int MAX_RAND = 10;
	raw_matrix mat;
	void mult_vec(const Vector & in, Vector & out) const;

public:
	Matrix(int m, int n);
	void print() const;
	void print(const std::string & message) const;
	void init_rand();

	Vector operator*(const Vector & in) const;
};

}

#endif /* EX_OPTIM_H_ */
