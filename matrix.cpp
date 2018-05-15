/*
 * matrix.cpp
 *
 *  Created on: May 13, 2018
 *      Author: okoc
 */

#include <iostream>
#include "optim.h"

namespace unconstr_optim {

Vector::Vector(int m_): m(m_), vec(m) {}

Vector::Vector(std::vector<double> & vals): m(vals.size()), vec(vals) {}

void Vector::init_rand() {
	for (int i = 0; i < m; i++) {
		vec[i] = rand() % MAX_RAND;
	}
}

void Vector::assign(const int idx, const double & val) {
	vec[idx] = val;
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

void Matrix::init_rand() {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			mat[i][j] = rand() % MAX_RAND;
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
