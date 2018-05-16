/*
 * optim.cpp
 *
 *  Created on: May 13, 2018
 *      Author: okoc
 */

#include <iostream>

#include "../matrix.h"
//#include "stdlib.h"

typedef double* vec;
typedef double** mat;

void multiply_mat_vec(const int m, const int n,
		              const mat & M, const vec & v_in, vec & v_out);
void print_vec(const int m, const std::string mes, const vec & v);
void init_rand_mat(const int m, int n, const int cap, mat & M);
void init_rand_vec(const int m, const int cap, vec & v);
vec alloc_vec(const int m);
mat alloc_mat(const int m, const int n);
void free_vec(vec & v);
void free_mat(const int m, mat & M);
void test_mat_vec_mult();
void test_mat_vec_mult_new();

int main() {

	std::cout << "Testing Optimization library...\n";

	test_mat_vec_mult();
	test_mat_vec_mult_new();

	return 0;
}

void test_mat_vec_mult() {

	std::cout << "Multiplying matrices...\n";

	int m = 4;
	int n = 2;

	// allocate and initialize
	vec vec1 = alloc_vec(n);
	mat mat1 = alloc_mat(m,n);
	vec out = alloc_vec(m);
	init_rand_mat(m,n,10,mat1);
	init_rand_vec(n,10,vec1);

	// operation
	multiply_mat_vec(m,n,mat1,vec1,out);

	// print the results
	print_vec(n, "in", vec1);
	print_vec(m, "out", out);

	// free mats and vectors
	free_vec(vec1);
	free_vec(out);
	free_mat(m,mat1);
}

void test_mat_vec_mult_new() {

	using namespace unconstr_optim;
	std::cout << "Multiplying matrices #2...\n";
	int m = 4;
	int n = 2;
	Vector in = Vector(n);
	Matrix mat = Matrix(m,n);
	in.init_rand();
	mat.init_rand();

	Vector out = mat * in;
	in.print("in");
	out.print("out");
}

void init_rand_mat(const int m, const int n, const int cap, mat & M) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			M[i][j] = rand() % cap;
		}
	}
}

void init_rand_vec(const int m, const int cap, vec & v) {
	for (int i = 0; i < m; i++) {
		v[i] = rand() % cap;
	}
}

void multiply_mat_vec(const int m, const int n,
		              const mat & M, const vec & v_in, vec & v_out) {

	double temp;
	for (int i = 0; i < m; i++) {
		temp = 0.0;
		for (int j = 0; j < n; j++) {
			temp += M[i][j] * v_in[j];
		}
		v_out[i] = temp;
	}
}

void print_vec(const int m, const std::string mes, const vec & v) {
	using namespace std;
	cout << mes << " = [";
	for (int i = 0; i < m-1; i++) {
		cout << v[i] << " ";
	}
	cout << v[m-1] << "]" << endl;
}

void free_vec(vec & v) {
	//free(v);
	delete[] v;
}

void free_mat(const int m, mat & M) {
	for (int i = 0; i < m; i++) {
		delete[] M[i]; //free(M[i]);
	}
	delete[] M; //free(M);
}

vec alloc_vec(const int m) {
	//return (vec)malloc(m * sizeof(double));
	return new double[m];
}

mat alloc_mat(const int m, const int n) {
	/*mat M = (mat)malloc(m * sizeof(double*));
	for (int i = 0; i < m; i++) {
		M[i] = (vec)malloc(n * sizeof(double));
	}
	return M;*/
	mat M = new vec[m];
	for (int i = 0; i < m; i++) {
		M[i] = new double[n];
	}
	return M;
}
