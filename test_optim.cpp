/*
 * optim.cpp
 *
 *  Created on: May 13, 2018
 *      Author: okoc
 */

#include <iostream>
#include "optim.h"
//#include "stdlib.h"

void test_mat_vec_mult();

int main() {

	std::cout << "Testing Optimization library...\n";

	test_mat_vec_mult();

	return 0;
}

void test_mat_vec_mult() {

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
