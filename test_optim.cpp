/*
 * optim.cpp
 *
 *  Created on: May 13, 2018
 *      Author: okoc
 */

#include <iostream>
#include "optim.h"
#include <armadillo>
#include <Eigen/Dense>
//#include "stdlib.h"

void require(const bool cond, const std::string & message);
void compare_speed_mat_vec_mult();
void check_accuracy_mat_vec_mult();

int main() {

    std::cout << "Testing Optimization library...\n";

    compare_speed_mat_vec_mult();
    check_accuracy_mat_vec_mult();

    return 0;
}

void require(const bool cond, const std::string & message) {
    if (!cond) {
        std::cout << message << std::endl;
        throw std::exception();
    }
}

void check_accuracy_mat_vec_mult() {
    using std::cout;
    using namespace unconstr_optim;
    cout << "\nChecking accuracy of multiplication...\n";
    cout << "Multiplying OWN matrix and vector...\n";
    int m = 4;
    int n = 4;
    Vector in = Vector(n);
    Matrix M = Matrix(m,n);

    cout << "Initializing elements to uniform random values...\n";
    in.set_seed(1);
    in.randu();
    M.set_seed(2);
    M.randu();
    Vector out = M * in;

    cout << "Comparing with ARMADILLO matrices/vectors...\n";
    using namespace arma;
    vec in_arma = zeros<vec>(n);
    mat mat_arma = zeros<mat>(m,n);
    cout << "Initializing elements to uniform random values...\n";

    std::default_random_engine engine;
    engine.seed(1);
    std::uniform_real_distribution<double> distr(0.0, MAX_RANDU);
    in_arma.imbue( [&]() { return distr(engine); } );
    engine.seed(2);
    mat_arma.imbue( [&]() { return distr(engine); } );
    mat_arma = mat_arma.t();
    //in_arma.randu();
    //mat_arma.randu();
    vec out_arma = mat_arma * in_arma;
    raw_vector out_arma_vec = conv_to<raw_vector>::from(out_arma);

    /*cout << in_arma.t();
    in.print("in");
    cout << mat_arma;
    M.print("mat");
    cout << out_arma;
    out.print("out");*/

    require(out.compare(out_arma_vec),"MATRICES ARE NOT EQUAL!");

}

void compare_speed_mat_vec_mult() {

    using std::cout;
    using namespace unconstr_optim;
    cout << "\nComparing speed of matrix-vector multiplication...\n";
    cout << "Multiplying OWN matrix and vector...\n";
    int m = 1000;
    int n = 1000;
    arma::wall_clock timer;
    timer.tic();
    Vector in = Vector(n);
    Matrix M = Matrix(m,n);

    cout << "Initializing elements to uniform random values...\n";
    in.randu();
    M.randu();

    Vector out = M * in;
    //in.print("in");
    //out.print("out");
    double time_own = timer.toc() * 1000;
    cout << "Multiplication took " << time_own << " ms.\n";

    cout << "Comparing with ARMADILLO matrices/vectors...\n";
    using namespace arma;
    timer.tic();
    vec in_arma = zeros<vec>(n);
    mat mat_arma = zeros<mat>(m,n);
    cout << "Initializing elements to uniform random values...\n";
    in_arma.randu();
    mat_arma.randu();
    vec out_arma = mat_arma * in_arma;
    //cout << out_arma.t();
    double time_arma = timer.toc() * 1000;
    cout << "Multiplication took " << time_arma << " ms.\n";

    cout << "Comparing with EIGEN matrices/vectors...\n";
    using namespace Eigen;
    timer.tic();
    VectorXd in_eigen = VectorXd::Zero(n);
    MatrixXd mat_eigen = MatrixXd::Zero(m,n);
    cout << "Initializing elements to uniform random values...\n";
    mat_eigen.setRandom();
    in_eigen.setRandom();
    VectorXd out_eigen = mat_eigen * in_eigen;
    //cout << out_eigen.t();
    double time_eigen = timer.toc() * 1000;
    cout << "Multiplication took " << time_eigen << " ms.\n";

    require(time_own < 3*time_eigen, "MULTIPLICATION IS TOO SLOW!");
    require(time_own < 3*time_arma, "MULTIPLICATION IS TOO SLOW!");

}
