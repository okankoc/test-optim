/*
 * optim.cpp
 *
 *  Created on: May 13, 2018
 *      Author: okoc
 */

#include <iostream>
#include <armadillo>
#include <Eigen/Dense>

#include "matrix.h"
#include "optim.h"
#include "test_functions.h"

namespace opt = unconstr_optim;

void require(const bool cond, const std::string & message);
void compare_speed_mat_vec_mult();
void check_accuracy_mat_vec_mult();
void test_grad_descent();
void compare_line_search_with_learn_rate();
void test_grad_descent_multiple_fncs();

int main() {

    std::cout << "Testing Optimization library...\n";

    compare_speed_mat_vec_mult();
    check_accuracy_mat_vec_mult();
    test_grad_descent();
    compare_line_search_with_learn_rate();
    test_grad_descent_multiple_fncs();

    return 0;
}

void test_grad_descent_multiple_fncs() {

    using std::cout;
    using namespace opt;
    cout << "\n\tTesting grad descent on multiple test functions\n";
    std::vector<test_fnc_info<Vector>> test_funcs = init_test_fnc<Vector>();
    const double ftol = 1e-10;
    const double xtol = 1e-10;
    const int DIM = 10;
    Vector x(DIM);
    for(unsigned i = 0; i < test_funcs.size(); i++) {
        x.randn();
        std::cout << "Starting test " << i << " from random x0...\n";
        grad_descent<Vector>(test_funcs[i].f,test_funcs[i].df,ftol,xtol,x);
        x.print("Final value xf");
    }
}

void compare_line_search_with_learn_rate() {

    using namespace opt;
    using std::cout;
    const int DIM = 100;
    const double ftol = 1e-10;
    const double xtol = 1e-10;
    cout << "\n\tComparing line search to fixed learning rate...\n";

    cout << "Grad descent with line search...\n";
    f_optim<Vector> f = f1<Vector>;
    df_optim<Vector> df = df1<Vector>;

    Vector x1(DIM);
    x1.randn();
    //x1.print("Initial value x0");
    arma::wall_clock timer;
    timer.tic();
    Vector x_init = x1;
    grad_descent<Vector>(f,df,ftol,xtol,x1);
    cout << "Optim took " << timer.toc() * 1000 << " ms.\n";
    //x1.print("Final value xf");

    cout << "Grad descent with fixed learn rate 0.1 ...\n";
    Vector x2(DIM);
    x2 = x_init;
    //x2.print("Initial value x0");
    timer.tic();
    const double learn_rate = 0.1;
    grad_descent<Vector>(f,df,ftol,xtol,learn_rate,x2);
    cout << "Optim took " << timer.toc() * 1000 << " ms.\n";
    //x2.print("Final value xf");

    require(f(x1) < f(x2), "FIXED LEARN RATE PERFORMS BETTER!");
}

void test_grad_descent() {

    using namespace opt;
    using std::cout;

    const int DIM = 10;
    const double ftol = 1e-10;
    const double xtol = 1e-10;
    //const double learn_rate = 0.1;

    cout << "\n\tTesting gradient descent...\n";
    f_optim<Vector> f = f1<Vector>;
    df_optim<Vector> df = df1<Vector>;

    Vector x(DIM);
    x.randn();
    x.print("Initial value x0");
    arma::wall_clock timer;
    timer.tic();
    Vector x_init = x;
    //grad_descent<Vector>(f,df,ftol,xtol,learn_rate,x);
    grad_descent<Vector>(f,df,ftol,xtol,x);
    cout << "Optim took " << timer.toc() * 1000 << " ms.\n";
    x.print("Final value xf");

    cout << "Testing gradient descent with ARMADILLO...\n";
    f_optim<arma::vec> f_arma = f1<arma::vec>;
    df_optim<arma::vec> df_arma = df1<arma::vec>;
    arma::vec x_arma = arma::zeros<arma::vec>(DIM);
    //x_arma.randn();
    for (int i = 0; i < DIM; i++)
        x_arma(i) = x_init[i];

    //x_arma.t().print("Initial value x0");
    timer.tic();
    //grad_descent<arma::vec>(f_arma,df_arma,ftol,xtol,learn_rate,x_arma);
    grad_descent<arma::vec>(f_arma,df_arma,ftol,xtol,x_arma);
    cout << "Optim took " << timer.toc() * 1000 << " ms.\n";
    x_arma.t().print("Final value xf");

    cout << "Testing gradient descent with EIGEN...\n";
    f_optim<Eigen::VectorXd> f_eigen = f1<Eigen::VectorXd>;
    df_optim<Eigen::VectorXd> df_eigen = df1<Eigen::VectorXd>;
    Eigen::VectorXd x_eigen = Eigen::VectorXd::Zero(DIM);
    //x_eigen.setRandom();
    for (int i = 0; i < DIM; i++)
        x_eigen(i) = x_init[i];
    //cout << "Initial value x0\n" << x_eigen.transpose() << std::endl;
    timer.tic();
    //grad_descent<Eigen::VectorXd>(f_eigen,df_eigen,ftol,xtol,learn_rate,x_eigen);
    grad_descent<Eigen::VectorXd>(f_eigen,df_eigen,ftol,xtol,x_eigen);
    cout << "Optim took " << timer.toc() * 1000 << " ms.\n";
    cout << "Final value xf\n" << x_eigen.transpose() << std::endl;

    raw_vector x_arma_vec = arma::conv_to<raw_vector>::from(x_arma);
    raw_vector x_eigen_vec(DIM);
    for (int i = 0; i < DIM; i++)
        x_eigen_vec[i] = x_eigen(i);

    cout << "Checking for equality between vectors...\n";
    require(x.compare(x_arma_vec,1e-4),"VECTORS ARE NOT EQUAL!");
    require(x.compare(x_eigen_vec,1e-4),"VECTORS ARE NOT EQUAL!");

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
    cout << "\n\tChecking accuracy of multiplication...\n";
    cout << "Multiplying OWN matrix and vector...\n";
    int m = 100;
    int n = 100;
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

    require(out.compare(out_arma_vec,1e-4),"VECTORS ARE NOT EQUAL!");

}

void compare_speed_mat_vec_mult() {

    using std::cout;
    using namespace unconstr_optim;
    cout << "\n\tComparing speed of matrix-vector multiplication...\n";
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
