/*
 * optim.cpp
 *
 *  Created on: May 15, 2018
 *      Author: okoc
 */

#include "matrix.h"
#include "optim.h"
#include <armadillo>
#include <Eigen/Dense>

namespace unconstr_optim {

// needed methods for interface unification
static double norm(const Eigen::VectorXd & x) {
    return x.norm();
}

template<typename T>
void grad_descent(const f_optim<T> & f,
                  const df_optim<T> & df,
                  const double & ftol,
                  const double & xtol,
                  const double & learn_rate,
                  T & x) {

    // fix a learning rate
    // use that learning rate to descend
    // if function or x value changes less than tol quit!

    T x_pre = x;
    double f_diff = 100;
    double x_diff = 100;
    while (f_diff > ftol || x_diff > xtol) {
        x -= df(x) * learn_rate;
        f_diff = fabs(f(x) - f(x_pre));
        x_diff = norm(x - x_pre);
        x_pre = x;
    }
}

// add more libraries if you like!

template void grad_descent<Vector>(const f_optim<Vector> & f,
                                   const df_optim<Vector> & df,
                                   const double & ftol,
                                   const double & xtol,
                                   Vector & x);
template void grad_descent<arma::vec>(const f_optim<arma::vec> & f,
                                   const df_optim<arma::vec> & df,
                                   const double & ftol,
                                   const double & xtol,
                                   arma::vec & x);
template void grad_descent<Eigen::VectorXd>(const f_optim<Eigen::VectorXd> & f,
                                   const df_optim<Eigen::VectorXd> & df,
                                   const double & ftol,
                                   const double & xtol,
                                   Eigen::VectorXd & x);

}
