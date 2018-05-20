#include "optim.h"
#include <armadillo>

namespace unconstr_optim {

template<>
double vdot<arma::vec>(const arma::vec & x1, const arma::vec & x2) {
    return dot(x1,x2);
}

template
void grad_descent<arma::vec>(const f_optim<arma::vec> & f,
                          const df_optim<arma::vec> & df,
                          const double & ftol,
                          const double & xtol,
                          const double & learn_rate,
                          arma::vec & x);

template
void grad_descent<arma::vec>(const f_optim<arma::vec> & f,
                          const df_optim<arma::vec> & df,
                          const double & ftol,
                          const double & xtol,
                          arma::vec & x);

template
double line_search<arma::vec>(const f_optim<arma::vec> & f,
                           const df_optim<arma::vec> & df,
                           const arma::vec & direction,
                           const arma::vec & x,
                           const double & rho,
                           const double & c);

template
double vnorm<arma::vec>(const arma::vec & x);
}
