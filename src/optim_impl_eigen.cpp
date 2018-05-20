#include "optim.h"
#include <Eigen/Dense>

namespace unconstr_optim {

template<>
double vdot<Eigen::VectorXd>(const Eigen::VectorXd & x1, const Eigen::VectorXd & x2) {
    return x1.dot(x2);
}


template
void grad_descent<Eigen::VectorXd>(const f_optim<Eigen::VectorXd> & f,
                          const df_optim<Eigen::VectorXd> & df,
                          const double & ftol,
                          const double & xtol,
                          const double & learn_rate,
                          Eigen::VectorXd & x);

template
void grad_descent<Eigen::VectorXd>(const f_optim<Eigen::VectorXd> & f,
                          const df_optim<Eigen::VectorXd> & df,
                          const double & ftol,
                          const double & xtol,
                          Eigen::VectorXd & x);

template
double line_search<Eigen::VectorXd>(const f_optim<Eigen::VectorXd> & f,
                           const df_optim<Eigen::VectorXd> & df,
                           const Eigen::VectorXd & direction,
                           const Eigen::VectorXd & x,
                           const double & rho,
                           const double & c);

template
double vnorm<Eigen::VectorXd>(const Eigen::VectorXd & x);

}
