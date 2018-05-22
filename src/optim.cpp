/*
 * optim.cpp
 *
 *  Created on: May 15, 2018
 *      Author: okoc
 */

#include "optim.h"
#include "matrix.h"
#include <armadillo>
#include <Eigen/Dense>

namespace unconstr_optim {

template<typename vec>
void grad_descent(const f_optim<vec> & f,
                  const df_optim<vec> & df,
                  const double & ftol,
                  const double & xtol,
                  const double & learn_rate,
                  vec & x) {

    // fix a learning rate
    // use that learning rate to descend
    // if function or x value changes less than tol quit!

    vec x_pre = x;
    double f_diff = 100;
    double norm_diff = 100;
    vec xdiff = x;
    while (f_diff > ftol || norm_diff > xtol) {
        x -= df(x) * learn_rate;
        f_diff = fabs(f(x) - f(x_pre));
        xdiff = x - x_pre;
        norm_diff = vnorm(xdiff);
        x_pre = x;
    }
}

template<typename vec>
void grad_descent(const f_optim<vec> & f,
                  const df_optim<vec> & df,
                  const double & ftol,
                  const double & xtol,
                  vec & x) {

    vec x_pre = x;
    double f_diff = 100;
    double norm_diff = 100;
    double learn_rate = 1;
    vec direction = -df(x);
    vec x_diff = x;
    while (f_diff > ftol || norm_diff > xtol) {
        direction = -df(x);
        learn_rate = line_search(f,df,direction,x,0.5,0.0001);
        x += direction * learn_rate;
        f_diff = fabs(f(x) - f(x_pre));
        x_diff = x - x_pre;
        norm_diff = vnorm(x_diff);
        x_pre = x;
    }
}

template<typename vec, typename mat>
void newtons_method(const f_optim<vec> & f,
                    const df_optim<vec> & df,
                    const ddf_optim<vec,mat> & ddf,
                    const double & ftol,
                    const double & xtol,
                    vec & x) {

    vec x_pre = x;
    double f_diff = 100;
    double norm_diff = 100;
    double learn_rate = 1;
    vec direction = -df(x);
    vec x_diff = x;
    mat hessian;
    while (f_diff > ftol || norm_diff > xtol) {
        hessian = ddf(x);
        direction = inv(hessian,-df(x));
        learn_rate = line_search(f,df,direction,x,0.5,0.0001);
        x += direction * learn_rate;
        f_diff = fabs(f(x) - f(x_pre));
        x_diff = x - x_pre;
        norm_diff = vnorm(x_diff);
        x_pre = x;
    }
}

template<typename vec>
double line_search(const f_optim<vec> & f,
                   const df_optim<vec> & df,
                   const vec & direction,
                   const vec & x,
                   const double & rho, // usually 1/2
                   const double & c) {// usually 10^-4
    double alpha = 1.0;
    double val_pre = f(x);
    vec grad = df(x);
    vec x_search = x + (direction * alpha);
    double val = f(x);
    while (val > val_pre + c*vdot(grad,direction)) {
        alpha *= rho;
        x_search = x + (direction*alpha);
        val = f(x_search);
    }
    return alpha;
}

template<typename vec>
double vnorm(const vec & x) {
    return sqrt(vdot(x,x));
}

}

#include "optim_impl_vector.cpp"
#include "optim_impl_arma.cpp"
#include "optim_impl_eigen.cpp"
