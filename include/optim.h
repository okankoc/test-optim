/*
 * optim.h
 *
 *  Created on: May 15, 2018
 *      Author: okoc
 */

#ifndef EX_OPTIM_H_
#define EX_OPTIM_H_

namespace unconstr_optim {

// alias declarations
template<typename vec>
using f_optim = double (*)(const vec & x);

template<typename vec>
using df_optim = vec (*)(const vec & x);

template<typename vec, typename mat>
using ddf_optim = mat (*)(const vec & x);

template<typename vec, typename mat>
void newtons_method(const f_optim<vec> & f,
                    const df_optim<vec> & df,
                    const ddf_optim<vec,mat> & ddf,
                    const double & ftol,
                    const double & xtol,
                    vec & x);

/**
 * Gradient descent with fixed learning rate
 */
template<typename vec>
void grad_descent(const f_optim<vec> & f,
                  const df_optim<vec> & df,
                  const double & ftol,
                  const double & xtol,
                  const double & learn_rate,
                  vec & x);

/**
 * Gradient descent with adjustable learning rate.
 * Learning rate is adjusted each iteration with line search;
 */
template<typename vec>
void grad_descent(const f_optim<vec> & f,
                  const df_optim<vec> & df,
                  const double & ftol,
                  const double & xtol,
                  vec & x);

/**
 * Backtracking line search
 */
template<typename vec>
double line_search(const f_optim<vec> & f,
                   const df_optim<vec> & df,
                   const vec & direction,
                   const vec & x,
                   const double & rho, // usually 1/2
                   const double & c);

// required for interface unification
template<typename vec>
double vnorm(const vec & x);

template<typename vec>
double vdot(const vec & x1, const vec & x2);

}

#endif /* EX_OPTIM_OPTIM_H_ */
