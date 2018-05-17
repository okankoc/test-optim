/*
 * optim.h
 *
 *  Created on: May 15, 2018
 *      Author: okoc
 */

#ifndef EX_OPTIM_H_
#define EX_OPTIM_H_

#include "matrix.h"
#include <armadillo>

namespace unconstr_optim {

// alias declarations
template<typename T>
using f_optim = double (*)(const T & x);

template<typename T>
using df_optim = T (*)(const T & x);

template<typename T>
void grad_descent(const f_optim<T> & f,
                  const df_optim<T> & df,
                  const double & ftol,
                  const double & xtol,
                  const double & learn_rate,
                  T & x);
}

#endif /* EX_OPTIM_OPTIM_H_ */
