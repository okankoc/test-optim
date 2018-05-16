/*
 * optim.h
 *
 *  Created on: May 15, 2018
 *      Author: okoc
 */

#ifndef EX_OPTIM_H_
#define EX_OPTIM_H_

#include "matrix.h"

namespace unconstr_optim {

const double LEARN_RATE = 0.1;

typedef double (*f_optim)(const Vector & x);
typedef Vector (*df_optim)(const Vector & x);

void grad_descent(const f_optim & f,
                  const df_optim & df,
                  const double & ftol,
                  const double & xtol,
                  Vector & x);
}


#endif /* EX_OPTIM_OPTIM_H_ */
