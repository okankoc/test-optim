/*
 * optim.cpp
 *
 *  Created on: May 15, 2018
 *      Author: okoc
 */

#include "matrix.h"
#include "optim.h"

namespace unconstr_optim {

typedef double (*f_optim)(const Vector & x);
typedef Vector (*df_optim)(const Vector & x);

void grad_descent(const f_optim & f,
                  const df_optim & df,
                  const double & ftol,
                  const double & xtol,
                  Vector & x) {

    // fix a learning rate
    // use that learning rate to descend
    // if function or x value changes less than tol quit!

    Vector x_pre = x;
    double f_diff = 100;
    double x_diff = 100;
    while (f_diff > ftol || x_diff > xtol) {
        x -= df(x) * LEARN_RATE;
        f_diff = fabs(f(x) - f(x_pre));
        x_diff = norm(x - x_pre);
        x_pre = x;
    }
}

}

