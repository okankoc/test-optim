#include "optim.h"
#include "matrix.h"

namespace unconstr_optim {

template<>
double vdot<Vector>(const Vector & x1, const Vector & x2) {
    return dot(x1,x2);
}

template
void grad_descent<Vector>(const f_optim<Vector> & f,
                          const df_optim<Vector> & df,
                          const double & ftol,
                          const double & xtol,
                          const double & learn_rate,
                          Vector & x);

template
void grad_descent<Vector>(const f_optim<Vector> & f,
                          const df_optim<Vector> & df,
                          const double & ftol,
                          const double & xtol,
                          Vector & x);

template
double line_search<Vector>(const f_optim<Vector> & f,
                           const df_optim<Vector> & df,
                           const Vector & direction,
                           const Vector & x,
                           const double & rho,
                           const double & c);

template
double vnorm<Vector>(const Vector & x);
}
