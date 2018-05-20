#include "ex_template.h"
#include <armadillo>

template<>
int add<int>(const int & v1, const int & v2) {
    return v1 + v2;
}

template<>
double add<double>(const double & v1, const double & v2) {
    return v1 + v2;
}

template<>
arma::vec add<arma::vec>(const arma::vec & v1, const arma::vec & v2) {
    return v1 + v2;
}

template<>
double vdot<arma::vec>(const arma::vec & x1, const arma::vec & x2) {
    return dot(x1,x2);
}
