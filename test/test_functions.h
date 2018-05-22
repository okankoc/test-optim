/*
 * Test functions for unconstrained optimization are located here.
 */

#include "optim.h"
#include <vector>

namespace opt = unconstr_optim;

// cost and grad function templates

template<typename vec>
struct test_fnc_info {

    // function
    opt::f_optim<vec> f;
    opt::df_optim<vec> df;

    //template<typename vec>
    //mat hessian(const vec & x);
};

// cost and grad function templates
template<typename vec>
double f1(const vec & x) {
    return opt::vdot(x,x);
}

template<typename vec>
vec df1(const vec & x) {
    return x*2.0;
}

template<typename vec>
double f2(const vec & x) {
    return opt::vdot(x-2,x-3);
}

template<typename vec>
vec df2(const vec & x) {
    return x*2.0 - 5.0;
}

template<typename vec>
std::vector<test_fnc_info<vec>> init_test_fnc() {

    std::vector<test_fnc_info<vec>> tests;
    test_fnc_info<vec> info1;
    info1.f = f1<vec>;
    info1.df = df1<vec>;
    tests.push_back(info1);
    test_fnc_info<vec> info2;
    info2.f = f2<vec>;
    info2.df = df2<vec>;
    tests.push_back(info2);

    return tests;

}
