#include <iostream>
#include "ex_template.h"
#include <armadillo>

int main() {

    using std::cout;
    using std::endl;
    using namespace arma;

    cout << "Adding two ints...\n";
    add(1,2);
    cout << "Adding two doubles...\n";
    add(1.0,2.0);
    cout << "Adding two armadillo vectors...\n";
    vec v1 = zeros<vec>(10);
    v1.randn();
    vec v2 = zeros<vec>(10);
    v2.randu();
    add(v1,v2);
    cout << "Dot product of two armadillo vectors...\n";
    cout << vdot(v1,v2) << endl;


    return 1;
}
