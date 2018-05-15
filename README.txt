CODING UNCONSTRAINED OPTIMIZATION IN C++

"What I cannot create, I do not understand." - Feynman

* Implement vector and matrix types -> DONE
* Allocate/deallocate vectors and matrices -> DONE
* Introduce class structure for vectors and matrices -> DONE
* Add support for Eigen and Armadillo vectors
* Add easy switching between types/libraries
* Test gradient descent for an arbitrary function
* Add line search
* Compute Hessian matrix
* Add matrix inversion
* Compare speed of matrix multiplication/inversion with mine/Eigen/Arma
* Test Newton's method with line search
* Add trust region method (Nocedal et al.)
* Add Quasi-Newton method (BFGS)
* Extend my vector & matrix type based comp. with BLAS/LAPACK
* Compare with NLOPT routines for speed

Using: make, C++11

Things to read throughout:
- C++ tips
- Numerical Recipes
- Gradient Descent + Line search proof
- NEWTON proof
- QN proof

Notes:
Including typedefs
Initially using malloc, and free
Checking leak with valgrind
Moved to C++ (new/delete)
Introduced classes for Vector and Matrix
Introduced inner_product and multiplication methods
Introduced std::vector instead of arrays 
Removed destructors after introducing vectors