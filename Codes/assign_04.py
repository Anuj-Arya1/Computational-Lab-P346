# Anuj Arya 
# Roll no. = 2311031

from anuj_library import *
import numpy as np

o1 = Gauss_Jordon_Elimination()
o2 = Matrix_Operation()

A = o2.read_matrix('DATA/Assign_04/Input_files/asgn_04_A_matrix')
b = o2.read_matrix('DATA/Assign_04/Input_files/asgn_04_b')
L,L_t = o1.choleski_decomposition(A)
print("Solution of linear equation:",o1.cholesky_forwd_back(A,b))
print(L)
print(L_t)
print(o2.matrix_multiply(L,L_t))
print(o1.Jacobi_iterative(A,b))

# --------------------- OUTPUT ----------------------------------------- 
# Question 1
# L = [[2.0, 0, 0, 0], [0.5, 1.6583123951777, 0, 0], [0.5, -0.753778361444409, 1.087114613009218, 0], [0.5, 0.45226701686664544, 0.08362420100070905, 1.2403473458920844]]
# L_transpose = [[2.0, 0.5, 0.5, 0.5], [0, 1.6583123951777, -0.753778361444409, 0.45226701686664544], [0, 0, 1.087114613009218, 0.08362420100070905], [0, 0, 0, 1.2403473458920844]]
# [[4.0, 1.0, 1.0, 1.0], [1.0, 3.0, -1.0, 1.0], [1.0, -1.0, 1.9999999999999998, 0.0], [1.0, 1.0, 0.0, 1.9999999999999996]]

# Question 2
# No. of iterations for convergence 42
# [2.980232238769531e-07, 1.0, 0.9999997019767761, 1.0000002980232239]



