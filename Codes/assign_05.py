# Anuj Arya 
# Roll no. = 2311031

from anuj_library import *
import numpy as np

o1 = Gauss_Jordon_Elimination()
o2 = Matrix_Operation()


Aq1 = o2.read_matrix('DATA/Assign_05/Input_files/asgn05_Q1_A')
bq1 = o2.read_matrix('DATA/Assign_05/Input_files/asgn05_Q1_b')
Aq2 = o2.read_matrix('DATA/Assign_05/Input_files/asgn_05_Q2_A')
bq2 = o2.read_matrix('DATA/Assign_05/Input_files/asgn_05_Q2_b')

# Q1 

print(o1.symmetry_check(Aq1))
print(o1.cholesky_forwd_back(Aq1,bq1))
print(o1.Gauss_seidel(Aq1,bq1))

# Q2

Aq2[2],Aq2[4] = Aq2[4],Aq2[2]
bq2[2],bq2[4] = bq2[4],bq2[2]

Aq2[0],Aq2[3] = Aq2[3],Aq2[0]
bq2[0],bq2[3] = bq2[3],bq2[0]

print(o1.Jacobi_iterative(Aq2,bq2))
print(o1.Gauss_seidel(Aq2,bq2))


# ==================== OUTPUT =================
# Question 1

# Its symmetric
# [1.0, 0.9999999999999999, 1.0, 1.0, 1.0, 1.0]
# No. of iterations for convergence 15
# [0.9999997530614102, 0.9999997892247294, 0.9999999100460266, 0.9999998509593769, 0.9999998727858708, 0.9999999457079743]

# Question 2

# No. of iterations for convergence 59
# [2.979165056795253, 2.215599391485432, 0.2112838649649856, 0.15231675099384495, 5.7150334073492335]
# No. of iterations for convergence 11
# [2.979165086347139, 2.2155996761867414, 0.21128402698819163, 0.15231700827754785, 5.715033568811629]