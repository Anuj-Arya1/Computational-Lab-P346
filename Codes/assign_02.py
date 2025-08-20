
from anuj_library import *
import numpy as np

o1 = Gauss_Jordon_Elimination()
o2 = Matrix_Operation()

A1 = o2.read_matrix('DATA/Assign_02/asgn2_matA')
b = o2.read_matrix('DATA/Assign_02/asgn2_matb')
A = o1.agumented_matrix(A1,b)
print(o1.GJE(A))
A2 = o2.read_matrix('DATA/Assign_02/asgn2_A2')
b1= o2.read_matrix('DATA/Assign_02/asgn2_mat_b2')
A00 = o1.agumented_matrix(A2,b1)
print(o1.GJE(A00))

