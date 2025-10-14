# Anuj Arya 
# Roll no. = 2311031

from anuj_library import *
import numpy as np

o1 = Gauss_Jordon_Elimination()
o2 = Matrix_Operation()

A1 = o2.read_matrix('DATA/Assign_02/Input_files/asgn2_matA')
b = o2.read_matrix('DATA/Assign_02/Input_files/asgn2_matb')
A = o1.agumented_matrix(A1,b)
print(o1.GJE(A))
A2 = o2.read_matrix('DATA/Assign_02/Input_files/asgn2_A2')
b1= o2.read_matrix('DATA/Assign_02/Input_files/asgn2_mat_b2')
A00 = o1.agumented_matrix(A2,b1)
print(o1.GJE(A00))

# print(o1.inverse_matrix(A1))
# print(o1.determinant(A1))



# ----------------------- OUTPUT
# Ques=1
# [[1.0, 0.0, 0.0, -2.0], [0.0, 1.0, 0.0, -2.0], [0.0, 0.0, 1.0, 1.0]]
# Ques=2
# [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.7618170439978598], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.8962280338740136],
#[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 4.051931404116158], [-0.0, -0.0, -0.0, 1.0, 0.0, 0.0, -1.6171308025395423],
#[-0.0, -0.0, -0.0, -0.0, 1.0, 0.0, 2.0419135385019125], [-0.0, -0.0, -0.0, -0.0, -0.0, 1.0, 0.1518324871559355]]

