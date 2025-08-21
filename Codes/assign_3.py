# Anuj Arya 
# Roll no. = 2311031

from anuj_library import *
import numpy as np

o1 = Gauss_Jordon_Elimination()
o2 = Matrix_Operation()


# ---------------ASSignment 3 ------------------

def LU_decomposition_verify(A):
        # Dolittle method
        # L matrix
        L = [[0 for _ in range(len(A[0]))] for _ in range(len(A))]
        for i in range(len(L)):
            L[i][i] = 1
        # U matrix
        U = [[0 for _ in range(len(A[0]))] for _ in range(len(A))]
        for j in range(len(A)):
            U[0][j] = A[0][j]

        for j in range(len(A)):
            for i in range(1,len(A)):
                sum1 = 0
                sum2 = 0
                for k in range(0,i):
                    sum1 += L[i][k]*U[k][j]
                if i<= j:
                    U[i][j] = A[i][j] - sum1
                for k in range(0,j):
                    sum2 += L[i][k]*U[k][j]
                if i>j:
                    L[i][j] = (A[i][j] - sum2)/ U[j][j]
        return L,U

# Question-1
A3 = o2.read_matrix('DATA/Assign_03/Input_files/asgn_3_LU')
L,U = LU_decomposition_verify(A3)
print("L matrix = ",L)
print("U matrix = ",U)
print('A matrix(A = LU) = ',o2.matrix_multiply(L,U))

#     OUTPUT
# L matrix =  [[1, 0, 0], [3.0, 1, 0], [2.0, 1.0, 1]]
# U matrix =  [[1.0, 2.0, 4.0], [0, 2.0, 2.0], [0, 0, 3.0]]
# A matrix(A = LU) =  [[1.0, 2.0, 4.0], [3.0, 8.0, 14.0], [2.0, 6.0, 13.0]] 

#Question-2

A33 = o2.read_matrix('DATA/Assign_03/Input_files/asgn3_Q2_A')
b33 = o2.read_matrix('DATA/Assign_03/Input_files/asgn3_b')
print( 'X =',o1.LU_back_frwd(A33,b33))

# OUTPUT 
# X = [0, 2.692648592283628, 4.88764337851929, -2.18274244004171, 1.6989051094890513, -0.0]