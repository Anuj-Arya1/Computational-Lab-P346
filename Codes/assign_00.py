# Anuj Arya, Roll no. 2311031 

from anuj_library import *

c1 = Number()
c2 = Series()
c3 = Matrix_Operation()
c4 = Mycomplex(1.3,-2.2)


A = c3.read_matrix("asgn0_matA")
B = c3.read_matrix("asgn0_matB")
C = c3.read_matrix("asgn0_vecC")
D = c3.read_matrix("asgn0_vecD")

print("Sum of first 20 odd numbers")
print(c1.sum_of_odd_numbers(20),"\n")
print("Factorial of 8")
print(c1.fact(8),"\n")
print("Sum of first 15 terms of GP")
print(c2.sum_of_GP(15, 1.25, 0.5),"\n")
print("Sum of first 15 terms of HP")
print(c2.sum_of_HP(15, 1.25, 1.5), "\n")
print("Matrix multiplication of A and B")
print(c3.matrix_multiply(A, B), "\n")
print("Dot product of C and D")
print(c3.dot_product(C, D), "\n")
print("Matrix multiplication of B and C")
print(c3.matrix_multiply(B,C), "\n")
print("Sum of given complex numbers")
print(c4.sum(-0.8,1.7), "\n")
print("Difference of given complex numbers")
print(c4.difference(-0.8,1.7), "\n")
print("Product of given complex numbers")
print(c4.product(-0.8,1.7), "\n")
print("Modulus of given complex numbers")
print(c4.modulus(1.3,-2.2), "\n")
print(c4.modulus(-0.8,1.7), "\n")

# OUTPUT OF THE CODE

'''
Sum of first 20 odd numbers
1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 400 

Factorial of 8
40320 

Sum of first 15 terms of GP
[1.25, 0.625, 0.3125, 0.15625, 0.078125, 0.0390625, 0.01953125, 0.009765625, 0.0048828125, 0.00244140625, 0.001220703125, 0.0006103515625, 0.00030517578125, 0.000152587890625, 7.62939453125e-05]
2.4999237060546875 

Sum of first 15 terms of HP
[0.8, 0.36363636363636365, 0.23529411764705882, 0.17391304347826086, 0.13793103448275862, 0.11428571428571428, 0.0975609756097561, 0.0851063829787234, 0.07547169811320754, 0.06779661016949153, 0.06153846153846154, 0.056338028169014086, 0.05194805194805195, 0.04819277108433735, 0.0449438202247191]
2.413957073365919 

Matrix multiplication of A and B
[[-0.3000000000000007, -3.5, 5.2], [-4.5, -2.0, 4.5], [9.3, 0.8, -7.0]] 

Dot product of C and D
-3.5 

Matrix multiplication of B and C
[[1.0], [-5.75], [-9.0]] 

Sum of given complex numbers
(0.5-0.5000000000000002j) 

Difference of given complex numbers
(2.1-3.9000000000000004j) 

Product of given complex numbers
(2.7+3.97j) 

Modulus of given complex numbers
2.5553864678361276 

1.8788294228055935 
'''