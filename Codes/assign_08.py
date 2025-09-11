
# Anuj Arya 
# Roll no. = 2311031

from anuj_library import *
import math

o1 = Gauss_Jordon_Elimination()
o2 = Matrix_Operation()

# fixed point method

def g(x1,x2,x3):
    return [(37-x2)**(0.5),(x1-5)**(0.5),3-x1-x2]

x=[6,5,4] # initial guess
o1.count = 0
res1,count1 = o1.fixed_pt_multi_val(g,x)
print("The solution obtained is ",res1,"; No. of iterations = ",count1)

# newton-rapson

def jacobian(x1,x2,x3):
    return [[2*x1,1,0],[1,-2*x2,0],[1,1,1]]

def F(x1,x2,x3):
    return [[x1**2+x2-37], [x1-x2**2-5], [x1+x2+x3-3]]
o1.count = 0
x=[5,5,5] # initial guess
res2,count2 = o1.newton_rapson_mul(x,jacobian,F)
print("The solution obtained is ",res2,"; No. of iterations = ",count2)

# OUTPUT 
# Fixed point method

# The solution obtained is  [6.000000047367461, 1.0, -3.999999431590468] ; No. of iterations =  10

# Newton rapson method

# The solution obtained is  [6.0, 1.0000000000000049, -4.000000001428575] ; No. of iterations =  6
