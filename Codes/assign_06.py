# Anuj Arya 
# Roll no. = 2311031

from anuj_library import *
import math

o1 = Gauss_Jordon_Elimination()

# Question 1
def f(x):
    return math.log(x/2) - math.sin(5*x/2)
o1.count = 0
x_0,count1 = o1.bisection(1.5,3,f)
print("The root of the given function is :",x_0," and the value of function at this is ",f(x_0),"by bisection method")
print("No. of iterations: ",count1)
o1.count=0
x11,y11,count2 = o1.regula_falsi(1.5,3,f)
print("The root of the given function is ",x11,"and the value of function at this is",f(x11),"by regula method")
print("No. of iterations: ",count2)

# # question 2
def g(x):
    return -x -math.cos(x)
x2,y2 = o1.bracketing(2,4,g)
print("Interval got after bracketing is ",(x2,y2))
print("The value of g(x2) and g(y2) is ",g(x2),"and",g(y2),"respectively")
# print(o1.bisection(x2,y2,g)) # just for check

# OUTPUT 
# QUESTION-1
# The root of the given function is : 2.6231406927108765 , and value of f(x_0) is  -7.235050047960101e-07 by bisection method
# No. of iterations:  22
# The root of the given function is  2.6231403354363083 and the value of function at this is 2.7755575615628914e-16 by regula method
# No. of iterations:  10


# QUESTION-2
# Interval got after bracketing is  (-1.1874849202000002, 4)
# The value of g(x2) and g(y2) is  0.8134913038547624 and -3.346356379136388 respectively


















