# Anuj Arya 
# Roll no. = 2311031

from anuj_library import *
import math

o1 = Gauss_Jordon_Elimination()



# quesntion 1
def f(x):
    return 3*x + math.sin(x) - math.exp(x)
def f1(x):
    return 3 + math.cos(x) - math.exp(x)

# j,k = o1.bracketing(-1.5,1.5,f)
# print(j,k)

o1.count = 0
x_0,count1 = o1.bisection(-1.5,1.5,f)
print("The root of the given function is :",x_0," and the value of function at this is ",f(x_0),"by bisection method")
print("No. of iterations: ",count1)
o1.count=0
x11,count2 = o1.regula_falsi(-1.5,1.5,f)
print("The root of the given function is ",x11,"and the value of function at this is",f(x11),"by regula method")
print("No. of iterations: ",count2)
o1.count=0
x1,count3 = o1.Newton_Raphson(0,f,f1)   # root 1
print("The 1st root of the given function is ",x1,"and the value of function at this is",f(x1))
print("No. of iterations: ",count3)
x2,count4 = o1.Newton_Raphson(1.2,f,f1)  # root 2
print("The 2nd root of the given function is ",x2,"and the value of function at this is",f(x2),"by regula method")
print("No. of iterations: ",count4)

# Question 2

def h(x):
    return x**2-2*x-3
def g(x):
    return (x**2 - 3)/2

# a,b = o1.bracketing(2,4,h)
# print("Interval got after bracketing",a,b)
o1.count =0
res,cont5 = o1.fixed_point(1,g)
print("One of the root of the given function is ",res,"and the value of function at this is",h(res),"by fixed point method")
print("No. of iterations: ",cont5)


#  OUTPUT
#Question 01

# The root of the given function is : 0.3604220151901245  and the value of function at this is  7.811408615499005e-07 by bisection method
# No. of iterations:  23
# The root of the given function is  0.36042170296032444 and the value of function at this is 0.0 by regula method
# No. of iterations:  24
# The 1st root of the given function is  0.3604217029603242 and the value of function at this is -4.440892098500626e-16
# No. of iterations:  4
# The 2nd root of the given function is  1.890029729252302 and the value of function at this is -1.2461143228392757e-12 by regula method
# No. of iterations:  18

#question 02

# Interval got after bracketing 2 4
# One of the root of the given function is  -1.0 and the value of function at this is 0.0 by fixed point method
# No. of iterations:  2