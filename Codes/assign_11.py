
import math
from anuj_library import *
o1 = Gauss_Jordon_Elimination()

def f(x):
    return 1/x
def g(x):
    return x*math.cos(x)
def h(x):
    return (math.sin(x))**2


# MID POINT 
# N = 289 for function f and N = 610 for function g
print("For function f (by method midpoint):    ",o1.midpoint_integral(1,2,289,f))
print("For function g (by method midpoint):    ",o1.midpoint_integral(0,math.pi/2,610,g)) 

# SYMPSON 
# N = 20 for function f and N = 22 for function g
print("For function f (by method sympson):    ",o1.simpson(1,2,20,f))
print("For function g (by method sympson):    ",o1.simpson(0,math.pi/2,22,g))

# MONTE CARLO
print('Integration value for funtion h using Monte Carlo: ',o1.Monte_carlo(-1,1,1000,h))



# OUTPUT

# For function f (by method midpoint):     0.6931468064035272
# For function g (by method midpoint):     0.5707970370864707

# For function f (by method sympson):     0.6931473746651162
# For function g (by method sympson):     0.570796987316687

# Integration value for funtion h using Monte Carlo:  0.5454643172257817

