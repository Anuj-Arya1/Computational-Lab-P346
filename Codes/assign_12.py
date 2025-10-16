#Anuj Arya 
# Roll no. = 2311031

from anuj_library import *
import numpy as np

o1 = Integration()

def f(x):
    return x**2/(1+x**4)

def g(x):
    return np.sqrt(1+x**4)
#Question 1
print("Integral Value (and n) of function f using Gaussian is :",o1.Gaussian_quadrature(f,-1,1,0.487495494399361,10**(-9)))
#Question 2
print("Integral Value (and n) of function g using Gaussian is :",o1.Gaussian_quadrature(g,0,1,1.089429413,10**(-9)))
print("Integral Value of function g using Simpson is :",o1.simpson(0,1,24,g))

# OUTPUT
# Integral Value (and n) of function f using Gaussian is : (np.float64(0.4874954942585569), 14)
# Integral Value (and n) of function g using Gaussian is : (np.float64(1.0894294131091893), 8)
# Integral Value of function g using Simpson is : 1.089429413076862







