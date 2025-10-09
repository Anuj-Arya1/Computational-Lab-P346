
import math
from anuj_library import *
o1 = Gauss_Jordon_Elimination()

def f(x):
    return 1/x
def g(x):
    return x*math.cos(x)
def h(x):
    return x*math.atan(x)
N = [4,8,15,20]

def res_func(a,b,N,f,method):
    List_f = []
    for i in range(len(N)):
        List_f.append(method(a,b,N[i],f))
    return List_f

print("N value :","                                    ",'4',"                 ",'8',"               ",'15',"                 ",'20','\n')
print("For function f (by method midpoint):    ",res_func(1,2,N,f,o1.midpoint_integral))
print("For function g (by method midpoint):    ",res_func(0,math.pi/2,N,g,o1.midpoint_integral))  # pi/2 - 1 = 0.57079632 
print("For function h (by method midpoint):    ",res_func(0,1,N,h,o1.midpoint_integral),'\n')

print("For function f (by method trapezoidal): ",res_func(1,2,N,f,o1.trapez_integral))
print("For function g (by method trapezoidal): ",res_func(0,math.pi/2,N,g,o1.trapez_integral))  # pi/2 - 1 = 0.57079632 
print("For function h (by method trapezoidal): ",res_func(0,1,N,h,o1.trapez_integral))


# # --------------------- OUTPUT ---------------------------------
# N value :                                      4                   8                 15                   20 

# For function f (by method midpoint):     [0.6912198912198912, 0.6926605540432034, 0.6930084263712958, 0.6930690982255869]
# For function g (by method midpoint):     [0.5874479167573121, 0.5749342733821311, 0.5719716590967575, 0.5714572867152204]
# For function h (by method midpoint):     [0.2820460493571144, 0.2845610193056679, 0.28516010270349235, 0.28526426016144524] 

# For function f (by method trapezoidal):  [0.6970238095238095, 0.6941218503718504, 0.6934248043580644, 0.6933033817926942]
# For function g (by method trapezoidal):  [0.5376071275673586, 0.5625275221623353, 0.5684462350385162, 0.569474588169518]
# For function h (by method trapezoidal):  [0.2920983458939516, 0.28707219762553304, 0.2858742642174127, 0.285665963360493]