
import math
from anuj_library import *
o1 = Integration()

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
        h = (b-a)/N[i]
        List_f.append(method(a,b,h,f))
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

# For function f (by method midpoint):     [0.6912198912198912, 0.6926605540432034, 0.6930084263712957, 0.6930690982255869]
# For function g (by method midpoint):     [0.5874479167573121, 0.574934273382131, 0.5719716590967573, 0.5714572867152207]
# For function h (by method midpoint):     [0.28204604935711447, 0.2845610193056679, 0.2851601027034923, 0.2852642601614453] 

# For function f (by method trapezoidal):  [0.6970238095238095, 0.6941218503718504, 0.6934248043580645, 0.6933033817926939]
# For function g (by method trapezoidal):  [0.5376071275673587, 0.5625275221623354, 0.5684462350385163, 0.5694745881695181]
# For function h (by method trapezoidal):  [0.29209834589395156, 0.28707219762553304, 0.2858742642174127, 0.28566596336049305]