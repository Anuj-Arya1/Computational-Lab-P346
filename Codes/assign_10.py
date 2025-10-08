
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
    List_f.append(method(a,b,N[0],f))
    List_f.append(method(a,b,N[1],f))
    List_f.append(method(a,b,N[2],f))
    List_f.append(method(a,b,N[3],f))
    return List_f

print("N value :","                                    ",'4',"                 ",'8',"               ",'15',"                 ",'20','\n')
print("For function f (by method midpoint):    ",res_func(1,2,N,f,o1.midpoint_integral))
print("For function g (by method midpoint):    ",res_func(0,math.pi/2,N,g,o1.midpoint_integral))  # pi/2 - 1 = 0.57079632 
print("For function h (by method midpoint):    ",res_func(0,1,N,h,o1.midpoint_integral),'\n')

print("For function f (by method trapezoidal): ",res_func(1,2,N,f,o1.trapez_integral))
print("For function g (by method trapezoidal): ",res_func(0,math.pi/2,N,g,o1.trapez_integral))  # pi/2 - 1 = 0.57079632 
print("For function h (by method trapezoidal): ",res_func(0,1,N,h,o1.trapez_integral))


# --------------------- OUTPUT ---------------------------------
# N value :                                      4                   8                 15                   20 

# For function f (by method midpoint):     [0.6926605540432034, 0.6931452732367775, 0.6931471804435301, 0.6931471805598317]
# For function g (by method midpoint):     [0.574934273382131, 0.5708124584762717, 0.5707963277794886, 0.5707963267958581]
# For function h (by method midpoint):     [0.28456101930566796, 0.2853948944563483, 0.28539816319792827, 0.2853981633972535] 

# For function f (by method trapezoidal):  [0.6941218503718504, 0.693150995228108, 0.6931471807927759, 0.6931471805601727]
# For function g (by method trapezoidal):  [0.5625275221623354, 0.5707640635401314, 0.5707963248257126, 0.5707963267929737]
# For function h (by method trapezoidal):  [0.2870721976255331, 0.28540470127576784, 0.2853981637964884, 0.285398163397838]
