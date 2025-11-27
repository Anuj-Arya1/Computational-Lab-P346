
# Anuj Arya
# 2311031

import numpy as np
import matplotlib.pyplot as plt
from anuj_library import *


o1 = Gauss_Jordon_Elimination()
o2 = Matrix_Operation()
o3 = Integration()
o4 = Differential_equation()
o5 = Fitting_data()
o6 = Random()

# Q.1

Par_list = o6.LCG(0.1,1103515245,12345,32768,20000)
Par_list = [i/32768 for i in Par_list]
r_p = 0 # particles on right side
l_p =5000 # particles on left side
count_lp = []
t = []
for i in range(len(Par_list)):
    Par_list[i] = int(Par_list[i]*5000)
    if Par_list[i] < l_p:
        l_p -= 1
        r_p += 1
    else:
        r_p -= 1
        l_p += 1
    count_lp.append(l_p)
    t.append(i)

plt.figure()
plt.plot(t, count_lp, label="Particles available on Left Side")
plt.xlabel("Time step")
plt.ylabel("Count of Particles on Left Side")
plt.legend()
plt.show()

print("Final left-side count:", l_p)

# OUTPUT
# Final left-side count: 2492  (at equillibrium, approx. =2500) 

# Q.2

Aq2 = o2.read_matrix('DATA/Com_end_exam/endexam_Q2_A')
bq2 = o2.read_matrix('DATA/Com_end_exam/endexam_Q2_b')

print(o1.Gauss_seidel(Aq2,bq2))

#OUTPUT
# No. of iterations for convergence 15
# [0.9999997530614102, 0.9999997892247294, 0.9999999100460266, 0.9999998509593769, 0.9999998727858708, 0.9999999457079743]

# Q.3

def f(x):
    return 2.5 - x*np.exp(x)
def f1(x):
    return -np.exp(x)*(x + 1)

o1.count = 0
a,count1 = o1.Newton_Raphson(0,f,f1)
print('Root by Newton Raphson:',a,'No. of iterations:',count1)

#OUTPUT
# Root(it is the maximum displacement) by Newton Raphson: 0.958586356728703 No. of iterations: 7

# Q.4

def g(x):
    return x**2
def g_(x):
    return x**3

denom = o3.simpson(0,2,10,g)
num =   o3.simpson(0,2,10,g_)
center_of_mass = num / denom
print('Center of mass:',round(center_of_mass, 4))

# OUTPUT
# Center of mass: 1.5

# Q.5

def w(t, Inteval):
    y, v = Inteval
    dy_dt = v
    dv_dt = -0.02 * v - 10
    return np.array([dy_dt, dv_dt])
t_val, Y_val = o5.RK4_new(w, 0, [0.0, 10], 10, 0.001)
y_val = Y_val[:, 0]
v_val = Y_val[:, 1]
a=0
m=0
for i in range(len(y_val)):
    if m < y_val[i]:
        m= y_val[i]
        index=i
    else: 
        pass
y_m = Y_val[index]

print(" Maximum height attained", y_m[0])
plt.figure()
plt.plot(y_val, v_val)
plt.xlabel("y ")
plt.ylabel("V")
plt.legend(["Velocity vs height"])
plt.show()

# OUTPUT
# Maximum height attained 4.934317509223537

#Q.6
def Yx0(x):
    return 20*abs(np.sin(np.pi*x))

# PLOTS 
result, T0, X0 = o5.heat_eqn(5001,21,0,2,0,4,Yx0)
time_int = [0,10,20,50,100,200,500,999]
for i in time_int:
    plt.plot(X0,result[i],label=f'Time = {T0[i]:.2f} s')
plt.xlabel('Length of rod (x)')
plt.ylabel('Temperature (T)')
plt.legend()
plt.show()


#Q.7

data = np.loadtxt('DATA/Com_end_exam/esem4fit.txt')
x_data = data[:,0]
y_data = data[:,1]

Coeff = o5.poly_fit(x_data, y_data, 4)
print('Coefficients of the polynomial fit:', Coeff)
# Coefficients of the polynomial fit: [np.float64(0.254629507211548), np.float64(-1.1937592138092277), np.float64(-0.45725541238296813), np.float64(-0.8025653910658186), np.float64(0.013239427477396298)]

def fit_curve(x, Coeff):
    f = Coeff[0] + Coeff[1]*x + Coeff[2]*x**2 + Coeff[3]*x**3 + Coeff[4]*x**4
    return f
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = fit_curve(x_fit, Coeff)
plt.scatter(x_data, y_data, label='Data points', color='blue')
plt.plot(x_fit, y_fit, label='Fitted curve', color='red')
plt.legend()
plt.show()