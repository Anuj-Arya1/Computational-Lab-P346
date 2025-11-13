#Anuj Arya 
# Roll no. = 2311031

from anuj_library import *
import matplotlib.pyplot as plt
import math as m

o1 = Differential_equation()
o2 = Matrix_Operation()

# boundary value problem (question 1)

def f_x(x,v,t):
    return v
def f_v(x,v,t): 
    a=0.01
    T_a = 20
    return -a*(T_a-x)


a,b,c,d,e,f,g,h,i = o1.BVP(40,10,20,0,10,f_x,f_v)

# x at which T = 100
for i in range(len(h)-1):
    if h[i]<=100 and h[i+1]>100:
        print('Temperature',h[i],'at position',g[i])
        print('Temperature',h[i+1],'at position',g[i+1])
        break

# print(h[-1])
# PLOTS
plt.plot(a,b,color='red',label='Curve with gh = 10')
plt.plot(d,e,color='blue',label='Curve with gl = 20')
plt.plot(g,h,color='black',label='Final Curve')
plt.xlabel('Time (t)')
plt.ylabel('Temperature (T)')
plt.legend()
plt.show()

# OUTPUT of question 1
# Temperature 98.78432481771769 at position 4.4
# Temperature 100.25272383202632 at position 4.5

# question 2
def Yx0(x):
    if abs(1-x)< 10**(-6):
        return 300
    else:
        return 0
# L=X,T =Y

def heat_eqn(nt,nx,x0,X,t0,T):
    # X is length of rod
    dx = (X-x0)/(nx-1)
    dt = (T-t0)/(nt-1)
    a= dt/(dx)**2
    V0 = []
    T0 = []
    X0 = []
    for i in range(nx):
        X0.append(x0 + i*dx)
    for t in range(nt):
        T0.append(t0 + t*dt)
    for i in range(nx):
        V0.append([Yx0(x0+i*dx)])
    A = [[0 for i in range(nx)] for i in range(nx)]  # A = nx x nx matrix
    res = [V0]
    for i in range(nx):
        for j in range(nx):
            if  i==j :
                A[i][j] = 1 - 2*a
            if i == j-1 or i == j+1:
                A[i][j] = a
    for k in range(0,nt):
        rr = o2.matrix_multiply(A,res[k])
        res.append(rr)
    return res, T0, X0

# PLOTS
result, T0, X0 = heat_eqn(2000,21,0,2,0,2)
time_int = [0,20,50,100,500,1000,1999]
for i in time_int:
    plt.plot(X0,result[i],label=f'Time = {T0[i]:.2f} s')
plt.xlabel('Length of rod (x)')
plt.ylabel('Temperature (T)')
plt.legend()
plt.show()

