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

def BVP(x,gh,gl,t,tf,f_x,f_v):
    x1,T1,z1 = o1.RK4_DSHO(x,gh,t,tf,0.1,f_x,f_v)
    x2,T2,z2 = o1.RK4_DSHO(x,gl,t,tf,0.1,f_x,f_v)
    g = gl + (((gh - gl)*(200 - T2[-1]))/(T1[-1] - T2[-1]))
    x3,T3,z3 = o1.RK4_DSHO(x,g,t,tf,0.1,f_x,f_v)
    return x1,T1,z1,x2,T2,z2,x3,T3,z3
# a,b,c,d,e,f,g,h,i = BVP(40,10,20,0,10,f_x,f_v)
# print(h[-1])
# # PLOTS
# plt.plot(a,b,color='red',label='Curve with gh = 10')
# plt.plot(d,e,color='blue',label='Curve with gl = 20')
# plt.plot(g,h,color='black',label='Final Curve')
# plt.xlabel('Time (t)')
# plt.ylabel('Temperature (T)')
# plt.legend()
# plt.show()

# question 2
def ux0(x):
    if abs(1-x)< 10**(-6):
        return 300
    else:
        return 0
# L=X,T = Y
def matrix_vector_mult(A, V):
    result = [0 for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i] += A[i][j]*V[j]
    return result
def heat_eqn(ny,nx,x0,y0,X,Y):
    # X is length of rod
    dx = (X-x0)/(nx-1)
    dy = (Y-y0)/(ny-1)
    a= dy/(dx)**2
    V0 = []
    for i in range(nx):
        V0.append(ux0(x0+i*dx))
    A = [[0 for i in range(nx)] for i in range(ny)]  # A = ny x nx matrix
    for j in range(ny):
        for j in range(nx):
            if  i==j :
                A[j][j] = 1 - 2*a
            if i == j-1 or i == j+1:
                A[j][j-1] = a
                A[j][j+1] = a
    V = []
    for k in range(1,ny):
        V1 = o2.matrix_multiply(A,V0)
        V.append(V1)
    return V,dx

V,dx = heat_eqn(50,20,0,0,10,5)
# PLOTS


