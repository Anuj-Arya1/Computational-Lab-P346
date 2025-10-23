
#Anuj Arya 
# Roll no. = 2311031

from anuj_library import *
import matplotlib.pyplot as plt
import math 

o1 = Differential_equation()

def f1(x,y):
    return (x+y)**2
def y1(x):
    return math.tan(x + math.pi/4)-x

def RK4(L_x,L_y,x0,y0,xf,f,h):
    L_x.append(x0),L_y.append(y0)
    x1 = x0+h
    k1 = h*f(x0,y0)
    k2 = h*f(x0+h/2,y0+k1/2)
    k3 = h*f(x0+h/2,y0+k2/2)
    k4 = h*f(x1,y0+k3)
    y1 = y0+(k1+2*k2+2*k3+k4)/6
    if x1>=xf-10**(-6)*h:
        L_x.append(x1),L_y.append(y1)
        return L_x,L_y
    else:
        return RK4(L_x,L_y,x1,y1,xf,f,h)

# x.. + mu x. +w^2 x = 0,  x = 1,v=0,k=1,m=1,mu=0.15, w^2 =k/m  

def RK4_DSHO(L_X,L_t,L_v,L_E,x,v,t,tf,h,k,m,mu):
    E0 = 0.5*(m*v**2 + k*x**2)
    L_t.append(t),L_v.append(v),L_E.append(E0),L_X.append(x)
    w2 = k/m
    k1x = h*v
    k1v = -h*(mu*v + (w2)*x)
    k2x = h*(v+k1v/2)
    k2v = -h*(mu*k1v + (w2)*k2x)
    k3x = h*(v+k2v/2)
    k3v = -h*(mu*k2v + (w2)*k3x)
    k4x = h*(v+k3v/2)
    k4v = -h*(mu*k3v + (w2)*k4x)
    x1 = x + (k1x + 2*k2x + 2*k3x + k4x)/6
    v1 = v + (k1v + 2*k2v + 2*k3v + k4v)/6
    t1 = t + h
    if t>= tf - 10**(-6):
        return L_t,L_X,L_v,L_E
    else:
        return RK4_DSHO(L_X,L_t,L_v,L_E,x1,v1,t1,tf,h,k,m,mu)

# Question1

a1,b1 = RK4([],[],0,1,math.pi/5,f1,0.1)
a2,b2 = RK4([],[],0,1,math.pi/5,f1,0.25)
a3,b3 = RK4([],[],0,1,math.pi/5,f1,0.45)

# PLOTS
x = np.linspace(0, math.pi/5, 10)
plt.plot(x,[y1(i) for i in x], color='red',label = 'Origional Curve')
plt.plot(a1,b1,color ='blue',  linestyle =':',label='RK4 Curve with h = 0.1')
plt.plot(a2,b2,color ='black', linestyle =':',label='RK4 Curve with h = 0.25')
plt.legend()
plt.show()
plt.plot(x,[y1(i) for i in x], color='red',label = 'Origional Curve')
plt.plot(a3,b3,color ='orange',linestyle =':',label='RK4 Curve with h = 0.45')
plt.legend()
plt.grid()
plt.show()

# Question 2

t,x,v,E = RK4_DSHO([],[],[],[],1,0,0,40,0.1,1,1,0.15)

#PLOTS
plt.scatter(t,x,s=2,color = 'red',marker = 'o')
plt.plot(t,x,label='x vs t curve',color= 'black')
plt.legend()
plt.show()
plt.scatter(t,v,s=2,color = 'red',marker = 'o')
plt.plot(t,v,label='v vs t curve',color= 'black')
plt.legend()
plt.show()
plt.scatter(t,E,s=2,color = 'red',marker = 'o')
plt.plot(t,E,label='E vs t curve',color= 'black')
plt.legend()
plt.show()
plt.plot(x,v,label='v vs x curve',color= 'blue')
plt.legend()
plt.show()

