
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

# Question1

a1,b1 = o1.RK4(0,1,math.pi/5,f1,0.1)
a2,b2 = o1.RK4(0,1,math.pi/5,f1,0.25)
a3,b3 = o1.RK4(0,1,math.pi/5,f1,0.45)

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
def f_x(x,v,t):
    return v

def f_v(x,v,t):
    k = 1.0
    m = 1.0
    mu = 0.15
    w = 1
    return -mu*v - (w**2)*x

t,x,v = o1.RK4_DSHO(1,0,0,40,0.1,f_x,f_v)
E = [0.5*(1*v[i]**2 + 1*x[i]**2) for i in range(len(x))]
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

