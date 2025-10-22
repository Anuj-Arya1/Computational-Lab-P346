#Anuj Arya 
# Roll no. = 2311031

from anuj_library import *
import matplotlib.pyplot as plt
import math as m
import numpy as np

def f1(x,y):
    return y-x**2
# approach to y(x) = x**2 + 2*x + 2 - 2*e^x  range x(0,2)
def f2(x,y):
    return (x+y)**2
# approach to taninv(x+y) = x+pi/4 range x(0,pi/5)


def forward_euler(L_x,L_y,f,y0,x0,xf,h):
    L_y.append(y0)
    L_x.append(x0)
    x1 = x0 + h
    y1 = y0 + h*f(x0,y0)
    print()
    if abs(xf - x1)<=10**(-6):
        L_x.append(x1),L_y.append(y1)
        return L_x,L_y
    else:
        return forward_euler(L_x,L_y,f,y1,x1,xf,h)

def Predic_Corr_method(L_x,L_y,f,y0,x0,xf,h):
    L_x.append(x0),L_y.append(y0)
    K1 = h*f(x0,y0)
    x1 = x0+h
    K2 = h*f(x1,y0 + K1) 
    y1 = y0 + (K1+K2)/2
    if abs(xf-x1)<= 10**(-6):
        L_x.append(x1),L_y.append(y1)
        return L_x,L_y
    else: 
        return Predic_Corr_method(L_x,L_y,f,y1,x1,xf,h)

# # For f1 function 
# a,b = forward_euler([],[],f1,0,0,2,0.1)
# p,q = Predic_Corr_method([],[],f1,0,0,2,0.1)
# # PLOTS
# x = np.linspace(0, 2, 1000)
# y = x**2 + 2*x + 2 - 2*np.exp(x)
# plt.plot(x, y, color='orange',label = 'Origional Curve')
# plt.plot(a,b,color ='blue',label='Curve plotted with euler')
# plt.plot(p,q,color ='red',linestyle =':',label='Predicted Curve')
# plt.legend()
# plt.grid()
# plt.show()

# For f2 function 
a1,b1 = forward_euler([],[],f2,1,0,0.7,0.1)
p1,q1 = Predic_Corr_method([],[],f2,1,0,0.7,0.1)
# PLOTS
def y2(x):
    return m.tan(x + m.pi/4)-x
xe = []
for i in range(0, 800):
    xe.append(0 + i*0.001)
    if xe[-1] >= math.pi/5:
        break
ye = [y2(i) for i in xe]
plt.plot(xe, ye, color='orange',label = 'Origional Curve')
plt.plot(a1,b1,color ='blue',label='Curve plotted with euler')
plt.plot(p1,q1,color ='red',linestyle =':',label='Predicted Curve')
plt.legend()
plt.grid()
plt.show()



