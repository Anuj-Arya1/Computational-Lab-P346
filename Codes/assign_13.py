#Anuj Arya 
# Roll no. = 2311031

from anuj_library import *
import matplotlib.pyplot as plt
import math as m
import numpy as np

o1 = Differential_equation()

def f1(x,y):
    return y-x**2
def y1(x):
    return x**2 + 2*x + 2 - 2*m.exp(x)
def f2(x,y):
    return (x+y)**2
def y2(x):
    return m.tan(x + m.pi/4)-x

# For f1 function 
a,b = o1.forward_euler([],[],f1,0,0,2,0.1)
p,q = o1.Predic_Corr_method([],[],f1,0,0,2,0.1)
# PLOTS
x = np.linspace(0, 2, 100)
plt.plot(x, [y1(i) for i in x], color='orange',label = 'Origional Curve')
plt.plot(a,b,color ='blue',label='Curve plotted with euler')
plt.plot(p,q,color ='red',linestyle =':',label='Predicted Curve')
plt.legend()
plt.grid()
plt.show()

# For f2 function 
a1,b1 = o1.forward_euler([],[],f2,1,0,m.pi/5,0.1)
p1,q1 = o1.Predic_Corr_method([],[],f2,1,0,m.pi/5,0.1)
# PLOTS
xe = np.linspace(0, m.pi/5, 100)
plt.plot(xe,[y2(i) for i in xe], color='orange',label = 'Origional Curve')
plt.plot(a1,b1,color ='blue',label='Curve plotted with euler')
plt.plot(p1,q1,color ='red',linestyle =':',label='Predicted Curve')
plt.legend()
plt.grid()
plt.show()



