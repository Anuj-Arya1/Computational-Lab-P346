#Anuj Arya 
# Roll no. = 2311031

from anuj_library import *
import matplotlib.pyplot as plt
import math as m
import numpy as np


# Question 1

lx = [2 , 3  ,5 , 8  ,12]
ly = [10, 15 ,25, 40 ,60]

def lagrange_interpolation(lx, ly, x):
    n = len(lx)
    P = 0
    for i in range(n):
        c=1
        for k in range(n):
            if k != i:
                c *= ((x-lx[k])/(lx[i]-lx[k]))
        P+=c*ly[i]
    return P

print('Fitted curve value at x= 6.7 is ',lagrange_interpolation(lx,ly,6.7))
# OUTPUT
# Fitted curve value at x= 6.7 is  33.5


x = np.linspace(min(lx), max(lx), 5)
plt.scatter(lx,ly,label='Data points',color='blue')
plt.plot(x,[lagrange_interpolation(lx,ly,i) for i in x],label = 'Fitted curve',color='red')
plt.legend()
plt.show()


# Question 2

Lx = [2.5,3.5,5.0,6.0,7.5,10.0,12.5,15.0,17.5,20.5]
Ly = [13.0,11.0,8.5,8.2,7.0,6.2,5.2,4.8,4.6,4.3]
sigma_i = [1 for i in range(len(Lx))]


def f(x,a,b):
    y_log = np.log(a) + b*np.log(x)
    return y_log 

Lx_log = [np.log(x0) for x0 in Lx]
Ly_log = [np.log(y0) for y0 in Ly]

def linear_reg_params(Lx,Ly,sigma_i):
    n = len(Lx)
    S,Sx,Sy,Sxx,Sxy,Syy = 0,0,0,0,0,0
    for i in range(n):
        S += 1/sigma_i[i]
        Sx += Lx[i]/sigma_i[i]
        Sy += Ly[i]/sigma_i[i]
        Sxx += (Lx[i]*Lx[i])/sigma_i[i]
        Sxy += (Lx[i]*Ly[i])/sigma_i[i]
        Syy += (Ly[i]*Ly[i])/sigma_i[i]
    delta = S*Sxx - Sx*Sx
    r2 = (Sxy)**2/(Sxx*Syy)
    a = (Sxx*Sy - Sx*Sxy)/delta
    b = (S*Sxy - Sx*Sy)/delta
    return a,b,r2

Sy,S,Sx,Sxy,Sxx,Syy = 0,0,0,0,0,0
# power law
a1,b1,r2_1 = linear_reg_params(Lx_log,Ly_log,sigma_i)

# exponential
a2,b2,r2_2 = linear_reg_params(Lx,Ly_log,sigma_i)


print('Pearson r2 for exponential:',r2_2)
print('Pearson r2 for power law:',r2_1,'(it is better!)')

# OUTPUT
# Pearson r2 for exponential: 0.5762426888065756
# Pearson r2 for power law: 0.7750435352872259 (it is better!)