#Anuj Arya 
# Roll no. = 2311031

from anuj_library import *
import matplotlib.pyplot as plt
import numpy as np

o1 = Fitting_data()
# Question 1

lx = [2 , 3  ,5 , 8  ,12]
ly = [10, 15 ,25, 40 ,60]

print('Fitted curve value at x= 6.7 is ',o1.lagrange_interpolation(lx,ly,6.7))

# OUTPUT
# Fitted curve value at x= 6.7 is  33.5


x = np.linspace(min(lx), max(lx), 5)
plt.scatter(lx,ly,label='Data points',color='blue')
plt.plot(x,[o1.lagrange_interpolation(lx,ly,i) for i in x],label = 'Fitted curve',color='red')
plt.legend()
plt.show()


# Question 2

Lx = [2.5,3.5,5.0,6.0,7.5,10.0,12.5,15.0,17.5,20.5]
Ly = [13.0,11.0,8.5,8.2,7.0,6.2,5.2,4.8,4.6,4.3]

sigma_i = [1 for i in range(len(Lx))]
Lx_log = [np.log(x0) for x0 in Lx]
Ly_log = [np.log(y0) for y0 in Ly]

# power law
a1,b1,r2_1 = o1.linear_reg_params(Lx_log,Ly_log,sigma_i)
print('Param. a = ',np.exp(a1),'Param. b = ', b1)
# Param. a =  21.046352159550004 Param. b =  -0.53740930145056

# exponential
a2,b2,r2_2 = o1.linear_reg_params(Lx,Ly_log,sigma_i)
print('Param. a = ',np.exp(a2),'Param. b = ', np.exp(b2))
# Param. a =  12.21299282456827 Param. b =  0.9432201802447238

plt.scatter(Lx,Ly,label='Data points',color='blue')
plt.plot(Lx,[np.exp(a1)*(x0**b1) for x0 in Lx],label = 'Fitted curve (power law)',color='purple')
plt.plot(Lx,[np.exp(a2 + b2*x0) for x0 in Lx],label = 'Fitted curve (exponential)',color='red')
plt.legend()
plt.show()

print('Pearson r2 for exponential:',r2_2)
print('Pearson r2 for power law:',r2_1,'(it is better!)')

# OUTPUT
# Pearson r2 for exponential: 0.5762426888065756
# Pearson r2 for power law: 0.7750435352872259 (it is better!)