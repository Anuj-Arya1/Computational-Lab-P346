# Anuj Arya, Roll no. 2311031

from anuj_library import *

o1 = Pi_estimation()
a = []
for i in range(5000):
    a.append(i+10)
pi_value=[]

for i in range(len(a)):
    pi_value.append(o1.pi_val_cal(a[i]))

p0 = o1.avg_pi_cal(pi_value)
plt.plot(a,pi_value)
plt.show()
print(p0) 