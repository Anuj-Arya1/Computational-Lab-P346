# Anuj Arya, Roll no. 2311031

from anuj_library import *

o1 = Pi_estimation()
o2 = Plots()
o3 = Random()

# Question-3
a = []
for i in range(5000):
    a.append(i+10)

pi_value=[]
for i in range(len(a)):
    pi_value.append(o1.pi_val_cal(a[i]))

p0 = o1.avg_pi_cal(pi_value)
o2.line_plot(a, pi_value, "Pi Value Estimation", "Number of Points", "Estimated Pi Value")
print(p0) # average pi value = 3.1192010453424746

# Question-4

x1 = o3.LCG(0.4,1103515245,12345,32768,5000)
for i in range(len(x1)):
    x1[i] = x1[i]/32768

y=[]
for i in range(len(x1)):
    y.append(-np.log(x1[i]))

o2.hist(y, "Histogram of Exponential Random Numbers", "Value", "Frequency", bins=40)