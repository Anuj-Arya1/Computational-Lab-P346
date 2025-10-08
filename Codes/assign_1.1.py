# Anuj Arya, Roll no. 2311031

from anuj_library import *

obj = Random()
plot_obj = Plots()

# Question-1
y1 = obj.pRNG(3.98, 1000)
y2 = obj.pRNG(2.78, 1000)
y3 = obj.pRNG(2.98, 1000)
y4 = obj.pRNG(3.76, 1000)
y5 = obj.pRNG(3.78, 1000)

a,b = obj.slicing(y1,5)
c,d = obj.slicing(y2,5)
e,f = obj.slicing(y3,5)
g,h = obj.slicing(y4,5)
i,j = obj.slicing(y5,5)

plot_obj.plot(a,b,'X_{i} vs X_{i+5}','X_i','X_{i+5}')
plot_obj.plot(c,d,'X_{i} vs X_{i+5}','X_i','X_{i+5}')
plot_obj.plot(e,f,'X_{i} vs X_{i+5}','X_i','X_{i+5}')
plot_obj.plot(g,h,'X_{i} vs X_{i+5}','X_i','X_{i+5}')
plot_obj.plot(i,j,'X_{i} vs X_{i+5}','X_i','X_{i+5}')

print('The correlation test score for first set of paramters is:',obj.correlation_test(5,y1))
print('The correlation test score for second set of paramters is:',obj.correlation_test(5,y2))
print('The correlation test score for third set of paramters is:',obj.correlation_test(5,y3))
print('The correlation test score for fourth set of paramters is:',obj.correlation_test(5,y4))
print('The correlation test score for fifth set of paramters is:',obj.correlation_test(5,y5))


# Question-2
lcg_list = obj.LCG(0.1,1103515245,12345,32768,1000)
a1,b1 = obj.slicing(lcg_list,5)
plot_obj.plot(a1,b1,'LCG : X_{i} vs X_{i+5}','X_i','X_{i+5}')
print('The correlation test score for LCG is:',obj.correlation_test(5,lcg_list))


#OUTPUT
# The correlation test score for first set of paramters is: -0.00838176303550453
# The correlation test score for second set of paramters is: 0.0013580370157126875
# The correlation test score for third set of paramters is: 0.0014377386744393594
# The correlation test score for fourth set of paramters is: 0.008485705008538347
# The correlation test score for fifth set of paramters is: 0.007126465931154791
# The correlation test score for LCG is: 1738712.3268877566