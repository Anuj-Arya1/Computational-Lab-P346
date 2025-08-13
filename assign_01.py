# Anuj Arya, Roll no. 2311031

from anuj_library import *

obj = Random()

y1 = obj.random_gen(3.98, 1000)
y2 = obj.random_gen(2.78, 1000)
y3 = obj.random_gen(2.98, 1000)
y4 = obj.random_gen(3.76, 1000)
y5 = obj.random_gen(3.78, 1000)
lcg_list = obj.LCG(0.1,1103515245,12345,32768,1000)

a1,b1 = obj.slicing(lcg_list,5)

a,b = obj.slicing(y1,5)
c,d = obj.slicing(y2,5)
e,f = obj.slicing(y3,5)
g,h = obj.slicing(y4,5)
i,j = obj.slicing(y5,5)

obj.plot(a,b)
obj.plot(c,d)
obj.plot(e,f)
obj.plot(g,h)
obj.plot(i,j)
obj.plot(a1,b1)

