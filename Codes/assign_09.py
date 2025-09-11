# Anuj Arya
# Roll no. = 2311031
from anuj_library import *
import math
o1 = Gauss_Jordon_Elimination()
# Question 1
coeff_p1 = [1,-1,-7,1,6]
coeff_p2 = [1,0,-5,0,4]
coeff_p3 = [2,0,-19.5,0.5,13.5,-4.5]  # earlier mistake coeff_p3 = [2,0,-19.5,0.5,13.5,4.5]

print("Root of polynomial P1 = ",o1.laguerre_algo(2,coeff_p1))
print("Root of polynomial P2 = ",o1.laguerre_algo(2,coeff_p2))
print("Root of polynomial P3 = ",o1.laguerre_algo(2,coeff_p3))

# OUTPUT

# Root of polynomial P1 =  [1.0, 2.9999999999984905, -0.9999999999924531, -2.000000000006038]
# Root of polynomial P2 =  [2, 1.0000000001004117, -1.0000000003012333, -1.9999999997991784]
# Root of polynomial P3 =  [0.5001154734921126, 0.49988451914195825, 2.9999999987592605, -0.9999999900548948, -3.000000001338437]
