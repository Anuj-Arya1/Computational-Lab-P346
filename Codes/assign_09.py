# Anuj Arya
# Roll no. = 2311031
from anuj_library import *
import math
o1 = Gauss_Jordon_Elimination()
# Question 1
coeff_p1 = [1,-1,-7,1,6]
coeff_p2 = [1,0,-5,0,4]
coeff_p3 = [1,-1,-7,1,6]

def Polynm(x, coeff):
    pol = 0
    for i in range(len(coeff)):
        pol += coeff[len(coeff)-(i+1)] * x**i
    return pol

def deriv(coeff):
    coeff_new = []
    maxdeg = len(coeff) - 1
    for j in range(len(coeff)-1):
        coeff_new.append((maxdeg-j) * coeff[j])
    return coeff_new

def deflation(coeff, r):
    list1 = [coeff[0]]
    for i in range(1, len(coeff)):
        p = r * list1[i-1] + coeff[i]
        list1.append(p)
    return list1[:-1]

def laguerre_single_root(b0, coeff):
    e = 1e-6
    n = len(coeff) - 1
    for _ in range(100):  # max iterations
        poly = Polynm(b0, coeff)
        if abs(poly) < e:
            return b0
        G = Polynm(b0, deriv(coeff)) / poly
        H = G**2 - Polynm(b0, deriv(deriv(coeff))) / poly
        m1 = G + math.sqrt((n-1)*(n*H - G**2))
        m2 = G - math.sqrt((n-1)*(n*H - G**2))
        if abs(m1) > abs(m2):
            a = n / m1
        else:
            a = n / m2
        b1 = b0 - a
        if abs(b1 - b0) < e:
            return b1
        b0 = b1
    return b0  

def laguerre_algo(guess,coeff):
    roots = []
    curr_coeff = coeff[:]
    while len(curr_coeff) > 2:
        root = laguerre_single_root(guess, curr_coeff)
        roots.append(root)
        curr_coeff = deflation(curr_coeff, root)
    if len(curr_coeff) == 2:
        roots.append(-curr_coeff[1]/curr_coeff[0])
    return roots



print("Root of polynomial P1 = ",laguerre_algo(2,coeff_p1))
print("Root of polynomial P2 = ",laguerre_algo(2,coeff_p2))
print("Root of polynomial P3 = ",laguerre_algo(2,coeff_p3))

# OUTPUT

# Root of polynomial P1 =  [1.0, 2.9999999999984905, -0.9999999999924531, -2.0000000000060374]
# Root of polynomial P2 =  [2, 1.0000000001004117, -1.0000000003012333, -1.9999999997991784]
# Root of polynomial P3 =  [1.0, 2.9999999999984905, -0.9999999999924531, -2.0000000000060374]

