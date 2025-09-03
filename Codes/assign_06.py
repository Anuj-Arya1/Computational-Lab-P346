
import math

def f(x):
    return math.log(x/2) - math.sin(5*x/2)
def g(x):
    return -x -math.cos(x)

def bracketing(a,b,f):
    step = 0.1
    if f(a)*f(b)<0:
        return a,b
    if f(a)*f(b)>0:
        if abs(f(a))<abs(f(b)):
            #shift a
            a = a- step*(b-a)
            return bracketing(a,b,f)
        elif abs(f(a))>abs(f(b)):
            #shift b
            b = b + step*(b-a)
            return bracketing(a,b,f)

def bisection(a,b):
    def f(x):
        return math.log(x/2) - math.sin(5*x/2)
    e = 10**(-6) #precision
    if abs(b-a)<e:
        return (a+b)/2
    else:
        c=(a+b)/2
        if f(c)*f(a)<0:
            return bisection(a,c)
        if f(c)*f(b)<0:
            return bisection(c,b)

def regula_falsi(a,b):
    def f(x):
        return math.log(x/2) - math.sin(5*x/2)
    e=10**(-6)
    if abs(b-a)<e:
        return a,b
    c = b-((b-a)*f(b))/(f(b)-f(a))
    if f(a)*f(c)<0:
        return regula_falsi(a,c)
    if f(b)*f(c)<0:
        return regula_falsi(c,b)


# def bisection2(a,b):  # only for checking question 2
#     def f(x):
#         return -x - math.cos(x)
#     e = 10**(-6) #precision
#     if abs(b-a)<e:
#         return (a+b)/2
#     else:
#         c=(a+b)/2
#         if f(c)*f(a)<0:
#             return bisection2(a,c)
#         if f(c)*f(b)<0:
#             return bisection2(c,b)
# Question 1

x_0 = bisection(1.5,3)
print("The root of the given function is :",x_0,", and the value of function at this is ",f(x_0),"by bisection method")

x11,y11 = regula_falsi(1.5,3)
print("The root of the given function is ",x11,"and the value of function at this is",f(x11),"by regula method")

# # question 2
x2,y2 = bracketing(2,4,g)
print("Interval got after bracketing is ",(x2,y2))
print("The value of g(x2) and g(y2) is ",g(x2),"and",g(y2),"respectively")
# print(bisection2(x2,y2))

# OUTPUT 
# QUESTION-1
# The root of the given function is : 2.6231406927108765 , and value of f(x_0) is  -7.235050047960101e-07 by bisection method
# The root of the given function is  2.6231403354363083 and the value of function at this is 2.7755575615628914e-16 by regula method

# QUESTION-2
# Interval got after bracketing is  (-1.1874849202000002, 4)
# The value of g(x2) and g(y2) is  0.8134913038547624 and -3.346356379136388 respectively


















