
# Anuj Arya 
# Roll no. = 2311031

from anuj_library import *
import math

o1 = Gauss_Jordon_Elimination()
o2 = Matrix_Operation()

# fixed point method

def g(x1,x2,x3):
    return [(37-x2)**(0.5),(x1-5)**(0.5),3-x1-x2]

count= 0
def fixed_pt_multi_val(g,x1,x2,x3):
    global count
    e = 10**(-6)
    x = [x1,x2,x3]
    x_new = g(x1,x2,x3)
    norm_new = math.sqrt(x_new[0]**2 + x_new[1]**2 + x_new[2]**2)
    norm_x = math.sqrt((x_new[0]-x[0])**2 + (x_new[1]-x[1])**2 + (x_new[2]-x[2])**2)
    if norm_x/norm_new <e:
        return x_new,count
    else:
        count+=1
        return fixed_pt_multi_val(g,x_new[0],x_new[1],x_new[2])
    
# res1,count1 = fixed_pt_multi_val(g,6,5,4)
# print("The soltion obtained is ",res1,"; in No. of iterations = ",count1)

# newton-rapson
count1 = 0
def jacobian(x1,x2,x3):
    return [[2*x1,1,0],[1,-2*x2,0],[1,1,1]]

def F(x1,x2,x3):
    return [[x1**2+x2-37], [x1-x2**2-5], [x1+x2+x3-3]]
def newton_rapson_mul(x1,x2,x3,jacobian,F):
    global count1
    e = 10**(-6)
    x = [x1,x2,x3]
    J = jacobian(x1,x2,x3)
    print("and",J)
    f = F(x1,x2,x3)
    inv = o1.inverse_matrix(J)  # inverse done using Gauss jordon elimination
    print(inv)
    list1 =  o2.matrix_multiply(inv,f)
    x_new =[0 for _ in range(3)]
    for i in range(3):
        x_new[i] = x[i]-list1[i][0]

    norm_new = math.sqrt(x_new[0]**2 + x_new[1]**2 + x_new[2]**2)
    norm_x = math.sqrt((x_new[0]-x[0])**2 + (x_new[1]-x[1])**2 + (x_new[2]-x[2])**2)
    if norm_x/norm_new <e:
        return x_new,count1
    else:
        count1 +=1
        return newton_rapson_mul(x_new[0],x_new[1],x_new[2],jacobian,F)
    
res2,count2 = newton_rapson_mul(5,5,5,jacobian,F)
print("The soltion obtained is ",res2,"; in No. of iterations = ",count2)

# OUTPUT 
# Fixed point method

# he soltion obtained is  [6.000000047367461, 1.0, -3.999999431590468] ; in No. of iterations =  10

# Newton rapson method

# The soltion obtained is  [6.0, 1.0000000000000049, -4.000000001428575] ; in No. of iterations =  6
