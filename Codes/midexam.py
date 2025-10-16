# Anuj Arya 
# Roll no. = 2311031


from anuj_library import *
import matplotlib.pyplot as plt

o1 = Gauss_Jordon_Elimination() 
o2 = Matrix_Operation()
o3 = Random()
o4 = Plots()

# Question 1

def points_inside_ellipse(x,y):
        count = 0
        lx_in =[]
        ly_in =[]
        for i in range(len(x)):
            if ((x[i]**2)/4 +y[i]**2) <1:
                count +=1
                lx_in.append(x[i])
                ly_in.append(y[i])
        return count,ly_in,lx_in

def ellipse_area(total_pt):
    x = o3.LCG(0.1,1103515245,12345,32768,total_pt)
    y = o3.LCG(0.4,1103515245,12345,32768,total_pt)
    for i in range(len(x)):
        x[i] = x[i]/16384
    for i in range(len(y)):
        y[i] = y[i]/32768
    inside_points,ly_in,lx_in = points_inside_ellipse(x,y)
    Area = 4*2*(inside_points / total_pt) # 2 is area of rectangle
    percentage = ((2*math.pi-Area)/2*math.pi)*100
    lx_out =[]
    ly_out =[]
    for i in range(len(x)):
        if ((x[i]**2)/4 +y[i]**2) >= 1:
            lx_out.append(x[i])
            ly_out.append(y[i])
    plt.scatter(lx_in,ly_in,color ='r')
    plt.scatter(lx_out,ly_out,color='b')
    plt.show()
    return Area,percentage


# print(ellipse_area(100000))

# (6.28352, -0.05257342529109105 %)

# Question 2

def f(x):
     return (x-5)*math.exp(x) + 5
def f1(x):
     return (x-4)*math.exp(x)

o1.count =0
x,county = o1.Newton_Raphson(6,f,f1)
# print(x) # value of x
h = 6.626e-34
k= 1.381e-23
c=3e8

# print("Wein's constant (b) is :", (h*c)/(k*x)) #b = hc/kx

# Wein's constant (b) is : 0.0028990103307379917


# Question 3

A = o2.read_matrix('DATA/MIDEXAM/ms_A_matrix')
B = o2.read_matrix('DATA/MIDEXAM/ms_A_matrix')
C = o2.read_matrix('DATA/MIDEXAM/ms_A_matrix')
D = o2.read_matrix('DATA/MIDEXAM/ms_A_matrix')
E = o2.read_matrix('DATA/MIDEXAM/ms_A_matrix')

n=len(A)
det = o1.determinant(A) # det non-zero
def round0(list,place):
    for i in range(5):
        list[i] = round(list[i],place)
    return list
I = [] 
row1 = o1.LU_back_frwd(A,[[1],[0],[0],[0],[0]])
I.append(round0(row1,3))
row2 = o1.LU_back_frwd(B,[[0],[1],[0],[0],[0]])
I.append(round0(row2,3))
row3 = o1.LU_back_frwd(C,[[0],[0],[1],[0],[0]])
I.append(round0(row3,3))
row4 = o1.LU_back_frwd(D,[[0],[0],[0],[1],[0]])
I.append(round0(row4,3))
row5 = o1.LU_back_frwd(E,[[0],[0],[0],[0],[1]])
I.append(round0(row5,3))
print((I)) # inverse 

# OUTPUT
# [[0.333, -0.0, 0.0, 0.0, 0.0], [0.455, 0.24, 0.373, -0.03, -2.591], [0.437, 0.212, 0.294, 0.227, -2.406], [0.174, 0.031, 0.054, -0.103, -0.593], [-0.701, -0.186, -0.302, 0.296, 3.796]]



# Question 4
A0 = o2.read_matrix('DATA/MIDEXAM/msem_gs')
b0 = o2.read_matrix('DATA/MIDEXAM/msem_bvec')

# print(o1.Gauss_seidel(A0,b0))

#OUTPUT
# No. of iterations for convergence 12
# [1.4999998297596435, -0.4999999999999992, 1.9999999999999996, -2.499999914864037, 1.0000000000000004, -0.9999999999957907]




          
     