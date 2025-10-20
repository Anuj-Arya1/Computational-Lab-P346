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
    y = o3.LCG(0.345,1103515245,12345,32768,total_pt)
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

    x = np.linspace(0, 2, 1000)
    y = np.sqrt(1 - (x**2)/4)
    plt.plot(x, y, color='black')
    plt.scatter(lx_in,ly_in,color ='r',s=1)
    plt.scatter(lx_out,ly_out,color='b',s=1)
    plt.show()
    return Area,percentage


print(ellipse_area(10000))

# (6.28352, -0.05257342529109105 %)

# Question 2

def f(x):
     return (x-5)*math.exp(x) + 5
def f1(x):
     return (x-4)*math.exp(x)

o1.count =0
x,county = o1.Newton_Raphson(6,f,f1)
print("The value of x comes out to be :",x) # value of x
h = 6.626e-34
k= 1.381e-23
c=3e8

print("Wein's constant (b) is :", (h*c)/(k*x)) #b = hc/kx

# Wein's constant (b) is : 0.0028990103307379917


# Question 3

A = o2.read_matrix('DATA/MIDEXAM/ms_A_matrix')

def round0(list,place):
        for i in range(5):
            list[i] = round(list[i],place)
        return list

if o1.determinant(A) != 0:
    I = [[[0] for _ in range(len(A))]for _ in range(len(A))]
    for i in range(len(A)):
        I[i][i] = [1]
    for row in I:
        m = round0(o1.LU_back_frwd(A,row),3)
        I[I.index(row)] = m

    print((o1.transpose_matrix(I))) # inverse

# OUTPUT
# [[-0.708, 2.531, 2.431, 0.967, -3.902], [-0.193, 0.31, 0.279, 0.058, -0.294], 
# [0.022, 0.365, 0.286, 0.051, -0.29], [0.273, -0.13, 0.132, -0.141, 0.449], 
# [0.782, -2.875, -2.679, -0.701, 4.234]]


# Question 4
A0 = o2.read_matrix('DATA/MIDEXAM/msem_gs')
b0 = o2.read_matrix('DATA/MIDEXAM/msem_bvec')

print(o1.Gauss_seidel(A0,b0))

#OUTPUT
# No. of iterations for convergence 12
# [1.4999998297596435, -0.4999999999999992, 1.9999999999999996, -2.499999914864037, 1.0000000000000004, -0.9999999999957907]




          
     