
# Anuj Arya, Roll no. 2311031

import matplotlib.pyplot as plt
import numpy as np

# ------------------- Assignment 00 ------------------------
class Number():
    def __init__(self):
        pass
    # Sum of first 20 odd numbers 
    def sum_of_odd_numbers(self, n):
        p=0
        for i in range(1,2*n):
            if i % 2 != 0:
                print(i, end=' ')
                p += i
        return p
    # factorial of N=8 
    def fact(self, n):
        if n == 0 or n == 1:
            return 1        
        else:
            return n * self.fact(n - 1)
class Series():
    def __init__(self):
        pass
    # Sum of 15 terms of a GP and HPseries for common difference 1.5 and common ratio 0.5 starting from t0 = 1.25.
    def sum_of_GP(self,n,a,r):
        p=[]
        for i in range(n):
            p.append(a * (r ** i))  # ith term of GP
        print(p)
        return sum(p)

    def sum_of_HP(self,n,a,d):
        m=[]
        for i in range(n):
            m.append(1 / (a + i * d))
        print(m)
        return sum(m)

class Matrix_Operation():
    def __init__(self):
        pass
    def read_matrix(self,filename):
        with open(filename,'r') as f:
            matrix =[]
            for line in f:
                # Split the line into numbers and convert them to float
                row = [float(num) for num in line.split()]
                matrix.append(row)
            return matrix

    def matrix_multiply(self,X,Y):
        if len(X[0]) != len(Y):
            raise ValueError("Number of columns in X must be equal to number of rows in Y")
        result = [[0 for _ in range(len(Y[0]))] for _ in range(len(X))]

        for i in range(len(X)):
            for j in range(len(Y[0])):
                for k in range(len(Y)):
                    result[i][j] += X[i][k] * Y[k][j]
        return result

    def dot_product(self,X,Y):
        if len(X) != len(Y):
            raise ValueError("Vectors must be of the same length")
        else:  
            p=0
            for i in range(len(X)):
                p += X[i][0] * Y[i][0]
            return p

class Mycomplex():
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag
    def sum(self, d1,d2):
        return (self.real + d1) + (self.imag + d2) * 1j

    def difference(self, d1,d2):
        return (self.real - d1) + (self.imag - d2) * 1j

    def product(self, d1,d2):
        return (self.real * d1 - self.imag * d2) + (self.real * d2 + self.imag * d1) * 1j

    def modulus(self,d1,d2):
        return (d1 ** 2 + d2 ** 2) ** 0.5

# ---------------------------- Assignment 01.1 ----------------------

class Random():
    def __init__(self):
        pass    
    def pRNG(self,c,n):
        x=[0.1]
        for i in range(n):
            p = c*x[i]*(1-x[i])
            x.append(p)
        return x

    def slicing(self,y1,k):
        # instead of this use slicing [:-k] & [-k:]
        a=[]
        b=[]
        for i in range(len(y1)):
            a.append(y1[i])
            if (i+k)<len(y1):
                b.append(y1[i+k])
        a = a[:-k]
        return a,b

    def LCG(self,seed,a,c,m,n):
        rand_num =[] 
        x=seed
        for _ in range(n):
            x= (a*x+c) % m
            rand_num.append(x)
        return rand_num

    def correlation_test(self,k, L):
        term1 = 0
        for i in range(len(L)):
            if (i+k) > len(L)-1:
                break

            term1 += L[i]*L[i+k]
        term1 = term1/len(L)

        term2 = 0
        for i in range(len(L)):
            term2 += L[i]
        term2 = term2/len(L)
        term2 = term2*term2

        return term1 - term2

class Plots():
    def __init__(self):
        pass

    def plot(self,x, y,title,xlabel, ylabel):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.scatter(x, y, marker='o', color='b')
        plt.grid(True)
        plt.show()
        #plt.savefig(f"{title}.png")

    def line_plot(self,x, y,title,xlabel, ylabel):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(x, y)
        plt.grid(True)
        plt.show()
        #plt.savefig(f"{title}.png")

    def hist(self, data, title, xlabel, ylabel, bins):
        plt.hist(data, bins=bins, color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()

# ------------------------ Assignment 01.2 ----------------------
class Pi_estimation():
    def __init__(self):
        self.rng = Random()

    def points_inside_circle(self,x,y):
        count = 0
        for i in range(len(x)):
            if (x[i]**2 +y[i]**2) <1:
                count +=1
        return count

    def pi_val_cal(self,total_pt):
        x = self.rng.LCG(0.1,1103515245,12345,32768,total_pt)
        y = self.rng.LCG(0.4,1103515245,12345,32768,total_pt)
        for i in range(len(x)):
            x[i] = x[i]/32768
        for i in range(len(y)):
            y[i] = y[i]/32768
        inside_points = self.points_inside_circle(x,y)
        pi_val = 4*(inside_points / total_pt)
        return pi_val

    def avg_pi_cal(self,pi_value):
        pi_value = pi_value[-200:]
        pi_sum = 0
        for i in range(len(pi_value)):
            pi_sum += pi_value[i]
        avg_pi = pi_sum/len(pi_value)
        return avg_pi
        
    def Exp_Rand_num(self,x):
        y=[]
        for i in range(len(x)):
            y.append(-np.log(x[i]))
        return y

# ------------------------- Assignment 02 --------------------------------------

class Gauss_Jordon_Elimination():
    def __init__(self):
        pass

    def agumented_matrix(self,A,b):
        m = [[0 for i in range(len(A)+1)] for j in range(len(A))]
        for i in range(len(A)):
            for j in range(len(A)):
                m[j][i] = A[j][i]
        for i in range(len(A)):
            m[i][len(A)] = b[i]
        return m 

    def GJE(self,A):
        # A - agumented matrix
        n = len(A)
        b = [row[-1] for row in A]
        p= 0
        for i in range(len(A)):
            if A[i][0]> p:
                p = A[i][0]
                m = i
        for j in range(n):
            A[m][j],A[0][j] = A[0][j],A[m][j]
            b[m], b[0] = b[0], b[m]
        for i in range(n):
            diag = A[i][i]
            for j in range(n):
                A[i][j] /= diag
            b[i][0] /= diag
            for k in range(n):
                if k != i:
                    factor = A[k][i]
                    for j in range(n):
                        A[k][j] -= factor * A[i][j]
                    b[k][0] -= factor * b[i][0] 
        for i in range(len(A)):
            A[i][len(A)] = b[i]
        return A    


