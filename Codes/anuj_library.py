
# Anuj Arya, Roll no. 2311031

import matplotlib.pyplot as plt

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

# ---------------------------- Assignment 01 ----------------------

class Random():
    def __init__(self):
        pass    
    def random_gen(self,c,n):
        x=[0.1]
        for i in range(n):
            p = c*x[i]*(1-x[i])
            x.append(p)
        return x

    def plot(self,x, y):
        plt.title('Map Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.scatter(x, y, marker='o', color='b')
        plt.grid(True)
        plt.show()

    def slicing(self,y1,k):
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

# ------------------------ Assignment 02 ----------------------

