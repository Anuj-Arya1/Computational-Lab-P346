
# Anuj Arya, Roll no. 2311031

import matplotlib.pyplot as plt
import numpy as np
import math

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

    def correlation_test(self,k,x):
        N = len(x)
        t1 =0
        t2 =0
        for i in range(N):
            if (i+k) >= N:
                break
            t1 += x[i]*x[i+k]
        for j in range(k,N):
                t2 += x[j]
        res = (t1/N) - (t2/N)**2
        return res

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
        # input b as [[a],[b],[c],[d]]
        m = [[0 for i in range(len(A)+1)] for j in range(len(A))]
        for i in range(len(A)):
            for j in range(len(A)):
                m[j][i] = A[j][i]
        for i in range(len(A)):
            m[i][len(A)] = b[i][0]
        return m 

    def GJE(self, A):
        n = len(A)
        for i in range(n):
            p = i
            for k in range(i + 1, n):
                if abs(A[k][i]) > abs(A[p][i]):
                    p = k
            if A[p][i] == 0:
                return "Singular Matrix - cannot proceed"
            A[i], A[p] = A[p], A[i]
            
            pivot = A[i][i]
            for j in range(i, len(A[0])):  
                A[i][j] /= pivot

            for k in range(n):
                if k != i:
                    factor = A[k][i]
                    for j in range(i, len(A[0])):
                        A[k][j] -= factor * A[i][j]
        return A
    
    def determinant(self,A):
        n = len(A)
        A = [row[:] for row in A]  # making a copy to avoid modifying the original matrix
        det = 1
        for i in range(n):
            p = i
            for k in range(i + 1, n):
                if abs(A[k][i]) > abs(A[p][i]):
                    p = k
            if A[p][i] == 0:
                return 0  # Singular matrix
            if p != i:
                A[i], A[p] = A[p], A[i]
                det *= -1

            for j in range(i + 1, n):
                factor = A[j][i] / A[i][i]
                for k in range(i, n):
                    A[j][k] -= factor * A[i][k]

        for i in range(n):
            det *= A[i][i]
        return det

    def inverse_matrix(self, A):
        if self.determinant(A) != 0:
            n = len(A)
            # Create an identity matrix
            I = [[0 for i in range(n)] for j in range(n)] 
            for i in range(n):
                I[i][i]=1
            for k in range(n):
                b1 = [[row[k]] for row in I]
                A1 = self.agumented_matrix(A, b1)
                A1 = self.GJE(A1)
                # replace the first column of I with the last column of A1
                for i in range(n):
                    I[i][k] = A1[i][-1]
            return I
        else:
            return ("Singular Matrix - cannot find inverse")

# ------------------------Assign-03 ------------------------------

    def LU_decomposition(self,A):
        # Dolittle method
        A = [row[:] for row in A] # copy origional matrix 
        for j in range(1,len(A)):
            A[j][0] = A[j][0] / A[0][0]
        for j in range(1,len(A)):
            for i in range(1,len(A)):
                sum1 = 0
                sum2 = 0
                for k in range(0,i):
                    sum1 += A[i][k]*A[k][j]
                if i<= j:
                    A[i][j] = A[i][j] - sum1
                for k in range(0,j):
                    sum2 += A[i][k]*A[k][j]
                if i>j:
                    A[i][j] = (A[i][j] - sum2)/ A[j][j]
        return A

    def LU_back_frwd(self,A,b):
        A0 = self.LU_decomposition(A)
        A = [row[:] for row in A0]
        n= len(A)
        # Ly = b
        y=[0 for i in range(n)]
        y[0] = b[0][0]
        x = [0 for _  in range(n)]
        for i in range(1,n):
            sum3 = 0
            for j in range(0,i):
                sum3 += A0[i][j]*y[j]            
            y[i] = b[i][0] - sum3
        # Ux = y
        x[n-1] =  y[n-1]/ A0[n-1][n-1]
        for i in range(n-2,-1,-1):
            sum4 = 0
            for j in range(i+1,n):
                sum4 += A0[i][j]*x[j]
            x[i] = (y[i] - sum4)/A0[i][i]
        return x

# ---------------------- Assign-04 ------------------------------

    def transpose_matrix(self,A):
        B =  [[0 for _ in range(len(A))]for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(A)):
                B[i][j] = A[j][i]
        return B

    def choleski_decomposition(self,A):
        n = len(A)
        L = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i+1):
                sum1 =0
                sum2 =0
                if i == j:
                    for k in range(j):
                        sum1 += L[i][k]**2
                    L[i][j] = (A[i][i] - sum1)**0.5
                else:
                    for k in range(j):
                        sum2 += L[i][k]*L[j][k]
                    L[i][j] = (A[i][j] - sum2) / L[j][j]
        L_t = self.transpose_matrix(L)
        return L,L_t
    
    def cholesky_forwd_back(self,A, b):
        L, L_t = self.choleski_decomposition(A)
        # Ly = b
        y=[0 for i in range(len(A))]
        y[0] = b[0][0]/L[0][0]
        x = [0 for _  in range(len(A))]
        for i in range(1,len(A)):
            sum3 = 0
            for j in range(0,i):
                sum3 += L[i][j]*y[j]
            y[i] = (b[i][0] - sum3)/L[i][i]
        # Ux = y
        x[len(A)-1] =  y[len(A)-1]/ L_t[len(A)-1][len(A)-1]
        for i in range(len(A)-2,-1,-1):
            sum4 = 0
            for j in range(i+1,len(A)):
                sum4 += L_t[i][j]*x[j]
            x[i] = (y[i] - sum4)/L_t[i][i]
        return x

    def Jacobi_iterative(self,A,b):
        e = 10**(-6) # precision
        n = len(A)
        x=[0 for _ in range(2*n)]
        for p in range(1000):
            m = 0
            for i in range(n):
                sum1=0
                for j in range(n):
                    if i!=j:
                        sum1 += A[i][j]*x[j]
                x[i+n] = (b[i][0] - sum1)/ A[i][i]
            for i in range(n):
                m+=(x[i+n] - x[i])**2
            if math.sqrt(m)<e:
                print("No. of iterations for convergence",p)
                break
            for i in range(n):
                x[i],x[i+n] = x[i+n],x[i]
        return x[n:]
    

# ------------------------------------Assign_05-----------------------------------  

    def symmetry_check(self,A):
        n= len(A)
        for i in range(n):
            for j in range(n):
                if A[i][j]== A[j][i]:
                    continue
                else:
                    return "Matrix is not symmetric!"
        return "Its symmetric"

    def Gauss_seidel(self,A,b):
        e = 10**(-6) # precision
        n = len(A)
        x=[0 for _ in range(n)]
        for p in range(1000):
            m = 0
            for i in range(n):
                sum1=0
                sum2=0
                for j in range(i):
                    sum1 += A[i][j]*x[j]
                for j in range(i+1,n):
                    sum2 += A[i][j]*x[j]
                m += (x[i]-(1/A[i][i])*(b[i][0] - sum1 - sum2))**2
                x[i] = (1/A[i][i])*(b[i][0] - sum1 - sum2)           
            if math.sqrt(m)<e:
                print("No. of iterations for convergence",p)
                break
        return x
    
# ------------------ Assign_06 ------------------------------
    def bracketing(self,a,b,f):
        step = 0.1
        if f(a)*f(b)<0:
            return a,b
        if f(a)*f(b)>0:
            if abs(f(a))<abs(f(b)):
                #shift a
                a = a- step*(b-a)
                return self.bracketing(a,b,f)
            elif abs(f(a))>abs(f(b)):
                #shift b
                b = b + step*(b-a)
                return self.bracketing(a,b,f)

    def bisection(self,a,b,f):
        e = 10**(-6) #precision
        self.count += 1
        if abs(b-a)<e:
            return (a+b)/2, (self.count-1)
        else:
            c=(a+b)/2
            if abs(f(c))<e:
                return c, (self.count-1)
            if f(c)*f(a)<0:
                return self.bisection(a,c,f)
            if f(c)*f(b)<0:
                return self.bisection(c,b,f)
            
    def regula_falsi(self,a,b,f):
        e=10**(-6) # precision
        self.count += 1
        if abs(b-a)<e:
            return a,b,(self.count-1)
        c = b-((b-a)*f(b))/(f(b)-f(a))
        if f(a)*f(c)<0:
            return self.regula_falsi(a,c,f)
        elif f(b)*f(c)<0:
            return self.regula_falsi(c,b,f)
        else:
            return c,(self.count-1)
        

# ------------------ Assign-07 ----------------------

    def Newton_Raphson(self,x0,f,f1):
        e= 10**(-6) # precision
        self.count += 1
        x = x0 - f(x0)/f1(x0) 
        if abs(x-x0)<e:
            return x,(self.count-1)
        else:
            return self.Newton_Raphson(x,f,f1)
        
    def fixed_point(self,x0,g):
        self.count+=1
        e= 10**(-6)
        x = g(x0)
        if abs(x-x0)<e:
            return x,(self.count-1)
        else:
            return self.fixed_point(x,g)

# ------------- Assign - 08 ----------------------------------
    
    def fixed_pt_multi_val(self,g,x):
        e = 10**(-6)
        x_new = g(x[0],x[1],x[2])

        norm_new = math.sqrt(x_new[0]**2 + x_new[1]**2 + x_new[2]**2) 
        norm_x = math.sqrt((x_new[0]-x[0])**2 + (x_new[1]-x[1])**2 + (x_new[2]-x[2])**2)
        if norm_x/norm_new <e:
            return x_new,self.count
        else:
            self.count+=1
            return self.fixed_pt_multi_val(g,x_new)
    
    def newton_rapson_mul(self,x,jacobian,F):
        matrix_class = Matrix_Operation()
        gauss = Gauss_Jordon_Elimination()
        e = 10**(-6)
        J = jacobian(x[0],x[1],x[2])
        f = F(x[0],x[1],x[2])
        inv = gauss.inverse_matrix(J)  # inverse done using Gauss jordon elimination
        list1 =  matrix_class.matrix_multiply(inv,f)
        x_new =[0 for _ in range(3)]
        for i in range(3):
            x_new[i] = x[i]-list1[i][0]

        norm_new = math.sqrt(x_new[0]**2 + x_new[1]**2 + x_new[2]**2)
        norm_x = math.sqrt((x_new[0]-x[0])**2 + (x_new[1]-x[1])**2 + (x_new[2]-x[2])**2)
        if norm_x/norm_new <e:
            return x_new,self.count
        else:
            self.count +=1
            return self.newton_rapson_mul(x_new,jacobian,F)

# ------------------- Assign - 09 ----------------------------------
    def Polynm(self,x, coeff):
        pol = 0
        for i in range(len(coeff)):
            pol += coeff[len(coeff)-(i+1)] * x**i
        return pol

    def deriv(self,coeff):
        coeff_new = []
        maxdeg = len(coeff) - 1
        for j in range(len(coeff)-1):
            coeff_new.append((maxdeg-j) * coeff[j])
        return coeff_new

    def deflation(self, coeff, r):
        list1 = [coeff[0]]
        for i in range(1, len(coeff)):
            p = r * list1[i-1] + coeff[i]
            list1.append(p)
        return list1[:-1]

    def laguerre_single_root(self,b0, coeff):
        e = 1e-6
        n = len(coeff) - 1
        poly = self.Polynm(b0, coeff)
        if abs(poly) < e:
            return b0
        G = self.Polynm(b0, self.deriv(coeff)) / poly
        H = G**2 - self.Polynm(b0, self.deriv(self.deriv(coeff))) / poly
        m1 = G + math.sqrt((n-1)*(n*H - G**2))
        m2 = G - math.sqrt((n-1)*(n*H - G**2))
        if abs(m1) > abs(m2):
            a = n / m1
        else:
            a = n / m2
        b1 = b0 - a
        if abs(b1 - b0) < e:
            return b1
        else:
            return self.laguerre_single_root(b1,coeff)

    def laguerre_algo(self,guess,coeff):
        res = []
        for _ in range(len(coeff)-1):
            root = self.laguerre_single_root(guess, coeff)
            res.append(root)
            coeff = self.deflation(coeff, root)
        return res

    # ------------------- Assign - 10 ----------------------------------
class Integration():
    # MIDPOINT METHOD OF INTEGRATION
    def midpoint_integral(self,a,b,h,f):
        #b>a
        if abs(b-a-h) <= 10**(-6) :
            return h*f(a+h/2.0)
        else:
            return h*f(a+h/2.0)+self.midpoint_integral(a+h,b,h,f)

    # TRAPEZOIDAL METHOD
    def trapez_integral(self,a,b,h,f):
        if abs(b-a-h) <= 10**(-6):
            return (h/2.0)*(f(a)+f(b))
        else: 
            return (h/2.0)*(f(a)+f(a+h))+self.trapez_integral(a+h,b,h,f)
    
# -----------------Assignment- 11 ----------------------------

    # SIMPSON'S 1/3 RULE
    def simpson(self,a,b,n,f):
        h = (b-a)/n
        sum = 0
        for i in range(1,n):
            xi = a + i*h
            if i%2 == 1:
                sum += 4*f(xi)
            elif i%2 ==0:
                sum += 2*f(xi)
        sum += f(a) + f(b)
        return (h/3)*sum 
    
    # MONTE CARLO INTEGRATION
    def Monte_carlo(self,a,b,n,f):
        o2 = Random()
        o3 = Plots()
        fn = []
        xa = []
        for i in range(1,n):
            X = o2.LCG(0.4,1103515245,12345,32768,i*100)
            xa.append(i*100)
            sum=0
            for i in range(len(X)):
                X[i] = a+(b-a)*(X[i]/32768)
                sum += f(X[i])
            F_n = (1/(len(X)))*(b-a)*sum
            fn.append(F_n)
        o3.line_plot(xa,fn,"F_N vs N","N","F_N")
        return fn[len(fn)-1]

# --------------------------Assign -12 -----------------------------------

    def Gaussian_quadrature(self,f,a,b,true_val,tor):
        for n in range(1,1000):
            intg = 0 
            roots, weights = np.polynomial.legendre.leggauss(n)
            new_roots = 0.5 *(b-a)*roots + 0.5*(b+a)
            new_weights = 0.5*(b-a)*weights
            for i in range(n):
                intg += new_weights[i]*f(new_roots[i])
            if abs(true_val - intg) <tor:
                return intg , n


# -----------------Assignment- 13 ----------------------------

class Differential_equation():
    def forward_euler(self,L_x,L_y,f,y0,x0,xf,h):
        L_y.append(y0)
        L_x.append(x0)
        x1 = x0 + h
        y1 = y0 + h*f(x0,y0)
        if x1>=xf - 10**(-6):
            L_x.append(x1),L_y.append(y1)
            return L_x,L_y
        else:
            return self.forward_euler(L_x,L_y,f,y1,x1,xf,h)

    def Predic_Corr_method(self,L_x,L_y,f,y0,x0,xf,h):
        L_x.append(x0),L_y.append(y0)
        K1 = h*f(x0,y0)
        x1 = x0+h
        K2 = h*f(x1,y0 + K1) 
        y1 = y0 + (K1+K2)/2
        if x1>=xf - 10**(-6):
            L_x.append(x1),L_y.append(y1)
            return L_x,L_y
        else: 
            return self.Predic_Corr_method(L_x,L_y,f,y1,x1,xf,h)

# -------------- Assign 14 -------------------------

    def RK4(self,x0,y0,xf,f,h):
        L_x = [x0]
        L_y = [y0]
        while x0 <= xf:
            x1 = x0+h
            k1 = h*f(x0,y0)
            k2 = h*f(x0+h/2,y0+k1/2)
            k3 = h*f(x0+h/2,y0+k2/2)
            k4 = h*f(x1,y0+k3)
            y1 = y0+(k1+2*k2+2*k3+k4)/6
            L_x.append(x1),L_y.append(y1)
            x0,y0 = x1,y1 
        return L_x,L_y

    def RK4_DSHO(self,x,v,t,tf,h,f_x,f_v):
        L_X = [x]
        L_t = [t]
        L_v = [v]
        while t <= tf:
            k1x = h*f_x(x,v,t)
            k1v = h*f_v(x,v,t)
            
            k2x = h*f_x(x+k1x/2,v+k1v/2,t+h/2)
            k2v = h*f_v(x+k1x/2,v+k1v/2,t+h/2)

            k3x = h*f_x(x+k2x/2,v+k2v/2,t+h/2)
            k3v = h*f_v(x+k2x/2,v+k2v/2,t+h/2)

            k4x = h*f_x(x+k3x,v+k3v,t+h)
            k4v = h*f_v(x+k3x,v+k3v,t+h)

            x = x + (k1x + 2*k2x + 2*k3x + k4x)/6
            v = v + (k1v + 2*k2v + 2*k3v + k4v)/6
            t = t + h
            L_t.append(t),L_v.append(v),L_X.append(x)
        return L_t,L_X,L_v
    
# -----------------Assign_15.py ----------------------------
    def BVP(self,x,final,gh,gl,t,tf,f_x,f_v):
        x1,T1,z1 = self.RK4_DSHO(x,gh,t,tf,0.1,f_x,f_v)
        x2,T2,z2 = self.RK4_DSHO(x,gl,t,tf,0.1,f_x,f_v)
        g = gl + (((gh - gl)*(final - T2[-1]))/(T1[-1] - T2[-1]))
        x3,T3,z3 = self.RK4_DSHO(x,g,t,tf,0.1,f_x,f_v)
        return x1,T1,z1,x2,T2,z2,x3,T3,z3
        


# ----------------Assign-16.py ----------------------------

class Fitting_data():
    def lagrange_interpolation(self,lx, ly, x):
        n = len(lx)
        P = 0
        for i in range(n):
            c=1
            for k in range(n):
                if k != i:
                    c *= ((x-lx[k])/(lx[i]-lx[k]))
            P+=c*ly[i]
        return P
    
    def lagrange_polynomial(self,lx, ly):
      lag_func = lambda x: self.lagrange_interpolation(lx, ly, x)
      return lag_func
    
    def linear_reg_params(self,Lx,Ly,sigma_i):
        n = len(Lx)
        S,Sx,Sy,Sxx,Sxy,Syy = 0,0,0,0,0,0
        for i in range(n):
            S += 1/sigma_i[i]
            Sx += Lx[i]/sigma_i[i]
            Sy += Ly[i]/sigma_i[i]
            Sxx += (Lx[i]*Lx[i])/sigma_i[i]
            Sxy += (Lx[i]*Ly[i])/sigma_i[i]
            Syy += (Ly[i]*Ly[i])/sigma_i[i]
        delta = S*Sxx - Sx*Sx
        r2 = (Sxy)**2/(Sxx*Syy)
        a = (Sxx*Sy - Sx*Sxy)/delta
        b = (S*Sxy - Sx*Sy)/delta
        err_a = (Sxx/delta)**0.5
        err_b = (S/delta)**0.5
        return a,b,r2
    
# -------------------- Endsem -------------------------------

    def poly_fit(self,x, y, deg):
        A = [[0 for j in range(deg+1)] for i in range(deg+1)]
        B = [0 for i in range(deg+1)]
        for i in range(deg+1):
            for k in range(deg+1):
                p=0
                for j in range(len(x)):
                    p += (x[j])**(i+k)
                A[i][k] = p
            q=0
            for j in range(len(x)):
                q += y[j]*(x[j])**i
            B[i] = q
        a = Gauss_Jordon_Elimination().agumented_matrix(A, [[val] for val in B])
        a = Gauss_Jordon_Elimination().GJE(a)
        a = [a[i][deg+1] for i in range(deg+1)]
        return a 

    def gauss_hermit(self,f,n):
        new_func = lambda x: f(x)*np.exp(x**2)
        h = np.polynomial.hermite.hermgauss(n)
        roots,weights = h[0], h[1]
        integral = 0
        for i in range(n):
            integral += weights[i]*new_func(roots[i])
        return integral

    def gauss_laguerre(self,f,n):    
        new_func = lambda x: f(x)*np.exp(x)
        L = np.polynomial.laguerre.laggauss(n)
        roots, weights = L[0], L[1]
        integral = 0
        for i in range(n):
            integral += weights[i]*new_func(roots[i])
        return integral
    
    
    def heat_eqn(self,nt,nx,x0,X,t0,T,Yx0):
        # X is length of rod
        dx = (X-x0)/(nx-1)
        dt = (T-t0)/(nt-1)
        a= dt/(dx)**2
        V0 = []
        T0 = []
        X0 = []
        for i in range(nx):
            X0.append(x0 + i*dx)
        for t in range(nt):
            T0.append(t0 + t*dt)
        for i in range(nx):
            V0.append([Yx0(x0+i*dx)])
        A = [[0 for i in range(nx)] for i in range(nx)]  # A = nx x nx matrix
        res = [V0]
        for i in range(nx):
            for j in range(nx):
                if  i==j :
                    A[i][j] = 1 - 2*a
                if i == j-1 or i == j+1:
                    A[i][j] = a
        for k in range(0,nt):
            rr = Matrix_Operation().matrix_multiply(A,res[k])
            res.append(rr)
        return res, T0, X0

    def RK4_new(self,F, t0, ly0, t_end, h):
        t = t0
        y = np.array(ly0, dtype=float)

        t_values = [t0]
        y_values = [y.copy()]

        while t < t_end:
            h_f = min(h, t_end - t)
            k1 = h_f * F(t, y)
            k2 = h_f * F(t + h_f / 2, y + k1 / 2)
            k3 = h_f * F(t + h_f / 2, y + k2 / 2)
            k4 = h_f * F(t + h_f, y + k3)

            y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            t = t + h_f

            t_values.append(t)
            y_values.append(y.copy())

            if h_f != h:
                break

        return np.array(t_values), np.array(y_values)
