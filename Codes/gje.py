
def agumented_matrix(A,b):
    m = [[0 for i in range(len(A)+1)] for j in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A)):
            m[j][i] = A[j][i]
    for i in range(len(A)):
        m[i][len(A)] = b[i]
    return m 

def GJE(A):
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
        b[i] /= diag

        for k in range(n):
            if k != i:
                factor = A[k][i]
                for j in range(n):
                    A[k][j] -= factor * A[i][j]
                b[k] -= factor * b[i] 
    for i in range(len(A)):
        A[i][len(A)] = b[i]
    return A    

A=[ [0,2,5],
    [3,-1,2],
    [1,-1,3]]

b = [1,-2,3]

A = agumented_matrix(A,b)
print(GJE(A))


















