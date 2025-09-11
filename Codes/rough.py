
# def Polynm(x,coeff):
#     pol = 0
#     for i in range(len(coeff)):
#         pol += coeff[len(coeff)-(i+1)]*x**(i)
#     return pol

# def deriv(coeff):
#     coeff_new =[]
#     maxdeg = len(coeff)-1
#     for j in range(len(coeff)):
#         coeff_new.append((maxdeg-j)*coeff[j])
#     return coeff_new

# def deflation(coeff_p1,r):
#     list1 = [coeff_p1[0]]
#     for i in range(1,len(coeff_p1)):
#         p=  r*list1[i-1] + coeff_p1[i]
#         list1.append(p)
#     return list1[:-1]

# def laguerre_algo(b0,coeff):
#     e = 10**(-6)
#     res = []
#     root = b0
#     poly = Polynm(root,coeff)
#     if poly <= e:
#         res.append(root)
#         # print(res)
#         dd = deflation(coeff,root)
#         return laguerre_algo(root,dd)
#     else: 
#         poly1 = Polynm(b0,coeff)
#         n= len(coeff)-1
#         xx = Polynm(b0,deriv(coeff))
#         G = xx/poly1
#         poly22 = Polynm(b0,deriv(deriv(coeff)))
#         H = G**2 - (poly22/poly1)
#         a0 =(n-1)*(n*H-G**2)
#         print("and",a0)
#         m1 = G + math.sqrt(a0)
#         m2 = G - math.sqrt(a0)
#         if abs(m1)>= abs(m2):
#             a = n/m1
#         else:
#             a = n/m2
        
#         b1 = b0 - a
#         if abs(b1-b0)<=e:
#             root = b1
#         else:
#             return laguerre_algo(b1,coeff)
#     return res

# print(laguerre_algo(1,coeff_p1))