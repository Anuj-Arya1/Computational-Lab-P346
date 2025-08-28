
# l = [2,4,5,6,7,8,6,5,3,2,7,87]
# print(l[4:])
l=[]
print(l)







#  # write code for polar plot in python i have data of angles and voltage corresponding to it \
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# # from prettytable import PrettyTable

# # Sample data
# angles = [-19,11,31,42,51,61,71,81,91,101,111,131,141,151,161,171,181,191,201,211,221,231,241,251,261,271,281,291,301,311,-39,-29,-19]
# angle =[a+19 for a in angles]
# voltages = [0.15,0.112,0.061,0.035,0.0176,0.0044,0.0001,0.0052,0.002,0.038,0.063,0.112,0.134,0.146,0.15,0.144,0.13,0.11,0.0846,0.0601,0.0341,0.0151,0.0038,0.0002,0.0056,0.018,0.039,0.063,0.091,0.114,0.134,0.147,.151]

# V2 =[0.061,0.108,0.139,0.143,0.138,0.126,0.106,0.082,0.058,0.034,0.016,0.0038,0.0002,0.0053,0.0186,0.0389,0.0638,0.0887,0.111,0.128,0.139,0.143,0.137,0.123,0.103,0.08,0.054,0.032,0.015,0.004,0.0001,0.004,0.018,0.037,0.061]
# t2_sample = [30,50,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,10,20,30]

# t2 = [t-30 for t in t2_sample]
# print(t2)
# # table 4
# v4 = [0.077,0.078,0.077,0.076,0.074,0.071,0.069,0.069,0.072,0.075,0.077,0.078,0.076,0.073,0.07,0.068,0.069,0.071,0.073,0.075,0.076,0.077]
# t4 = [0,20,40,50,60,80,100,120,140,160,180,200,220,240,260,280,300,320,330,-20,-10,0]
# t4_radians = [math.radians(a) for a in t4]
# # table 5
# V5=[0.098,0.113,0.109,0.088,0.062,0.038,0.033,0.046,0.087,0.108,0.113,0.099,0.074,0.047,0.033,0.037,0.058,0.072,0.087,0.098]
# t5=[0,20,40,60,80,100,120,140,170,190,210,230,250,270,290,310,330,-20,-10,0]
# t5_radians = [math.radians(a) for a in t5]
# #Convert angles to radians
# angle_rad = [math.radians(ang) for ang in angle]
# t2_radians = [math.radians(a) for a in t2]
# coslist =[]
# for i in angle_rad:
#     coslist.append((math.cos(i))**2)

# # Create polar plot
# plt.polar(angle_rad, voltages)
# plt.polar(t2_radians,V2)

# # Add title
# plt.title("Polar Plot of Voltage vs Angle(in deg.)")
# # Show plot
# plt.show()
# plt.polar(t5_radians,V5)
# plt.show()

# # table 4 plot
# plt.polar(t4_radians,v4)
# plt.show()