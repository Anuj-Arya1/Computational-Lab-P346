
import matplotlib.pyplot as plt
import numpy as np
from anuj_library import *

o1 = Gauss_Jordon_Elimination()
o2 = Matrix_Operation()
o3 = Fitting_data()


# # Q5
#By manual calculation we get y_max= 5m
def RK4_vector(F, t0, y0_vector, t_end, h):
    t = t0
    y = np.array(y0_vector, dtype=float)

    t_values = [t0]
    y_values = [y.copy()]

    while t < t_end:
        h_actual = min(h, t_end - t)

        k1 = h_actual * F(t, y)
        k2 = h_actual * F(t + h_actual / 2, y + k1 / 2)
        k3 = h_actual * F(t + h_actual / 2, y + k2 / 2)
        k4 = h_actual * F(t + h_actual, y + k3)

        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        t = t + h_actual

        t_values.append(t)
        y_values.append(y.copy())

        if h_actual != h:
            break

    return np.array(t_values), np.array(y_values)
v0 = 10.0
g = 10.0
gamma = 0.02

def F(t, Y):
    y, v = Y
    dy_dt = v
    dv_dt = -gamma * v - g
    return np.array([dy_dt, dv_dt])

t0 = 0.0
t_end = 10.0  # Any arbitrary time greater than its time of flight works here 
h = 0.001

t_values, Y_values = RK4_vector(F, 0, [0.0, 10], 10, 0.001)
y_values = Y_values[:, 0]
v_values = Y_values[:, 1]
a=0
max=0
for i in range(len(y_values)):
    if max < y_values[i]:
        max= y_values[i]
        a=i
    else: 
        pass
y_max = Y_values[a]

print(f"Q5: Maximum height (from RK4)", y_max[0])
plt.figure()
plt.plot(y_values, v_values)
plt.xlabel("Height y")
plt.ylabel("Velocity v")
plt.title("Velocity vs height with air resistance")
plt.grid(True)
plt.show()




# def RK4(L_x,L_y,x0,y0,xf,f,h):
#     L_x.append(x0),L_y.append(y0)
#     x1 = x0+h
#     k1 = h*f(x0,y0)
#     k2 = h*f(x0+h/2,y0+k1/2)
#     k3 = h*f(x0+h/2,y0+k2/2)
#     k4 = h*f(x1,y0+k3)
#     y1 = y0+(k1+2*k2+2*k3+k4)/6
#     if x1>=xf-10**(-6)*h:
#         L_x.append(x1),L_y.append(y1)
#         return L_x,L_y
#     else:
#         return RK4(L_x,L_y,x1,y1,xf,f,h)


# def RK4_DSHO(L_X,L_t,L_v,L_E,x,v,t,tf,h,k,m,mu):
#     E0 = 0.5*(m*v**2 + k*x**2)
#     L_t.append(t),L_v.append(v),L_E.append(E0),L_X.append(x)
#     w2 = k/m
#     k1x = h*v
#     k1v = -h*(mu*v + (w2)*x)
#     k2x = h*(v+k1v/2)
#     k2v = -h*(mu*k1v + (w2)*k2x)
#     k3x = h*(v+k2v/2)
#     k3v = -h*(mu*k2v + (w2)*k3x)
#     k4x = h*(v+k3v/2)
#     k4v = -h*(mu*k3v + (w2)*k4x)
#     x1 = x + (k1x + 2*k2x + 2*k3x + k4x)/6
#     v1 = v + (k1v + 2*k2v + 2*k3v + k4v)/6
#     t1 = t + h
#     if t>= tf - 10**(-6):
#         return L_t,L_X,L_v,L_E
#     else:
#         return RK4_DSHO(L_X,L_t,L_v,L_E,x1,v1,t1,tf,h,k,m,mu)
        



# # # Adaptive runge kutta method
# # def adaptive_rk(f, y0, t0, tf, h0, tol):
# #     t = t0
# #     y = y0
# #     h = h0
# #     result = [(t, y)]

# #     while t < tf:
# #         if t + h > tf:
# #             h = tf - t

# #         # Single step with step size h
# #         k1 = f(t, y)
# #         k2 = f(t + h / 2, y + h / 2 * k1)
# #         k3 = f(t + h / 2, y + h / 2 * k2)
# #         k4 = f(t + h, y + h * k3)
# #         y1 = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# #         # Two steps with step size h/2
# #         h2 = h / 2
# #         k1 = f(t, y)
# #         k2 = f(t + h2 / 2, y + h2 / 2 * k1)
# #         k3 = f(t + h2 / 2, y + h2 / 2 * k2)
# #         k4 = f(t + h2, y + h2 * k3)
# #         y_half = y + (h2 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# #         t_half = t + h2
# #         k1 = f(t_half, y_half)
# #         k2 = f(t_half + h2 / 2, y_half + h2 / 2 * k1)
# #         k3 = f(t_half + h2 / 2, y_half + h2 / 2 * k2)
# #         k4 = f(t_half + h2, y_half + h2 * k3)
# #         y2 = y_half + (h2 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# #         # Estimate the error
# #         error = abs(y1 - y2)

# #         if error < tol:
# #             # Accept the step
# #             t += h
# #             y = y1
# #             result.append((t, y))

# #         # Adjust step size
# #         if error == 0:
# #             s = 2
# #         else:
# #             s = (tol * h / (2 * error)) ** 0.25

# #         h *= s
# #         h = max(h, h2)

# #     return result

# # # result 

# import math as m

# # inverse calculate using LU decomposition

# # def inverse_LU(L,U):
# #     n = len(L)
# #     inv = [[0 for _ in range(n)] for _ in range(n)]
# #     for i in range(n):
# #         e = [0]*n
# #         e[i] = 1
# #         y = [0]*n
# #         for j in range(n):
# #             y[j] = e[j]
# #             for k in range(j):
# #                 y[j] -= L[j][k]*y[k]
# #             y[j] /= L[j][j]
# #         x = [0]*n
# #         for j in range(n-1, -1, -1):
# #             x[j] = y[j]
# #             for k in range(j+1, n):
# #                 x[j] -= U[j][k]*x[k]
# #             x[j] /= U[j][j]
# #         for j in range(n):
# #             inv[j][i] = x[j]
# #     return inv

# # def determinant(A):
# #     n = len(A)
# #     if n == 1:
# #         return A[0][0]
# #     if n == 2:
# #         return A[0][0]*A[1][1] - A[0][1]*A[1][0]
# #     det = 0
# #     for c in range(n):
# #         minor = [row[:c] + row[c+1:] for row in A[1:]]
# #         det += ((-1)**c) * A[0][c] * determinant(minor)
# #     return det

# def determinant(A):
#     n = len(A) 
#     if n==1:
#         return A[0][0]
#     elif n==2:
#         return A[0][0]*A[1][1] - A[0][1]*A[1][0]
#     elif [row[0] for row in A] == [0 for _ in range(n)]:
#         return 0
#     else:
#         det = 1
#         p=0
#         for i in range(len(A)):
#             if abs(A[i][0])> p:
#                 p = A[i][0]
#                 m = i 
#         if m != 0:
#             A[m],A[0] = A[0],A[m]
#             det *= -1
#         for i in range(n):
#             for j in range(i+1,n):
#                 if A[i][i] != 0:
#                     fact = A[j][i] / A[i][i]
#                     for k in range(n):
#                         A[j][k] -= fact * A[i][k]
#                 else:
#                     # If the pivot element is zero, we need to swap rows
#                     for k in range(j+1, n):
#                         if A[k][i] != 0:
#                             A[j], A[k] = A[k], A[j]
#                             det *= -1
#                             break 
#         # calculate determinant
#         for i in range(n): 
#             det *= A[i][i] 
#     return det


# # Van der pol oscillator project 

# """
# Project: 2) Adaptive Runge-Kutta: Van der Pol oscillator
#          3) SVD image compression

# This file contains:
#   - A compact 4-5 page repoimport shutil

# # ...existing code...

# if os.path.exists(example_image):
#     A = load_image_grayscale(example_image, max_size=512)
#     print('Image shape:', A.shape)
#     svd_results = plot_svd_reconstructions(A, ranks=(5,10,50,100,200,250), outdir=outdir)
#     print('SVD results:', svd_results)
#     # Copy the original image to output folder
#     shutil.copy(example_image, os.path.join(outdir, os.path.basename(example_image)))
# else:
#     print(f"No example image found at '{example_image}'. Please place an image with that name or change the path in main().")rt outline (as comments) you can paste into your report.
#   - A runnable Python starter notebook/script with:
#       * fixed-step RK4 integrator
#       * wrappers using scipy.solve_ivp (adaptive RK45 / DOP853)
#       * comparison harness (error / step-count / CPU time) for Van der Pol
#       * plotting helpers to visualize time-series and step-size behavior
#       * SVD image compression functions and metrics (Frobenius error, PSNR)
#       * example `if __name__ == "__main__"` usage demonstrating both projects

# Requirements:
#   numpy, scipy, matplotlib, pillow (PIL)
#   Install: pip install numpy scipy matplotlib pillow

# Use:
#   - Edit parameters in the `main()` examples at the bottom.
#   - Run as a script: python van_der_pol_svd.py

# References (put in your report):
#   - E. Hairer, S.P. Norsett, G. Wanner. Solving Ordinary Differential Equations I.
#   - J. D. Lambert. Numerical Methods for Ordinary Differential Systems.
#   - G. H. Golub & C. F. Van Loan. Matrix Computations.
#   - Madras & Sokal (for SAW) — not used here but useful to cite generally.

# """

# # ============================
# # 4-5 PAGE REPORT OUTLINE
# # ============================
# # (Copy-paste this block into your report and expand as needed)
# #
# # Title: Numerical study of the Van der Pol oscillator (adaptive vs fixed-step)
# #        and SVD-based image compression
# #
# # 1. Objective (1 paragraph)
# #    - Introduce the Van der Pol oscillator and why stiffness appears for large mu.
# #    - State comparative goal: adaptive RK vs fixed RK4 in accuracy/efficiency.
# #    - Introduce the SVD compression task and metrics (Frobenius norm, PSNR).
# #
# # 2. Problem statement (math)
# #    - Van der Pol: x'' - mu(1-x^2)x' + x = 0. Put as first-order system y' = f(t,y).
# #    - SVD: Given image A (m x n), compute rank-r approx A_r minimizing ||A-A_r||_F.
# #
# # 3. Numerical method (algorithms, parameters)
# #    - Fixed-step RK4: show Butcher tableau or update formula and step selection.
# #    - Adaptive RK: summarize Dormand–Prince / RKF45 idea and local error control.
# #    - For SVD: use numpy.linalg.svd; for large images mention randomized SVD.
# #    - List tolerances, time-span, initial conditions used.
# #
# # 4. Implementation details (short code description)
# #    - Mention verification: refine dt or tol and check convergence; energy-like diagnostics.
# #    - For SVD: mention reconstruction, PSNR calculation.
# #
# # 5. Results and Figures (2-3 figures)
# #    - Figure 1: x(t) for mu=1 and mu=100 comparing RK4 and adaptive solver.
# #    - Figure 2: adaptive step-size h(t) vs t for mu=100.
# #    - Figure 3: reconstructed images for several r and error vs r plot.
# #
# # 6. Discussion and verification
# #    - Discuss where RK4 fails or requires tiny dt for mu=100; adaptive solver advantages.
# #    - Discuss SVD compression trade-offs and visual artifacts.
# #
# # 7. References
# #    - List the references above and any webpages used.
# #
# # ============================
# # START OF PYTHON CODE
# # ============================

# import time
# import numpy as np
# from math import isfinite
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
# from PIL import Image
# import os
# import shutil

# # ----------------------------
# # Van der Pol system
# # ----------------------------

# def van_der_pol(t, y, mu):
#     """Van der Pol as first-order system.
#     y = [x, x_dot]
#     y' = [x_dot, mu*(1-x^2)*x_dot - x]
#     Note sign: original x'' - mu(1-x^2)x' + x = 0 -> x'' = mu(1-x^2)x' - x
#     """
#     x, xdot = y
#     return np.array([xdot, mu*(1 - x*x)*xdot - x])


# def rk4_step(f, t, y, h, *args):
#     k1 = f(t, y, *args)
#     k2 = f(t + 0.5*h, y + 0.5*h*k1, *args)
#     k3 = f(t + 0.5*h, y + 0.5*h*k2, *args)
#     k4 = f(t + h, y + h*k3, *args)
#     return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)


# def integrate_rk4(f, t_span, y0, dt, args=()):
#     t0, tf = t_span
#     n_steps = int(np.ceil((tf - t0)/dt))
#     ts = np.zeros(n_steps+1)
#     ys = np.zeros((n_steps+1, len(y0)))
#     ts[0] = t0
#     ys[0] = y0
#     t = t0
#     y = y0.copy()
#     for i in range(n_steps):
#         h = min(dt, tf - t)
#         y = rk4_step(f, t, y, h, *args)
#         t = t + h
#         ts[i+1] = t
#         ys[i+1] = y
#     return ts, ys


# def integrate_adaptive(f, t_span, y0, args=(), rtol=1e-6, atol=1e-9, method='RK45'):
#     # Uses scipy.integrate.solve_ivp with dense output
#     sol = solve_ivp(lambda t, y: f(t, y, *args), t_span, y0, method=method,
#                     rtol=rtol, atol=atol, dense_output=True)
#     return sol.t, sol.y.T, sol


# def compare_vdp(mu=100.0, t_span=(0.0, 300.0), y0=(2.0, 0.0), dt=0.01, rtol=1e-6):
#     """Run a comparison between fixed RK4 and adaptive RK.
#     Returns dictionaries of results including CPU times and errors vs reference.
#     """
#     # Reference solution: adaptive with very tight tolerances
#     print(f"Computing reference solution for mu={mu}...")
#     t_ref, y_ref, sol_ref = integrate_adaptive(van_der_pol, t_span, np.array(y0), args=(mu,), rtol=1e-10, atol=1e-12, method='DOP853')

#     # Interpolator for reference
#     def interp_ref(t_query):
#         return sol_ref.sol(t_query)

#     # Fixed-step RK4
#     print("Running fixed-step RK4...")
#     t0 = time.time()
#     ts_rk4, ys_rk4 = integrate_rk4(van_der_pol, t_span, np.array(y0), dt, args=(mu,))
#     rk4_time = time.time() - t0

#     # Adaptive (user tolerance)
#     print("Running adaptive RK (solve_ivp)...")
#     t0 = time.time()
#     ts_adapt, ys_adapt, sol_adapt = integrate_adaptive(van_der_pol, t_span, np.array(y0), args=(mu,), rtol=rtol, atol=1e-9, method='RK45')
#     adapt_time = time.time() - t0

#     # Compute errors by interpolating reference onto solver times
#     def compute_error(ts, ys):
#         ys_ref_on_ts = np.array([interp_ref(ti) for ti in ts])
#         err = np.linalg.norm(ys - ys_ref_on_ts, axis=1)  # Euclidean error per time
#         return err

#     err_rk4 = compute_error(ts_rk4, ys_rk4)
#     err_adapt = compute_error(ts_adapt, ys_adapt)

#     results = {
#         'reference': {'t': t_ref, 'y': y_ref},
#         'rk4': {'t': ts_rk4, 'y': ys_rk4, 'time': rk4_time, 'err': err_rk4},
#         'adaptive': {'t': ts_adapt, 'y': ys_adapt, 'time': adapt_time, 'err': err_adapt, 'sol': sol_adapt}
#     }
#     return results


# # ----------------------------
# # Plotting helpers for Van der Pol
# # ----------------------------

# def plot_vdp_results(results, mu, outdir=None):
#     ref = results['reference']
#     rk4 = results['rk4']
#     adapt = results['adaptive']

#     plt.figure(figsize=(10,4))
#     plt.plot(ref['t'], ref['y'][:,0], label='reference x(t)', linewidth=1)
#     plt.plot(rk4['t'], rk4['y'][:,0], label='RK4 x(t)', linewidth=0.8, alpha=0.7)
#     plt.plot(adapt['t'], adapt['y'][:,0], label='Adaptive x(t)', linewidth=0.8, alpha=0.7)
#     plt.legend(); plt.title(f'Van der Pol x(t), mu={mu}'); plt.xlabel('t'); plt.tight_layout()
#     if outdir:
#         plt.savefig(os.path.join(outdir, f'vdp_xt_mu_{mu}.png'), dpi=150)
#     else:
#         plt.show()
#     plt.close()

#     # plot error vs time for both
#     plt.figure(figsize=(10,3))
#     plt.semilogy(rk4['t'], rk4['err'], label='RK4 error')
#     plt.semilogy(adapt['t'], adapt['err'], label='Adaptive error')
#     plt.legend(); plt.title('Error vs time (Euclidean error to reference)'); plt.xlabel('t'); plt.tight_layout()
#     if outdir:
#         plt.savefig(os.path.join(outdir, f'vdp_err_mu_{mu}.png'), dpi=150)
#     else:
#         plt.show()
#     plt.close()

#     # plot adaptive step sizes (if dense output available, plot distances between time points)
#     h_adapt = np.diff(adapt['t'])
#     plt.figure(figsize=(10,3))
#     plt.plot(adapt['t'][:-1], h_adapt)
#     plt.yscale('log'); plt.xlabel('t'); plt.ylabel('h (adaptive)'); plt.title('Adaptive step size vs time'); plt.tight_layout()
#     if outdir:
#         plt.savefig(os.path.join(outdir, f'vdp_steps_mu_{mu}.png'), dpi=150)
#     else:
#         plt.show()
#     plt.close()

#     print(f"RK4 cpu time: {rk4['time']:.3f}s, adaptive cpu time: {adapt['time']:.3f}s")


# # ----------------------------
# # SVD image compression
# # ----------------------------

# def load_image_grayscale(path, max_size=None):
#     img = Image.open(path).convert('L')
#     if max_size is not None:
#         img.thumbnail((max_size, max_size))
#     arr = np.array(img).astype(float) / 255.0
#     return arr


# def svd_truncate(A, r):
#     # A is 2D array; returns rank-r reconstruction
#     U, s, Vt = np.linalg.svd(A, full_matrices=False)
#     Ur = U[:, :r]
#     sr = s[:r]
#     Vtr = Vt[:r, :]
#     A_r = (Ur * sr) @ Vtr
#     return A_r, (U, s, Vt)


# def frobenius_rel_error(A, A_r):
#     num = np.linalg.norm(A - A_r, 'fro')
#     den = np.linalg.norm(A, 'fro')
#     return num/den


# def psnr(A, A_r):
#     mse = np.mean((A - A_r)**2)
#     if mse == 0:
#         return float('inf')
#     max_I = 1.0
#     return 20 * np.log10(max_I / np.sqrt(mse))


# def plot_svd_reconstructions(A, ranks=(5,10,50,100), outdir=None, cmap='gray'):
#     fig, axes = plt.subplots(1, len(ranks)+1, figsize=(3*(len(ranks)+1),3))
#     axes[0].imshow(A, cmap=cmap); axes[0].set_title('Original'); axes[0].axis('off')
#     errors = []
#     psnrs = []
#     for i, r in enumerate(ranks):
#         A_r, _ = svd_truncate(A, r)
#         err = frobenius_rel_error(A, A_r)
#         p = psnr(A, A_r)
#         errors.append(err)
#         psnrs.append(p)
#         axes[i+1].imshow(np.clip(A_r,0,1), cmap=cmap)
#         axes[i+1].set_title(f'r={r}\nerr={err:.3e}\nPSNR={p:.2f}dB')
#         axes[i+1].axis('off')
#     plt.tight_layout()
#     if outdir:
#         plt.savefig(os.path.join(outdir, 'svd_reconstructions.png'), dpi=150)
#     else:
#         plt.show()
#     plt.close()

#     # error vs r plot
#     plt.figure(figsize=(6,3))
#     plt.plot(ranks, errors, marker='o')
#     plt.yscale('log'); plt.xlabel('rank r'); plt.ylabel('relative Frobenius error'); plt.title('Error vs rank')
#     if outdir:
#         plt.savefig(os.path.join(outdir, 'svd_error_vs_rank.png'), dpi=150)
#     else:
#         plt.show()
#     plt.close()

#     return {'ranks': ranks, 'errors': errors, 'psnr': psnrs}


# # ----------------------------
# # Example driver
# # ----------------------------

# def main(outdir=None):
#     if outdir and not os.path.exists(outdir):
#         os.makedirs(outdir)

#     # --- Van der Pol experiment ---
#     mu_list = [1.0, 100.0]
#     for mu in mu_list:
#         print('Running Van der Pol for mu=', mu)
#         results = compare_vdp(mu=mu, t_span=(0.0, 250.0), y0=(2.0, 0.0), dt=0.01, rtol=1e-6)
#         plot_vdp_results(results, mu, outdir=outdir)

#     # --- SVD experiment ---
#     # Put an example image path here (replace with your own file)
#     example_image = 'C:\\Users\\Lenovo\\OneDrive - niser.ac.in\\N_computer\\Computational-Lab-P346\\Computational-Lab-P346\\Codes\\image.png'
#     # example_image = 'C:\Users\Lenovo\OneDrive - niser.ac.in\N_computer\Computational-Lab-P346\Computational-Lab-P346\Project\image02.png'  # Change to your image path
#     if os.path.exists(example_image):
#         A = load_image_grayscale(example_image, max_size=512)
#         print('Image shape:', A.shape)
#         svd_results = plot_svd_reconstructions(A, ranks=(5,10,50,100,200,250), outdir=outdir)
#         print('SVD results:', svd_results)
#         # Copy the original image to output folder
#         shutil.copy(example_image, os.path.join(outdir, os.path.basename(example_image)))
#     else:
#         print(f"No example image found at '{example_image}'. Please place an image with that name or change the path in main().")


# if __name__ == '__main__':
#     # Example: create an `output` folder in the current directory to save plots
#     main(outdir='output')

# # End of file
