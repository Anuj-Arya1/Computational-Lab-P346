from matplotlib.mlab import detrend
import numpy as np
import matplotlib.pyplot as plt
import math as m
import os
import scipy.integrate as spi
import random


# ============== PART 1: Comparision of different numerical methods ==============

def calculate_error(traj1, traj2):
    diff = traj1 - traj2
    return np.sqrt(np.mean(diff**2))

def plot_all_trajectories(results, t_eval):
    fig = plt.figure(figsize=(16, 12))
    # 3D subplots
    methods = [k for k in results.keys() ]
    n_met = len(methods)
    rows = (n_met + 2) // 3
    
    for idx, method in enumerate(methods, 1):
        ax = fig.add_subplot(rows, 3, idx, projection='3d')
        traj = results[method]['solution']
        
        x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
        ax.plot(x, y, z, linewidth=0.5, alpha=0.7)
        
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.set_title(f'{method}\n({results[method]["order"]} order, {results[method]["time"]:.3f}s)', 
                    fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=7)
    
    plt.suptitle('Rössler Attractor - All Methods Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    # plt.savefig('all_methods_3d.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_accuracy_comparison(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))    
    # Use DOP853 as reference 
    ref_traj = results['DOP853']['solution']
    methods = [k for k in results.keys() ]
    errors = []
    times = []
    names = []
    
    for method in methods:
        if method == 'DOP853':
            continue # Skip reference
        traj = results[method]['solution']
        error = calculate_error(traj, ref_traj)
        errors.append(error)
        times.append(results[method]['time'])
        names.append(method)
    
    # Plot 1: Error by method
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    bars = ax1.bar(range(len(names)), errors, color=colors)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('RMS Error vs DOP853', fontsize=11, fontweight='bold')
    ax1.set_title('Accuracy (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Time by method
    ax2 = axes[0, 1]
    all_methods = [k for k in results.keys() ]
    all_times = [results[m]['time'] for m in all_methods]
    colors2 = plt.cm.plasma(np.linspace(0, 1, len(all_methods)))
    bars2 = ax2.bar(range(len(all_methods)), all_times, color=colors2)
    ax2.set_xticks(range(len(all_methods)))
    ax2.set_xticklabels(all_methods, rotation=45, ha='right')
    ax2.set_ylabel('Computation Time (s)', fontsize=11, fontweight='bold')
    ax2.set_title('Speed (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    axes[1, 1].axis('off')
    
    # Plot 3: Accuracy vs Speed
    ax3 = axes[1, 0]
    scatter_colors = plt.cm.coolwarm(np.linspace(0, 1, len(names)))
    for i, name in enumerate(names):
        ax3.scatter(times[i], errors[i], s=200, color=scatter_colors[i], 
                   alpha=0.7, edgecolors='black', linewidth=2)
        ax3.annotate(name, (times[i], errors[i]), fontsize=9, 
                    xytext=(5, 5), textcoords='offset points')
    
    ax3.set_xlabel('Computation Time (s)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('RMS Error', fontsize=11, fontweight='bold')
    ax3.set_title('Accuracy vs Speed\n(Top-left corner is best)', 
                 fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    



# ============== PART 2: RÖSSLER SYSTEM Analysis ==============
def rossler_attractor_equ(t, state, a, b, c):
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return np.array([dxdt, dydt, dzdt])

def rossler_sol_lists(sol,):
    t_values = sol.t
    x_values = sol.y[0].tolist()
    y_values = sol.y[1].tolist()
    z_values = sol.y[2].tolist()
    return t_values, x_values, y_values, z_values

def plot_3d(x_values, y_values, z_values, int_cond,colr):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_values, y_values, z_values,color= colr)
    ax.set_title(f'Rössler attractor (3D) - {int_cond}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.show()

def plot_2d(x_values, y_values, int_con, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values)
    plt.title(f"Rössler attractor with initial condition-{int_con}({xlabel} vs {ylabel})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_time_series(t_values, x_values, y_values, z_values, int_con):
    rand_color = lambda : random.randint(0,255)/255.0
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, x_values, label='x(t)', color=(rand_color(), rand_color(), rand_color()))
    plt.plot(t_values, y_values, label='y(t)', color=(rand_color(), rand_color(), rand_color()))
    plt.plot(t_values, z_values, label='z(t)', color=(rand_color(), rand_color(), rand_color()))
    plt.title(f"Rössler attractor time series with initial condition-{int_con}")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.legend()
    plt.show()

def plot_spectrum(data, int_con,t1):
    x_proc = detrend(data - np.mean(data))
    N = len(x_proc)
    Xf = np.fft.rfft(x_proc)
    freqs = np.fft.rfftfreq(N, d=t1[1]-t1[0])
    psd = (np.abs(Xf)**2) / N
    # plt.figure(figsize=(4, 4))
    plt.semilogy(freqs, psd)
    plt.title(f"Power Spectrum of Rössler attractor with initial condition-{int_con}")  
    plt.xlabel("Frequency")
    plt.xlim(-0.01, 4)
    plt.ylim(1e-10, m.ceil(max(psd)))
    plt.ylabel("Power Spectral Density")
    plt.show()





