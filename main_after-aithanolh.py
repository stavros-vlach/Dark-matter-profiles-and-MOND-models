from numba import jit
import sys
import emcee
import corner
import numpy
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from scipy.odr import ODR, Model, Data, RealData
from scipy.optimize import differential_evolution
from scipy import stats
from multiprocessing import Pool, cpu_count
import time
import scipy
import scipy.integrate as integrate
import scipy.special as special
from scipy.integrate import quad
import os
from scipy.special import gammaincc, gamma
import utils as ut
from tqdm import tqdm
matplotlib.use('TkAgg')
labels = ["Galaxy",
          "Hubble Type",
          "Distance (Mpc)",
          "Mean D error (Mpc)",
          "Distance Method",
          "Inclination (deg)",
          "Mean Inc error (deg)",
          "Total Luminosity at [3.6](10+9solLum)",
          "Effective Radius at [3.6](kpc)",
          "Effective Surface Brightness at [3.6](solLum/pc2)",
          "Disk Scale Length at [3.6] (kpc)",
          "Disk Central Surface Brightness at [3.6] (solLum/pc2)",
          "Total HI mass (10+9solMass)",
          "HI radius at 1 Msun/pc2 (kpc)",
          "Asymptotically Flat Rotation Velocity(km/s)",
          "Mean Vflat error (km/s)",
          "Mean error on Vflat (km/s)",
          "Quality Flag (3)",
          "Refs"]

# GALAXY_NAME = input("galaxy name = ")
GALAXY_NAME = "UGC02953"
# GALAXY_NAME = 'IC2574'
# GALAXY_NAME = "ESO563-G021"
PROFILE = "Burkert"
# os.chdir("/media/stavros/aaf194dd-624d-4ece-8e56-395e27a93185/Pythonproject")
name = "Rotmod_LTG/" + GALAXY_NAME + "_rotmod.dat"
fname = f'newPar_fit_{GALAXY_NAME}__{PROFILE}'
R, V, Verr, Vgas, Vbul, Vdisk, = np.loadtxt(name, unpack=True, usecols=(0, 1, 2, 3, 4, 5))
data_length = len(R)

g_obs = (V ** 2) / R
# The full dataset of http://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt,
# with minor changes for parsing
data = pd.read_csv('data.txt', delimiter=';', skiprows=98, names=labels)
# print(data.iloc[1])
# print(stop)
data = data[data['Galaxy'] == GALAXY_NAME]
# -FOTIS-: BE CAREFUL --- inclination in rads EVERYWHERE
i_set = data["Inclination (deg)"].to_numpy()[0] * np.pi / 180
err_i_set = data["Mean Inc error (deg)"].to_numpy()[0] * np.pi / 180
D_quot = data["Distance (Mpc)"].to_numpy()[0]
err_D_quot = data["Mean D error (Mpc)"].to_numpy()[0]
L_tot = data["Total Luminosity at [3.6](10+9solLum)"].to_numpy()[0]

Lb = pd.read_csv('Lbul.txt', skiprows=7, names=['Galaxy', 'Lbul (10^9 Lsun)'])
#print(Lb['Galaxy'])
for i in Lb['Galaxy']:
    gal = i.split()
    if gal[0] == GALAXY_NAME:
        L_bul = float(gal[1])
        break

# print(D_quot)
# print(stop)
# ---- Fotis: this part needs to be organized as a function ----- #
# f = scipy.interpolate.interp1d(R, Vgas)
# V_gas = f(R)
g = scipy.interpolate.interp1d(R, Vbul)
ybul = g(R)

x = np.array(R)
y = np.array(V)
err_x = x * 0
err_y = np.array(Verr)

H0 = 73
h70 = H0 / 70
G = 4.3 * (10 ** (-6))
M_sun = 1.98847 * (10 ** 30)
kpc = 3.08567758 * (10 ** 16)

x_arr = np.array(R)
y_arr = np.array(V)
err_y_arr = np.array(Verr)

labels = ["$i_{fit}$", "$log_{10}D_{fit}$", "$log_{10}Y_{disk}$", "$log_{10}Y_{bul}$", "$M_{200}$", "$C_{200}$"]
optim_limits = [(0.0, i_set + 10*err_i_set), \
                 (-np.log10(1 + (err_D_quot / D_quot)),np.log10(1 + (err_D_quot / D_quot))),\
                (0.001, 0.5), \
                (0.001, 0.5), \
                (1, 1e15), \
                (3, 100)]
text = f'Fit of {PROFILE} profile on {GALAXY_NAME} rot curve data from SPARC'
nwalkers = 18#1000
max_iters = 2#000

    ##############################################################################################
#def func(s, a, b, gamma):
#    f = (s ** (2 - gamma)) / ((1 + s ** a) ** (b - gamma) / a)
#      return f

def calc_v_NFW_halo(quant200):
    _, _, _, _, M_200, C_200 = quant200
    r_200 = 0.02063 * ((M_200 / M_sun) ** (1/3)) * h70 ** (-2/3)
    #a, b, gamma = 1, 3, 1
    integral1 = np.log(1 + C_200 * x_arr / r_200) + 1 / (1 + C_200 * x_arr / r_200)
    integral2 = np.log(1 + C_200) + 1 / (1 + C_200)
    NFW_v = (10 ** 3) * ((4.3 * (M_200 / (M_sun * (10 ** 12))) / (x_arr / kpc)) * (integral1 / integral2)) ** (1/2)
    return NFW_v


def lnprob_withYs(x):
        # Parametrized according to https://iopscience.iop.org/article/10.3847/1538-4357/ac93fc/pdf
        i_fit, logD_fit, logY_disk, logY_bul, M_200, C_200 = x
        # -Fotis-: These calcs could probably be done AFTER the if bellow...
        Y_disk = (10 ** (logY_disk)) * 0.5
        Y_bul = (10 ** (logY_bul)) * 0.7
        # Y_gas = (10 ** (logY_gas)) * 1.33
        D_fit = 10 ** logD_fit
        #M_star = Y_disk * 0.5 * (L_tot - L_bul) + Y_bul * 0.7 * L_bul
        # - Fotis - : CAN WE GET RID OF Ygas somehow?
        # YES! see https://iopscience.iop.org/article/10.3847/1538-4357/abbb96/pdf
        # expr 4.
        X1 = 0.75 - 38.2 * ((Y_disk * 0.5 * (L_tot - L_bul) + Y_bul * 0.7 * L_bul) / \
            (1.5 * 10 ** 24)) ** 0.22
        Y_gas = 1/X1 *1.33
        # X1 **= (-1)
        # print(np.log10(X1/1.33) - 0.1)
        print(x)
        print('+++++++++++++++++++++++++++++++++++++++=')
        listConds = [not (0.0 <= i_fit <= i_set*10.5) ,
           not (0.001 <= D_fit <= 1e25),
           # not (0.001 <= Y_gas <= 0.9),
           not (0.0001 <= Y_disk <= 0.9),
           not(0.0001 <= Y_bul <= 0.9),
           not (0.0001 < M_200 < 1e20),
           not (0.0001 < C_200 < 1900)]

        for cond in listConds:
            print(cond)
        print('---------------------------------------------------')
        if not ( 0.0 <= i_fit <= i_set*1.5) \
           or not (-10*np.log10(1 + err_D_quot / D_quot) <= D_fit <= 10*np.log10(1 + err_D_quot / D_quot))\
           or not (0.1 < C_200 < 900)\
           or not (0.0001 <= Y_disk <= 0.3) or not (0.0001 <= Y_bul <= 0.3)\
           or not (0.1 < M_200):
            # print('Mphka')
            return -np.inf

        Vhalo = calc_v_NFW_halo(x)
        V_rot = V * np.sin(i_set) / np.sin(i_fit)

        # Vdisk = V_d(logMdisk)
        V_th = np.sqrt(Vhalo ** 2 + (Y_disk * Vdisk ** 2 + Y_gas * Vgas * np.abs(Vgas) + Y_bul * Vbul ** 2) * D_fit / D_quot)
        # print(f'v_theor: {V_th}')
        # print(f'V_obs: {y_arr*np.sin(i_fit*np.pi/180)/np.sin(i_set)}')
        dist = V_rot - V_th
        sigma = err_y_arr * np.sin(i_set) / np.sin(i_fit)
        chi_sq = (dist / sigma) ** 2
        L = np.exp(-chi_sq / 2) * 2 * np.pi * (sigma ** 2)

        # with open("test.dat", "ab") as f:
        #     np.savetxt(f, np.array([logMdisk, logrho0, r0]), fmt='%1.3f', newline=" ")
        #     f.write(b"\n")
        if np.min(L) < 1.e-300:
            return -1.e300

        return np.sum(np.log(L))

def chisq(x):
    '''
    This is the x^2 function - the object function for minimization
    '''
    i_fit, logD_fit, logY_disk, logY_bul, M_200, C_200 = x
    Y_disk = (10 ** (logY_disk)) * 0.5
    Y_bul = (10 ** (logY_bul)) * 0.7
    # Y_gas = (10 ** (logY_gas)) * 1.33
    D_fit = 10 ** logD_fit

    X1 = 0.75 - 38.2 * ((Y_disk * 0.5 * (L_tot - L_bul) + Y_bul * 0.7 * L_bul) / \
            (1.5 * 10 ** 24)) ** 0.22
    Y_gas = 1/X1 *1.33
    Vhalo = calc_v_NFW_halo(x)
    V_rot = V * np.sin(i_set) / np.sin(i_fit)
    # -Fotis- : Perhaps high absoluite values of  Vgas * np.abs(Vgas)?
    mean = np.sqrt(Vhalo ** 2 + (Y_disk * Vdisk ** 2 + Y_gas * Vgas * np.abs(Vgas) + Y_bul * Vbul ** 2) \
        * D_fit / D_quot)
    #- Fotis -: the  casting below perhaps is not needed...
    dist = np.array(V_rot) - mean
    sigma_squared = err_y_arr * np.sin(i_set) / np.sin(i_fit)
    chi_sq = (dist / sigma_squared) ** 2
    res = np.sum(chi_sq)
    return res

calc_halo = calc_v_NFW_halo
ndim = len(labels)
res = differential_evolution(chisq, optim_limits, maxiter=int(1e5), strategy='best1bin')
# Fotis: this print could be included somehow in the log file that
print(res)
Y_gas = res.x[4]
Y_bul = res.x[3]
Y_disk = res.x[2]
D_fit = res.x[1]
i_fit = res.x[0]
Vhalo = calc_halo(res.x)

V_th = np.sqrt(Vhalo ** 2 + (Y_disk * Vdisk ** 2 + Y_gas * Vgas * np.abs(Vgas) + Y_bul * Vbul ** 2) * (D_fit / D_quot))
####### Plot the results of the initial optimizer, this could be a function #####
plt.plot(x, V_th, 'red')
# -fotis-: The 'err_y' does not get corrected according to inclination...
plt.errorbar(x, y * np.sin(i_fit) / np.sin(i_set), yerr=np.array(err_y))
plt.ylabel('$V(km/s)$')
plt.xlabel('$R(kpc)$')
plt.savefig(fname + '_fit_results-beforeMCMC_new.pdf')

def plot_result(x, y, yerrs, sampler):
    b = 0.3
    nwalkers, states, nparams = sampler.chain.shape
    pars_mean = []
    burn_in = int(sampler.chain.shape[1] * b)
    for i in range(nparams):
        mcmc = np.percentile(sampler.get_chain(discard=burn_in, thin=1, flat=True)[:, i], [15.87, 50, 84.13])
        pars_mean.append(mcmc[1])
        q = np.diff(mcmc)
    Y_gas = pars_mean[4]
    Y_bul = pars_mean[3]
    Y_disk = pars_mean[2]
    D_fit = pars_mean[1]
    i_fit = pars_mean[0]
    Vhalo = calc_halo(pars_mean)
    V_th = np.sqrt(Vhalo ** 2 + (Y_disk * Vdisk ** 2 + Y_gas * Vgas * np.abs(Vgas) + Y_bul * Vbul ** 2) * (D_fit / D_quot))
    ################################################
    plt.plot(x, V_th, 'red')
    plt.errorbar(x, y * np.sin(i_fit) / np.sin(i_set), yerr=yerrs)
    plt.ylabel('$V(km/s)$')
    plt.xlabel('$R(kpc)$')
    plt.savefig(fname + '_fit_results-afterMCMC.pdf')

############ Make the initial positions for the walkers in a generic way ##################
res_ODR = (y - V_th)
rms_ODR = np.sqrt(np.mean(res_ODR ** 2., dtype=np.float64))  # observed scatter
s_ODR = np.sqrt(rms_ODR ** 2. - np.mean(err_y) ** 2.)  # estimate of intrinsic scatter

# make the list of initial positions for the emcee
initial = []
for i in range(nwalkers):
    # FOTIS:this should be checked again...
    errs = []
    means = []
    for j in range(len(labels)):
        means.append(res.x[j])
        errs.append(res.x[j] * 0.01)
    p_i = np.random.normal(means, errs)
    initial.append(p_i)
# print(initial)

if __name__ == '__main__':
    ### Run the emcee
    sampler, multi_time = ut.run_emcee(max_iters, initial, lnprob_withYs)   
    samples = sampler.chain[:, int(0.3 * max_iters):, :].reshape((-1, ndim))
    n = data_length - ndim
    multi_time = 0
    ## make logs and save the chains ##
    ut.log_maker(sampler, n, multi_time, fname, labels, sxolia=text, b=0.3, fname_auto='')
    ## Plot the emcee result ###
    ax = plt.figure()
    bs = 40
    ax = corner.corner(samples, plot_contours=True, fill_contours=True, plot_datapoints=False, \
                       show_titles=True, bins=bs, levels=[0., 0.6827, 0.9545, 0.9973], labels=labels, title_fmt=".4f")
    ax.savefig(fname + '.pdf')
    plt.close()

    plot_result(x_arr, y_arr, err_y, sampler)
