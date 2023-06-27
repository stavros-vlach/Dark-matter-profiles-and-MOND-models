import sys
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.odr import ODR, Model, Data, RealData
from scipy.optimize import differential_evolution
from scipy import stats
from multiprocessing import Pool, cpu_count
import time
import scipy
import os
from scipy.special import gammaincc, gamma
import utils as ut

matplotlib.use('TkAgg')
########## 15/01/2023 ###########
# Things TODO:
# (1) -Σταυρος- The code should have different lnprob for each Halo profile OR MOND model
# (2) -Φωτης, Σταυρος- We have to enrich τα  δεδομενα μας δηλ εκτος του σπαρκ να βάλουμε ΚΑΙ άλλα δεδομενα
#        * πρεπει να καταλάβουμε τις υποθεσεις κάθε σετ δεδομενων
# (3) -Σταυρος- Να ξεχωρισουμε ολους τους γαλαξιες που έχουν Ν > Ν_ελχ, αρχικα Ν_ελαχιστο = 10
# (4) --Σταυρος, Φωτης-- Να μαζεψουμε τη βιβλιογραφια με τον εξης τροπο: Ταδε δεινοπουλος, 2022, χρησιμοποιησε την ταδε πιθανοφανεια
#        με τις χ υποθεσεις. (Να γραφτει στα αγγλικά, σε template. )
# Τελικος σκοπος:: (a) Συγκριση όλων των προφιλ ΚΑΙ των MOND σε μια κοινη βαση -- κοινο συνολο δεδομενων, ιδια συμπεριληψη
# εγγενων ελευθερων παραμετρων πχ Υ_disk
# (b) συλλογη περιεργων γαλαξιων ,οπου ολα τα μοντέλα δεν τα καταφερνουν καλα
# (c) συγκριση ολων των παραμετροποιησεων
##########################################################################################
# We need to beutify and reorganize the code so to allow for easier debugging. Currently there are parts of the
# code that are defined multiple times.
# TODO  - 1 May 2023 -
# [1] The first part, where we load the data ought to be a function that takes galaxy name and returns everything needed
# [2] Each 'if' case regarding a aprticular profile should be define all needed variables and function to be given at the mcmc
# sampler. For example, 'loglikelihood' function, 'labels' variable (needed for the  plots) and a new function named e.g. 'make_model'
# which will contain part of the 'plot_result' function.
# [3] We need to test each profile in detail. We compare results from this code with published ones.
# [4] There are inline comments for additions/modifications.
# [5] code style -- use this tool to check it: https://pypi.org/project/pylint/
##########################################################################################
# GALAXY_NAME = input("galaxy name = ")
#GALAXY_NAME = "UGC02953"
#GALAXY_NAME = 'IC2574'
GALAXY_NAME = "ESO563-G021"
PROFILE = "Burkert"
#os.chdir(r"C:/Users/User\PycharmProjects\pythonProject\kwdikas")
name = "Rotmod_LTG/" + GALAXY_NAME + "_rotmod.dat"
fname = f'newPar_fit_{GALAXY_NAME}__{PROFILE}'
R, V, Verr, Vgas, Vbul = np.loadtxt(name, unpack=True, usecols=(0, 1, 2, 3, 5))
D = open(name, "r")
D = (D.readline().split('=')[1])
D = D.split()
D = float(D[0])

# ---- Fotis: this part needs to be organized as a function ----- #
f = scipy.interpolate.interp1d(R, Vgas)
ynew = f(R)
g = scipy.interpolate.interp1d(R, Vbul)
ybul = g(R)
x = np.array(R)
y = np.array(V)
err_x = x * 0
err_y = np.array(Verr)

# Fotis: H0 in km/kpc * s^-1
HO = 67

def distance():
    Rd = (30 / 206.265) * D
    return Rd


Rd = distance()

G = 4.3 * 10 ** (-6)
z = []
data_length = len(R)
for i in range(data_length):
    z.append(1.6 * R[i] / (3.2 * Rd))
K0 = [scipy.special.kv(0, z[0])]
I0 = [scipy.special.iv(0, z[0])]
K1 = [scipy.special.kv(1, z[0])]
I1 = [scipy.special.iv(1, z[0])]
for i in range(1, data_length):
    K0.append(scipy.special.kv(0, z[i]))
    I0.append(scipy.special.iv(0, z[i]))
    K1.append(scipy.special.kv(1, z[i]))
    I1.append(scipy.special.iv(1, z[i]))
I0 = np.array(I0)
I1 = np.array(I1)
K0 = np.array(K0)
K1 = np.array(K1)

x_arr = np.array(R)
y_arr = np.array(V)
err_y_arr = np.array(Verr)
def x_par(x_arr, r_s):
    r = x_arr / r_s
    return r

# ------------------------------------------------------------------ #

def V_d(logMdisk):
    diskV = np.sqrt(0.5 * G * (10 ** logMdisk) * ((3.2 * (x_arr / (3.2 * Rd))) ** 2) * (I0 * K0 - I1 * K1) / Rd)
    return diskV


if PROFILE == "Burkert":
    ########################## Definition of profile specific constants ###########################
    labels = ["$log(M_{disk})$", "$Y_{disk}$", "$log(\rho_s)$", "r_{s}"]
    means = np.array([10, 0.5, 200, 500, 250])
    optim_limits = [(5, 15), (0.0, 3.0), (-7, 1), (2.0, 500),]
    text = f'Fit of {PROFILE} profile on {GALAXY_NAME} rot curve data from SPARC'
    nwalkers = 2000
    max_iters = 2500


    ###############################################################################################
    def func(beta, x):
        r = x_par(x_arr, beta[2])
        y = np.sqrt(beta[0] / r) * np.sqrt((1/2 * np.log(1+r**2) + np.log(1+r) - np.arctan(r)) /
                                    (1/2 * np.log(1+beta[0]**2) + np.log(1+beta[0]) - np.arctan(beta[0]))) * beta[1]
        y = np.array(y)
        return y


    def calc_v_burkert_halo(Y):
        r_s, logrho_s  = Y
        # All models have r_s, rho_s as free parameters (apart from i etc)
        # r = x_par(x_arr, r_s)
        # Fotis: function calls are slow, so we avoide them if its free - no extra code ugliness 
        r = x_arr / r_s
        M = 2*np.pi*r_s**3 *(0.5*np.log(1+r**2) + np.log(r+1) - np.arctan(r))
        # Fotis: The following defs are generic for all models
        rho = np.exp(logrho_s)/(1+(r/r_s)**2)
        r200 = (3*M/(4*np.pi*rho))**(1/3)
        C_200 = r200 / r_s
        V_200 = 10 * C_200 * r_s * HO
        

        burkert_v = np.sqrt(C_200 / r) * np.sqrt((1/2 * np.log(1+r**2) + np.log(1+r) - np.arctan(r)) /
                                    (1/2 * np.log(1+C_200**2) + np.log(1+C_200) - np.arctan(C_200))) * V_200
        return burkert_v


    '''def lnprob_orthogonal(x):
        D, i, V_200, C_200, r_s = x
        if not (0 < D) or not (0 < i) or not (0 < V_200) or not (1 < sigma < 100):
            return -np.inf
        print(x)    
        Vhalo = np.sqrt((6.4 * G * (rho0 * (r0 ** 3)) / x_arr) * (np.log(1 + x_arr / r0) - np.arctan(x_arr / r0) +
                                                                  (1 / 2) * np.log(1 + (x_arr / r0) ** 2)))
        Vdisk = V_d(logMdisk)
        V_th = np.sqrt(Vhalo ** 2 + Vdisk ** 2 + ynew ** 2)

        dist = y_arr - V_th
        error = np.sqrt(err_y_arr ** 2 + sigma ** 2)
        chi_sq = dist * dist / error ** 2
        L = np.exp(-chi_sq / 2.) / (np.sqrt(2. * np.pi) * error)

        if np.min(L) < 1.e-300:
            return -1.e300

        return np.sum(np.log(L))'''


    def lnprob_withYs(x):
        logMdisk, Y_disk, r_s, logrho_s = x

        if not (0 < logMdisk < 30) or not (0 < r_s < 800) or not(-7 < logrho_s < 1)\
                or not (0.01 < Y_disk < 0.6):
            return -np.inf

        Vhalo = calc_v_burkert_halo([r_s,logrho_s])

        Vdisk = V_d(logMdisk)
        V_th = np.sqrt(Vhalo ** 2 + Y_disk * Vdisk ** 2 + ynew ** 2)

        dist = y_arr - V_th
        sigma = 0
        error = np.sqrt(err_y_arr ** 2 + sigma ** 2)
        chi_sq = dist * dist / (error ** 2)
        L = np.exp(-chi_sq / 2.) / (np.sqrt(2. * np.pi) * error)

        # with open("test.dat", "ab") as f:
        #     np.savetxt(f, np.array([logMdisk, logrho0, r0]), fmt='%1.3f', newline=" ")
        #     f.write(b"\n")
        if np.min(L) < 1.e-300:
            return -1.e300

        return np.sum(np.log(L))


    # print(lnprob_withYs([11.02564559,  7.84779257,  8.47000321,  1.03669173]))
    # print(stop)
    def chisq(x):
        '''
        This is the x^2 function - the object function for minimization

        '''
        logMdisk, Y_disk, r_s, logrho_s = x

        Vhalo = calc_v_burkert_halo([r_s,logrho_s])

        Vdisk = V_d(logMdisk)

        mean = np.sqrt(Vhalo ** 2 + Y_disk * Vdisk ** 2 + ynew ** 2)

        dist = np.array(y_arr) - mean
        sigma_squared = np.power(err_y_arr, 2)
        chi_sq = dist * dist / sigma_squared
        res = np.sum(chi_sq)
        return res


    # logMdisk, logrho0, r0, = pars_mean[0], pars_mean[1], pars_mean[2]
    # rho0 = 10 ** logrho0
    # Vhalo = np.sqrt((6.4 * G * (rho0 * (r0 ** 3)) / x_arr) * (np.log(1 + x_arr / r0) - np.arctan(x_arr / r0) +
    #                                                                     (1 / 2) * np.log(1 + (x_arr / r0) ** 2)))

    # Vdisk = np.sqrt(0.5 * G * (10 ** logMdisk) * ((3.2 * (x_arr / (3.2 * Rd))) ** 2) * (I0 * K0 - I1 * K1) / Rd)
    calc_halo = calc_v_burkert_halo

############ General stuff for all profiles ###########
ndim = len(labels)
res = differential_evolution(chisq, optim_limits, maxiter=int(1e5), strategy='best1bin')
# Fotis: this print could be included somehow in the log file that
print(res)
### FOTIS: VERY DANGEROUS!!-assuming that the last parameter will be Y_disk for all profiles.
Y_disk = res.x[1]
Vhalo = calc_halo(res.x)
### FOTIS: AGAIN VERY DANGEROUS!!-assuming that the first parameter will be logMdisk for all profiles.
logMdisk = res.x[0]
Vdisk = V_d(logMdisk)
V_th = np.sqrt(Vhalo ** 2 + Y_disk * Vdisk ** 2 + ynew ** 2)
####### Plot the results of the initial optimizer, this could be a function #####
plt.plot(x, V_th, 'red')
plt.errorbar(x, y, yerr=np.array(err_y))
plt.ylabel('$V(km/s)$')
plt.xlabel('$R(kpc)$')
plt.savefig(fname + '_fit_results-beforeMCMC_new.pdf')


######################################################
def plot_result(x, y, yerrs, sampler):
    b = 0.3
    nwalkers, states, nparams = sampler.chain.shape
    pars_mean = []
    burn_in = int(sampler.chain.shape[1] * b)
    for i in range(nparams):
        mcmc = np.percentile(sampler.get_chain(discard=burn_in, thin=1, flat=True)[:, i], [15.87, 50, 84.13])
        pars_mean.append(mcmc[1])
        q = np.diff(mcmc)
    logMdisk = pars_mean[0]
    Y_disk = pars_mean[1]
    Vhalo = calc_halo(pars_mean)
    # Vhalo = np.sqrt((6.4 * G * (rho0 * (r0 ** 3)) / x_arr) * (np.log(1 + x_arr / r0) - np.arctan(x_arr / r0) +
    #                                                                     (1 / 2) * np.log(1 + (x_arr / r0) ** 2)))

    # Vdisk = np.sqrt(0.5 * G * (10 ** logMdisk) * ((3.2 * (x_arr / (3.2 * Rd))) ** 2) * (I0 * K0 - I1 * K1) / Rd)
    Rd = distance()
    Vdisk = np.sqrt(0.5 * G * (10 ** logMdisk) * ((3.2 * (x_arr / (3.2 * Rd))) ** 2) * (I0 * K0 - I1 * K1) / Rd)
    V_th = np.sqrt(Vhalo ** 2 + Y_disk * Vdisk ** 2 + ynew ** 2)
    ################################################
    plt.plot(x, V_th, 'red')
    plt.errorbar(x, y, yerr=yerrs)
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
        means.append(res.x[j]*0.5)
        if j == len(res.x) - 1:
            errs.append(0.01)
        else:
            errs.append(res.x[j] * 0.0001)
    p_i = np.random.normal(means, errs)
    initial.append(p_i)
print(initial)

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

################ Previous stuff #################

# print(res)
# pars_mean = res.x

# logMdisk, logrho0, Rc = pars_mean[0], pars_mean[1], pars_mean[2]
# rho0 = 10 ** logrho0
# Vhalo = Iso(rho0, Rc)
# Y_disk = pars_mean[3]
# Vdisk = np.sqrt(0.5 * G * (10 ** logMdisk) * ((3.2 * (x_arr / (3.2 * Rd))) ** 2) * (I0 * K0 - I1 * K1) / Rd)

# V_th = np.sqrt(Vhalo ** 2 + Y_disk * Vdisk ** 2 + ynew ** 2)

# plt.plot(x, V_th, 'red')
# plt.errorbar(x, y, yerr=np.array(err_y))
# plt.ylabel('$V(km/s)$')
# plt.xlabel('$R(kpc)$')
# plt.savefig('fit_results-beforeMCMC.pdf')

# beta = res.x
# x = np.array(x)
# y = np.array(y)
# err_x = np.array(err_x)
# err_y = np.array(err_y)

# a_ODR = beta[0]
# b_ODR = beta[1]
# c_ODR = beta[2]
# d_ODR = beta[3]

# res_ODR = (y - V_th)

# rms_ODR = np.sqrt(np.mean(res_ODR ** 2., dtype=np.float64))  # observed scatter
# s_ODR = np.sqrt(rms_ODR ** 2. - np.mean(err_y) ** 2.)  # estimate of intrinsic scatter

# nwalkers = 100

# ndim = 4

# max_iters = 500

# p0 = []
# for i in range(nwalkers):
#     pi = [np.random.normal(a_ODR, 0.01), np.random.normal(b_ODR, 0.01), np.random.normal(c_ODR, 0.01),
#           # np.random.normal(rms_ODR, rms_ODR/10.),\
#           np.random.normal(0.5, 0.001)]
#     p0.append(pi)

# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_withYs)

# for sample in sampler.sample(p0, iterations=max_iters, progress=True):

#     if sampler.iteration % 100:
#         continue

# samples = sampler.chain[:, int(0.3 * max_iters):, :].reshape((-1, ndim))

# labels = ["$log(M_{disk})$", "$log(\\rho_{0})$", "$R_{s}$", "$r_{s}$", "$\\alpha$", "$\\sigma$"]

# fname = 'test'
# ax = plt.figure()
# bs = 40
# ax = corner.corner(samples, plot_contours=True, fill_contours=True, plot_datapoints=False, \
#                    show_titles=True, bins=bs, levels=[0., 0.6827, 0.9545, 0.9973], labels=labels, title_fmt=".4f")
# ax.savefig(fname + '.pdf')
# plt.close()

# text = ''
# n = data_length - ndim
# multi_time = 0
# ut.log_maker(sampler, n, multi_time, fname, labels, sxolia=text, b=0.3, fname_auto='')


# def plot_result(x, y, yerrs, sampler):
#     b = 0.3
#     nwalkers, states, nparams = sampler.chain.shape
#     pars_mean = []
#     burn_in = int(sampler.chain.shape[1] * b)
#     for i in range(nparams):
#         mcmc = np.percentile(sampler.get_chain(discard=burn_in, thin=1, flat=True)[:, i], [15.87, 50, 84.13])
#         pars_mean.append(mcmc[1])
#         q = np.diff(mcmc)

#     logMdisk, logrho0, Rc, Y_disk = pars_mean[0], pars_mean[1], pars_mean[2], pars_mean[3]
#     rho0 = 10 ** logrho0
#     Vhalo = Iso(rho0, Rc)

#     Rd = distance()
#     Vdisk = np.sqrt(0.5 * G * (10 ** logMdisk) * ((3.2 * (x_arr / (3.2 * Rd))) ** 2) * (I0 * K0 - I1 * K1) / Rd)
#     V_th = np.sqrt(Vhalo ** 2 + Y_disk * Vdisk ** 2 + ynew ** 2)
#     ################################################
#     plt.plot(x, V_th, 'red')
#     plt.errorbar(x, y, yerr=yerrs)
#     plt.ylabel('$V(km/s)$')
#     plt.xlabel('$R(kpc)$')
#     plt.savefig('fit_results-afterMCMC.pdf')
# plot_result(x_arr, y_arr, err_y, sampler)