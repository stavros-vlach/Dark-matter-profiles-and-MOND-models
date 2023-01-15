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
import utils as ut
import os
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
matplotlib.use('TkAgg')


# os.chdir('C:/Users/User\PycharmProjects\pythonProject\Rotmod_LTG')
# name = input("galaxy name = ")
name = "UGC01281"
name = "Rotmod_LTG/" +name + "_rotmod.dat"
R, V, Verr, Vgas, Vbul = np.loadtxt(name, unpack=True, usecols=(0, 1, 2, 3, 5))
D = open(name, "r")
D = (D.readline().split('=')[1])
D = D.split()
D = float(D[0])

def distance():
    Rd = (30 / 206.265) * D
    return Rd

f = scipy.interpolate.interp1d(R, Vgas)

xnew = R
ynew = f(xnew)

g = scipy.interpolate.interp1d(R, Vbul)
ybul = g(R)
x = np.array(R)
y = np.array(V)
err_x = x*0
err_y = np.array(Verr)

Rd = distance()

G = 4.3*10**(-6)
z = []
data_length = len(R)
for i in range(data_length):
    z.append(1.6 * R[i] / 3.2 * Rd)
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
K1 = np.array(K1)
K0 = np.array(K0)

# os.chdir(r"C:/Users/User\PycharmProjects\pythonProject\kwdikas")

# profile = input("profile = ")
profile = "Burkert"


if profile == "Burkert":
    def func(beta, x):

        y = np.sqrt((0.5 * G * beta[0] * (3.2 * x ** 2)*(I0 * K0-I1 * K1)/Rd) ** 2 + (6.4 * G * (beta[1] * (beta[2] ** 3)) / x *
              (np.log(1 + x / beta[2]) - np.arctan(R / beta[2]) + (1/2) * np.log(1 + (x / beta[2])**2))) ** 2 + ynew ** 2 + ybul ** 2)
        y = np.array(y)
        return y
elif profile == "NFW":
    def func(beta, x):

        y = np.sqrt((0.5 * G * beta[0] * (3.2 * x ** 2)*(I0 * K0-I1 * K1)/Rd) ** 2 +
                    ((4 * np.pi * G * beta[1] * (beta[2] ** 3)) / x_arr * ((np.log(1 + x_arr / beta[2])) - ((x_arr / beta[2]) / (x_arr / beta[2]) + 1))) ** 2)
        y = np.array(y)
        return y
def Burkert(rho0, r0):
    burkerthalo = np.sqrt((6.4 * G * (rho0 * (r0 ** 3)) / x_arr) * (np.log(1 + x_arr / r0) - np.arctan(x_arr / r0) +
                                                                    (1/2) * np.log(1 + (x_arr / r0)**2)))
    return burkerthalo

def NFW(rho0, rs):
    NFWhalo = np.sqrt(((4 * np.pi * G * rho0 * (rs ** 3)) / x_arr) * ((np.log(1 + x_arr / rs)) - ((x_arr / rs) / ((x_arr / rs) + 1))))
    return NFWhalo

x_arr = np.array(R)
y_arr = np.array(V)
err_y_arr = np.array(Verr)
np.savetxt("test.dat", [0, 0, 0], fmt='%1.3f', newline=" ")

with open("test.dat", 'w'):
    pass

def lnprob_orthogonal(x):
    logMdisk = x[0]
    if profile == "Burkert":
        logrho0, r0, sigma = x[1], x[2], x[3]
        rho0 = 10 ** logrho0

        if not(0 < logMdisk) or not(0 < r0) or not(0 < rho0) or not(1 < sigma < 100):
            return -np.inf

        Vhalo = Burkert(rho0, r0)

    elif profile == "NFW":
        logrho0, rs, sigma = x[1], x[2], x[3]
        rho0 = 10 ** logrho0
        if not(0 < rho0) or not(0 < rs) or not(1 < sigma < 100):
            return -np.inf
        Vhalo = NFW(rho0, rs)

    Vdisk = np.sqrt(0.5 * G * (10 ** logMdisk) * ((3.2 * (x_arr / (3.2 * Rd))) ** 2) * (I0 * K0 - I1 * K1) / Rd)
    V_th = np.sqrt(Vhalo ** 2 + Vdisk ** 2 + ynew ** 2)

    dist = np.array(y_arr) - V_th
    error = np.sqrt(err_y_arr**2 + sigma**2)
    chi_sq = dist * dist / error**2

    L = np.exp(-chi_sq/2.) / (np.sqrt(2.*np.pi) * error)
    if profile == "Burkert":
        with open("test.dat", "ab") as f:
            np.savetxt(f, np.array([logMdisk, logrho0, r0]), fmt='%1.3f', newline=" ")
            f.write(b"\n")
    elif profile == "NFW":
        with open("test.dat", "ab") as f:
            np.savetxt(f, np.array([logMdisk, logrho0, rs]), fmt='%1.3f', newline=" ")
            f.write(b"\n")
    if np.min(L) < 1.e-300:
        return -1.e300

    return np.sum(np.log(L))


def lnprob_withYs(x):
    '''
    The likelihood is constructed according to 1803.00022
    Also check https://arxiv.org/pdf/1809.06875.pdf
    --> many different DM profiles
    '''
    logMdisk = x[0]
    # if profile == "Burkert":
    logrho0, r0,Y_disk = x[1], x[2], x[3]
    rho0 = 10 ** logrho0

    if not(0 < logMdisk) or not(0 < r0) or not(0 < rho0)\
    or not(0.001 < Y_disk < 1.2):
        return -np.inf

    Vhalo = Burkert(rho0, r0)

    # elif profile == "NFW":
    #     logrho0, rs, sigma = x[1], x[2], x[3]
    #     rho0 = 10 ** logrho0
    #     if not(0 < rho0) or not(0 < rs) or not(1 < sigma < 100):
    #         return -np.inf
    #     Vhalo = NFW(rho0, rs)

    Vdisk = np.sqrt(0.5 * G * (10 ** logMdisk) * ((3.2 * (x_arr / (3.2 * Rd))) ** 2) * (I0 * K0 - I1 * K1) / Rd)
    V_th = np.sqrt(Vhalo ** 2 + Y_disk*Vdisk ** 2 + ynew ** 2)

    dist = np.array(y_arr) - V_th
    sigma = 0
    error = np.sqrt(err_y_arr**2 + sigma**2)
    chi_sq = dist * dist / error**2

    L = np.exp(-chi_sq/2.) / (np.sqrt(2.*np.pi) * error)
    # if profile == "Burkert":
    #     with open("test.dat", "ab") as f:
    #         np.savetxt(f, np.array([logMdisk, logrho0, r0]), fmt='%1.3f', newline=" ")
    #         f.write(b"\n")
    # elif profile == "NFW":
    #     with open("test.dat", "ab") as f:
    #         np.savetxt(f, np.array([logMdisk, logrho0, rs]), fmt='%1.3f', ne
    #         f.write(b"\n")
    if np.min(L) < 1.e-300:
        return -1.e300

    return np.sum(np.log(L))

def chisq(x):
    '''
    This is the x^2 function - the object function for minimization

    '''
    logMdisk = x[0]
    # if profile == "Burkert":
    logrho0, r0, Y_disk = x[1], x[2],  x[3]
    rho0 = 10 ** logrho0
    Vhalo = Burkert(rho0, r0)
    # elif profile == "NFW":
    #     logrho0, rs = x[1], x[2]
    #     rho0 = 10 ** logrho0
    #     Vhalo = NFW(rho0, rs)

    Vdisk = np.sqrt(0.5 * G * (10**logMdisk) * ((3.2 * (x_arr / (3.2 * Rd))) ** 2) * (I0 * K0 - I1 * K1) / Rd)

    mean = np.sqrt(Vhalo**2 + Y_disk*Vdisk ** 2 + ynew ** 2)

    dist = np.array(y_arr) - mean
    sigma_squared = np.power(err_y_arr,2)
    chi_sq = dist * dist / sigma_squared
    res = np.sum(chi_sq)
    return res

res = differential_evolution(chisq, [(5,15),(5,10),(1e-3,100),(0.01,3)], args=(), strategy='best1bin')
print(res)
pars_mean = res.x
if profile == "Burkert":
    logMdisk, logrho0, r0,  = pars_mean[0], pars_mean[1], pars_mean[2]
    rho0 = 10 ** logrho0
    Vhalo = Burkert(rho0, r0)
elif profile == "NFW":
    logMdisk, logrho0, rs = pars_mean[0], pars_mean[1], pars_mean[2]
    rho0 = 10 ** logrho0
    Vhalo = NFW(rho0, rs)
Y_disk = pars_mean[3]
Vdisk = np.sqrt(0.5 * G * (10**logMdisk) * ((3.2 * (x_arr / (3.2 * Rd))) ** 2) * (I0 * K0 - I1 * K1) / Rd)

V_th = np.sqrt(Vhalo**2 + Y_disk*Vdisk ** 2 + ynew ** 2)
################################################

plt.plot(x, V_th, 'red')
plt.errorbar(x,y,yerr=np.array(err_y))
plt.ylabel('$V(km/s)$')
plt.xlabel('$R(kpc)$')
plt.savefig('fit_results-beforeMCMC.pdf')

beta = res.x
x = np.array(x)
y = np.array(y)
err_x = np.array(err_x)
err_y = np.array(err_y)

a_ODR = beta[0]
b_ODR = beta[1]
c_ODR = beta[2]
d_ODR = beta[3]
res_ODR = (y - V_th)

rms_ODR = np.sqrt(np.mean(res_ODR**2., dtype=np.float64)) #observed scatter
s_ODR = np.sqrt(rms_ODR**2. - np.mean(err_y)**2.) #estimate of intrinsic scatter

nwalkers = 600

ndim = 4

max_iters = 3000

p0 = []
for i in range(nwalkers):
    pi = [np.random.normal(a_ODR, 0.01), np.random.normal(b_ODR, 0.01), np.random.normal(c_ODR, 0.01),\
     # np.random.normal(rms_ODR, rms_ODR/10.),\
    np.random.normal(0.5,0.001)]
    p0.append(pi)


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_withYs)

for sample in sampler.sample(p0, iterations=max_iters, progress=True):

    if sampler.iteration % 100:
        continue

samples = sampler.chain[:, int(0.3*max_iters):, :].reshape((-1, ndim))
if profile == "Burkert":
    labels = ["$log(M_{disk})$", "$log(\\rho_{0})$", "$r_{0}$", "$\\sigma$", "$Y_{disk}$"]
elif profile == "NFW":
    labels = ["$log(M_{disk})$", "$log(\\rho_{0})$", "$r_{s}$", "$\\sigma$"]
fname = 'test'
ax = plt.figure()
bs = 40
ax = corner.corner(samples,plot_contours = True, fill_contours = True,plot_datapoints = False, \
    show_titles = True, bins = bs, levels = [0.,0.6827,0.9545,0.9973],labels=labels,title_fmt=".4f")
ax.savefig(fname +'.pdf')
plt.close()


text = ''
n = data_length - ndim
multi_time = 0
ut.log_maker(sampler, n, multi_time, fname, labels, sxolia=text, b=0.3, fname_auto='')

def plot_result(x,y,yerrs,sampler):
    b = 0.3
    nwalkers, states, nparams = sampler.chain.shape
    pars_mean = []
    burn_in = int(sampler.chain.shape[1] * b)
    for i in range(nparams):
        mcmc = np.percentile(sampler.get_chain(discard=burn_in, thin=1, flat=True)[:, i], [15.87, 50, 84.13])
        pars_mean.append(mcmc[1])
        q = np.diff(mcmc)

    if profile == "Burkert":

        logMdisk, logrho0, r0, Y_disk = pars_mean[0], pars_mean[1], pars_mean[2], pars_mean[3]
        rho0 = 10 ** logrho0
        Vhalo = Burkert(rho0, r0)
    elif profile == "NFW":
        logMdisk, logrho0, rs, Y_disk = pars_mean[0], pars_mean[1], pars_mean[2], pars_mean[3]
        rho0 = 10 ** logrho0
        Vhalo = NFW(rho0, rs)
    Rd = distance()
    Vdisk = np.sqrt(0.5 * G * (10**logMdisk) * ((3.2 * (x_arr / (3.2 * Rd))) ** 2) * (I0 * K0 - I1 * K1) / Rd)
    V_th = np.sqrt(Vhalo**2 + Y_disk*Vdisk ** 2 + ynew ** 2)
    ################################################
    plt.plot(x,V_th,'red')
    plt.errorbar(x,y,yerr=yerrs)
    plt.ylabel('$V(km/s)$')
    plt.xlabel('$R(kpc)$')
    plt.savefig('fit_results-afterMCMC.pdf')
    #Fotis: A plot right below with the residuals is also needed
plot_result(x_arr,y_arr,err_y,sampler)

