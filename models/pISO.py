import random
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from scipy.optimize import differential_evolution
import scipy
from scipy.optimize import minimize
from tqdm import tqdm
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
# GALAXY_NAME = "UGC02953"
# GALAXY_NAME = 'NGC0300'
# GALAXY_NAME = "ESO563-G021"
GALAXY_NAME = "UGC08490"
name = "/media/stavros/aaf194dd-624d-4ece-8e56-395e27a93185/Pythonproject/Rotmod_LTG/" + GALAXY_NAME + "_rotmod.dat"

R, V, Verr, Vgas, Vbul, Vdisk = np.loadtxt(name, unpack=True, usecols=(0, 1, 2, 3, 4, 5))
data_length = len(R)

g_obs = (V ** 2) / R
# The full dataset of http://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt,
# with minor changes for parsing
data = pd.read_csv('data.txt', delimiter=';', skiprows=98, names=labels)

data = data[data['Galaxy'] == GALAXY_NAME]
i_set = data["Inclination (deg)"].to_numpy()[0] * np.pi / 180
err_i_set = data["Mean Inc error (deg)"].to_numpy()[0] * np.pi / 180
D_quot = data["Distance (Mpc)"].to_numpy()[0]
err_D_quot = data["Mean D error (Mpc)"].to_numpy()[0]
L_tot = data["Total Luminosity at [3.6](10+9solLum)"].to_numpy()[0]

Lb = pd.read_csv('Lbul.txt', delimiter = '\s+', skiprows=7, names=['Galaxy', 'Lbul (10^9 Lsun)'])
Lb = Lb[Lb['Galaxy'] == GALAXY_NAME]
L_bul = Lb['Lbul (10^9 Lsun)'].to_numpy()[0]

#constants
G = 4.3e-6

#Model as in https://iopscience.iop.org/article/10.3847/2041-8213/ac1bb7

def pISO_halo(rho0, Rc):
    v_halo_sq = 4 * np.pi * G * rho0 * Rc ** 2 * ( 1 - (Rc / r) * np.arctan(r / Rc))
    return v_halo_sq
    
def total_velocity_squared(Y_star, rho0, Rc):
    v_star_sq = Y_star * v_disk_sq + 1.4 * Y_star * v_bulge_sq
    v_halo_sq = pISO_halo(rho0, Rc)
    v_total_sq = v_star_sq + v_gas_sq + v_halo_sq
    return v_total_sq
    
def log_prior(Y_star, rho0, Rs):
    """Define the log prior."""
    if 0 < Y_star < 1 and 0 < rho0 < 1e9 and 0 < Rs < 1e4:
        return 0.0
    return -np.inf

def log_likelihood(theta):
    """Define the log likelihood."""
    Y_star, rho0, Rc = theta
    v_model_sq = total_velocity_squared(Y_star, rho0, Rc)
    chi_sq = np.sum(((v_obs**2 - v_model_sq) / v_err**2) ** 2)
    chi_sq += ((Y_star - Y_hat_star) / sigma_Y_star) ** 2
    return -0.5 * chi_sq

def log_probability(theta):
    """Define the log probability function for emcee."""
    Y_star, rho0, Rc = theta
    lp = log_prior(Y_star, rho0, Rc)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def chi_square(theta):
    """Calculate the chi-square value for the model fit."""
    Y_star, rho0, Rc = theta
    v_model_sq = total_velocity_squared(Y_star, rho0, Rc)
    chi_sq = np.sum(((v_obs**2 - v_model_sq) / v_err**2) ** 2)
    chi_sq += ((Y_star - Y_hat_star) / sigma_Y_star) ** 2
    return chi_sq / (len(r) - len(theta))  # Adjusted for degrees of freedom
    
# Observed data and errors
r = np.array(R)
v_obs = np.array(V)
v_err = np.array(Verr)
v_disk_sq = np.array(Vdisk)**2
v_bulge_sq = np.array(Vbul)**2
v_gas_sq = np.array(Vgas)**2
Y_hat_star = 0.5
sigma_Y_star = 0.25 * Y_hat_star

bounds = [(0, 1), (1e3, 1e10), (1e-1, 1e2)]

result = differential_evolution(chi_square, bounds)
best_fit_params = result.x
print(result)

Y_star_best, rho0_best, Rc_best = best_fit_params
v_model_sq_best = total_velocity_squared(Y_star_best, rho0_best, Rc_best)
v_model_best = np.sqrt(v_model_sq_best)

plt.plot(r, v_model_best, label = 'Best-fit model before emcee', color = 'red')
plt.errorbar(r, v_obs, yerr=v_err, fmt='o', label='Observed data')
plt.ylabel('$V(km/s)$')
plt.xlabel('$R(kpc)$')
plt.legend()
plt.savefig('fit_results-beforeMCMC.pdf')
plt.close()

ndim = len(best_fit_params)
nwalkers = 100
nsteps = 1000

# Set up the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

# Initialize the walkers
pos = best_fit_params + 1e-2 * np.random.randn(nwalkers, ndim)

# Run the sampler
sampler.run_mcmc(pos, nsteps, progress=True)

# Extract the samples
samples = sampler.get_chain(discard=100, thin=10, flat=True)

fig = corner.corner(samples, labels=["$Y_\\star$", "$\\rho_0$", "$R_s$"], truths=[Y_hat_star, 0, 0])
fig.savefig("fit_corner.pdf")
plt.close()

Y_star_mcmc, rho0_mcmc, Rs_mcmc = np.percentile(samples, 50, axis=0)
Y_star_mcmc_err = np.percentile(samples[:, 0], [16, 84])
rho0_mcmc_err = np.percentile(samples[:, 1], [16, 84])
Rs_mcmc_err = np.percentile(samples[:, 2], [16, 84])

print("Best-fit parameters and their 1-sigma intervals from MCMC:")
print(f"Y_star: {Y_star_mcmc:.4f} (+{Y_star_mcmc_err[1]-Y_star_mcmc:.4f}, -{Y_star_mcmc-Y_star_mcmc_err[0]:.4f})")
print(f"rho0: {rho0_mcmc:.4e} (+{rho0_mcmc_err[1]-rho0_mcmc:.4e}, -{rho0_mcmc-rho0_mcmc_err[0]:.4e})")
print(f"Rs: {Rs_mcmc:.4f} (+{Rs_mcmc_err[1]-Rs_mcmc:.4f}, -{Rs_mcmc-Rs_mcmc_err[0]:.4f})")

best_fit_params = np.median(samples, axis=0)
Y_star_best, rho0_best, Rs_best = best_fit_params
v_model_sq_mcmc = total_velocity_squared(Y_star_best, rho0_best, Rs_best)
v_model_mcmc = np.sqrt(v_model_sq_best)

plt.errorbar(r, v_obs, yerr=v_err, fmt='o', label='Observed data')
plt.plot(r, v_model_mcmc, label='Best-fit model after emcee', color='red')
plt.xlabel('Radius (kpc)')
plt.ylabel('Velocity (km/s)')
plt.legend()
plt.savefig('fit_results-afterMCMC.pdf')
