import random
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from scipy.optimize import differential_evolution, minimize
import scipy
from tqdm import tqdm
from scipy.special import gamma, gammainc
import os

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

#constants
a0 = 3702.81 #from 1.2 * 10 ^ {-10} m / s^2 to km^2 / s^2 / kpc
Y_hat_star = 0.5
sigma_Y_star = 0.25 * Y_hat_star

#Model as in https://iopscience.iop.org/article/10.3847/2041-8213/ac1bb7

def v_newtonian_square(Y_star):
	v_star_sq = Y_star * v_disk_sq + 1.4 * Y_star * v_bulge_sq
	return v_star_sq + v_gas_sq

def newtonian_acceleration(Y_star):
	v_N_sq = v_newtonian_square(Y_star)
	a_N = v_N_sq / r
	return a_N
	
def mond_acceleration(Y_star, a_N):
	a = a_N * np.sqrt(0.5 + 0.5 * np.sqrt(1 + (2 * a0 / a_N)**2))
	return a
	
def mond_velocity_squared(Y_star):
	v_N_sq = v_newtonian_square(Y_star)
	a_N = newtonian_acceleration(Y_star)
	#a = mond_acceleration(r, v_disk_sq, v_bulge_sq, v_gas_sq, Y_star, a_N)
	v_MOND_sq = v_N_sq * np.sqrt(0.5 + 0.5 * np.sqrt(1 + (2 * a0 / a_N)**2))
	return v_MOND_sq
	
def log_prior(Y_star):
	"""Define the log prior."""
	if 0 < Y_star < 1:
		return 0.0
	return -np.inf

def log_likelihood(theta):
	"""Define the log likelihood."""
	Y_star = theta
	v_model_sq = mond_velocity_squared(Y_star)
	chi_sq = np.sum(((v_obs**2 - v_model_sq) / v_err**2) ** 2)
	chi_sq += ((Y_star - Y_hat_star) / sigma_Y_star) ** 2
	return -0.5 * chi_sq

def log_probability(theta):
	"""Define the log probability function for emcee."""
	Y_star = theta
	lp = log_prior(Y_star)
	if not np.isfinite(lp):
		return -np.inf
	return lp + log_likelihood(theta)
   
def chi_square(theta):
	Y_star = theta
	v_MOND_sq = mond_velocity_squared(Y_star)
	chi_sq = np.sum(((v_obs**2 - v_MOND_sq) / v_err**2)**2)
	chi_sq += ((Y_star - Y_hat_star) / sigma_Y_star) ** 2
	return chi_sq / (len(r) - len(theta))

#bounds for differential_evolution
bounds = [(0, 2)]

#parameters for emcee
ndim = 1
nwalkers = 100
nsteps = 1000

path = "/media/stavros/aaf194dd-624d-4ece-8e56-395e27a93185/Pythonproject/Rotmod_LTG/"
for GALAXY_NAME in os.listdir(path):
	name = path + GALAXY_NAME
	R, V, Verr, Vgas, Vbul, Vdisk = np.loadtxt(name, unpack=True, usecols=(0, 1, 2, 3, 4, 5))
	data_length = len(R)
	
	GALAXY_NAME = GALAXY_NAME[:-len("_rotmod.dat")]
	
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

	r = np.array(R)
	v_obs = np.array(V)
	v_err = np.array(Verr)
	v_disk_sq = np.array(Vdisk)**2
	v_bulge_sq = np.array(Vbul)**2
	v_gas_sq = np.array(Vgas)**2
	
	result = differential_evolution(chi_square, bounds, args=())
	Y_star_best = result.x[0]
	print(result)
	v_MOND_sq_best = mond_velocity_squared(Y_star_best)
	v_MOND_best = np.sqrt(v_MOND_sq_best)

	plt.plot(r, v_MOND_best, label = 'Best-fit model before emcee', color = 'red')
	plt.errorbar(r, v_obs, yerr=v_err, fmt='o', label='Observed data')
	plt.ylabel('$V(km/s)$')
	plt.xlabel('$R(kpc)$')
	plt.legend()
	plt.savefig("/media/stavros/aaf194dd-624d-4ece-8e56-395e27a93185/Pythonproject/de/" + GALAXY_NAME + ' fit_results-beforeMCMC.pdf')
	plt.close()

	# Set up the sampler
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=())

	# Initialize the walkers
	pos = Y_star_best + 1e-2 * np.random.randn(nwalkers, ndim)

	# Run the sampler
	sampler.run_mcmc(pos, nsteps, progress=True)

	# Extract the samples
	samples = sampler.get_chain(discard=100, thin=10, flat=True)

	# Create a corner plot
	fig = corner.corner(samples, labels=["$Y_\\star$"], truths=[Y_hat_star])
	fig.savefig("/media/stavros/aaf194dd-624d-4ece-8e56-395e27a93185/Pythonproject/de/" + GALAXY_NAME + "fit_corner.pdf")
	plt.close()

	Y_star_mcmc = np.percentile(samples, 50, axis=0)
	Y_star_mcmc_err = np.percentile(samples[:, 0], [16, 84])

	print("Best-fit parameters and their 1-sigma intervals from MCMC:")
	print("Y_star: ", Y_star_mcmc, "(+ ", Y_star_mcmc_err[1] - Y_star_mcmc, " - ", Y_star_mcmc - Y_star_mcmc_err[0], " )")

	# Extract best-fit parameters from MCMC
	Y_star_mcmc = np.median(samples, axis=0)

	# Compute model velocities with best-fit parameters from MCMC
	v_model_sq_mcmc = mond_velocity_squared(Y_star_mcmc)
	v_model_mcmc = np.sqrt(v_model_sq_mcmc)

	# Plot the data
	plt.errorbar(r, v_obs, yerr=v_err, fmt='o', label='Observed data')

	# Plot the model
	plt.plot(r, v_model_mcmc, label='Best-fit model after emcee', color='red')
	plt.xlabel('Radius (kpc)')
	plt.ylabel('Velocity (km/s)')
	plt.legend()
	plt.savefig("/media/stavros/aaf194dd-624d-4ece-8e56-395e27a93185/Pythonproject/de/" + GALAXY_NAME + 'fit_results-afterMCMC.pdf')
	plt.close()
