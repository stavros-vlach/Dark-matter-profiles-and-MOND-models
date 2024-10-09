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
import csv
import dynesty

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
def mond_velocity_squared(Y_star):
        v_star_sq = Y_star * v_disk_sq + 1.4 * Y_star * v_bulge_sq
        v_N_sq = v_star_sq + v_gas_sq
        a_N = v_N_sq / r
        v_MOND_sq = v_N_sq * np.sqrt(0.5 + 0.5 * np.sqrt(1 + (2 * a0 / a_N)**2))
        return v_MOND_sq

def log_probability(theta):
        """Define the log probability function for emcee."""
        Y_star = theta
        if not(0 < Y_star < 10.0):
                return -np.inf
        v_model_sq = mond_velocity_squared(Y_star)
        chi_sq = np.sum(((v_obs**2 - v_model_sq) / v_err**2) ** 2)
        chi_sq += ((Y_star - Y_hat_star) / sigma_Y_star) ** 2
        log_likelihood = -0.5 * chi_sq
        if not np.isfinite(log_likelihood):
                print(1)
                return -np.inf
        return log_likelihood

def log_probability_dynesty(theta):
        """Define the log probability function for dynesty."""
        Y_star = theta

        v_model_sq = mond_velocity_squared(Y_star)
        chi_sq = np.sum(((v_obs**2 - v_model_sq) / v_err**2) ** 2)
        chi_sq += ((Y_star - Y_hat_star) / sigma_Y_star) ** 2
        log_likelihood = -0.5 * chi_sq
        if not np.isfinite(log_likelihood):
                print(1)
                return -np.inf
        return log_likelihood[0]
def prior_transform(u):
    """Transforms our unit cube samples `u` to a flat prior between -10. and 10. in each variable."""
    return 10. * u
 
def chi_square(theta):
	Y_star = theta
	v_MOND_sq = mond_velocity_squared(Y_star)
	chi_sq = np.sum(((v_obs**2 - v_MOND_sq) / v_err**2)**2)
	chi_sq += ((Y_star - Y_hat_star) / sigma_Y_star) ** 2
	return chi_sq / (len(r) - len(theta))

#bounds for differential_evolution
bounds = [(0, 10.0)]

#parameters for emcee
ndim = 1
nwalkers = 80
nsteps = 200
with open('parameters_MOND.csv','w') as testfile:
	csv_writer=csv.writer(testfile)
	csv_writer.writerow(["Galaxy", "Y*", "Y*_err", "log(Z)", "sigma_log(Z)"])
	
	path = "Rotmod_LTG/"
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
		
		v_MOND_sq_best = mond_velocity_squared(Y_star_best)
		v_MOND_best = np.sqrt(v_MOND_sq_best)

		plt.plot(r, v_MOND_best, label = 'Best-fit model before emcee', color = 'red')
		plt.errorbar(r, v_obs, yerr=v_err, fmt='o', label='Observed data')
		plt.ylabel('$V(km/s)$')
		plt.xlabel('$R(kpc)$')
		plt.legend()
		plt.savefig(GALAXY_NAME + ' fit_results-beforeMCMC.pdf')
		plt.close()

		# Set up the sampler
		# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=())
		# initialize our nested sampler
		dsampler = dynesty.DynamicNestedSampler(log_probability_dynesty, prior_transform, ndim=1,
		                                        bound='multi', sample='unif')
		dsampler.run_nested(maxiter=20000, use_stop=False, maxcall=1e6)
		dres = dsampler.results

		# Initialize the walkers
		pos = Y_star_best + 1e-2 * np.random.randn(nwalkers, ndim)

		# # Run the sampler
		# sampler.run_mcmc(pos, nsteps, progress=True)

		# Extract the samples
		# samples = sampler.get_chain(discard=100, thin=10, flat=True)

		samples = dsampler.results.samples  # Posterior samples
		logl = dsampler.results.logl  # Log-likelihoods of the samples
		logwt = dsampler.results.logwt  # Log weights
		weights = np.exp(logwt - logwt.max())  # Exponentiate the log weights
		weights /= np.sum(weights)  # Normalize the weights
		indices = np.random.choice(len(samples), size=len(samples), p=weights)
		resampled_samples = samples[indices]

		#Bayesian evidense
		log_z = np.median(dsampler.results.logz)
		sigma_log_z = np.percentile(log_z, [16, 84])

		# Create a corner plot
		fig = corner.corner(resampled_samples, labels=["$Y_\\star$"])
		fig.savefig(GALAXY_NAME + "_fit_corner_test.pdf")
		plt.close()

		Y_star_mcmc = np.percentile(resampled_samples, 50, axis=0)
		Y_star_mcmc_err = np.percentile(resampled_samples[:, 0], [16, 84])
	
		csv_writer.writerow([GALAXY_NAME, float(Y_star_mcmc), Y_star_mcmc_err, log_z,sigma_log_z])

		# Extract best-fit parameters from MCMC
		Y_star_mcmc = np.median(resampled_samples, axis=0)

		# Compute model velocities with best-fit parameters from MCMC
		v_model_sq_mcmc = mond_velocity_squared(Y_star_mcmc)
		v_model_mcmc = np.sqrt(v_model_sq_mcmc)

		# Plot the data
		plt.errorbar(r, v_obs, yerr=v_err, fmt='o', label='Observed data')

		# Plot the model
		plt.plot(r, v_model_mcmc, label='Best-fit model after dynesty', color='red')
		plt.xlabel('Radius (kpc)')
		plt.ylabel('Velocity (km/s)')
		plt.legend()
		plt.savefig(GALAXY_NAME + '_fit_results-afterMCMC_test.pdf')
		plt.close()
