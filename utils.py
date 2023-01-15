import numpy as np
from scipy.optimize import minimize,fsolve,newton
from scipy.integrate import quad,simps,odeint,solve_ivp,romb,cumtrapz,trapz,romberg
import scipy.special as sp
import scipy.stats as st
# import matplotlib.pyplot as plt
import emcee
# import corner
import utils as ut
import multiprocessing
import time as tm
try:
    from schwimmbad import MPIPool
except ModuleNotFoundError as e:
    print('Schwimmbad is not installed')


def run_emcee(states, initial,lnlikelihood,save_chain = 'test',prog_bar = True,extra_force = 'multithreading',print_log= ''):
    '''
    Runs the emcee3 sampler on diferent occasions (multithreading OR parallell).
    Input: states--->the number of steps, initial (a ndim * nwalkers ), lnlikelihood: is the likelihood function
    save_chain---> default '' else saves the chain 
    in the specified filename.h5.
    !!! The "parallell" kewyword HAS NOT tested yet. The print_log is to be defined. Currently, the whole work is done
    with the log_maker function.
    ####If save_chain == '', True, there is an error:  , does not go within "if save_chain".
    '''
    nwalkers, ndim = len(initial),len(initial[0])
    # print(nwalkers)
    # print(ndim)
    if save_chain:
        filename = save_chain + ".h5"
        backends = emcee.backends.HDFBackend(filename)
        backends.reset(nwalkers, ndim)
    
        if extra_force == "multithreading":
            with multiprocessing.Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlikelihood, pool=pool,backend=backends)
                start = tm.time()
                sampler.run_mcmc(initial, states, progress=prog_bar)
                end = tm.time()
                multi_time = end - start
                # print("Multiprocessing took {0:.1f} seconds".format(multi_time))
        elif extra_force == "parallell":
            with MPIPool() as pool:
                if not pool.is_master():
                    pool.wait()
                    sys.exit(0)
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlikelihood, pool=pool,backend=backends)
                start = tm.time()
                sampler.run_mcmc(initial, states, progress=prog_bar)
                end = tm.time()
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlikelihood)
            start = tm.time()
            sampler.run_mcmc(initial, states, progress=prog_bar)
            end = tm.time()
        multi_time = end - start
    ##############################
    return sampler,multi_time

def ioi(vec1,vec2,cov1,cov2):
    '''
    Provides the Index of Inconsistency (IOI) as defined at
    e.g arxiv:1708.09813 and references therein.
    Add.Refs: 1705.05303,
    INPUT:
        vec1: (n x 1 array) the parameters values from the FIRST experiment
        vec2: (n x 1 array) the parameters of the SECOND experiment
        cov1: (n x n array) the cov matrix for the parameters of the 1st exp.
        cov2: (n x n array) the cov matrix for the parameters of the 2st exp.

    OUTPUT:
        IOI: the numerical value of ioi
    '''
    try:
        delta_vec = (vec1-vec2) 
    except TypeError:
        delta_vec = np.array(vec1) - np.array(vec2)
    ioi = 0.5 * (delta_vec @ np.linalg.inv(cov1 + cov2) @ delta_vec)
    return ioi

def covar_of_pars(samplerOrfilename):
    '''
    Constructs the covariance between the bf parameters of an emcee run
    using the standard definition:
        \Sigma_{ij} = E[(X_{i}-\mu_{i})(X_{j}-\mu_{j})]
    INPUT: sampler object OR .h5 filename
    OUTPUT: The cov matrix and sigmas for the parameters
    '''
    if True:
        reader = emcee.backends.HDFBackend(samplerOrfilename+'.h5')
    # else:
        # samplerOrfilename.get_chain()
        tau = reader.get_autocorr_time()
        burnin = int(2*np.max(tau))
        thin = int(0.5*np.min(tau))
        # samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
        # log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
        # log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)
        states,n = reader.get_chain(discard=burnin, flat=True, thin=thin).shape
        # print(n)
        # print(states)
        chain = reader.get_chain(discard=burnin, flat=True, thin=thin)
        means = np.zeros(n)
        for i in range(n):
            means[i] = sum(chain[:,i])/states
        Cov = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                Cov[i][j] = sum((chain[:,i] - means[i])*(chain[:,j] - means[j]))/states
        sigmas = np.diag(Cov)**0.5
    return Cov,sigmas

def cov_of_errors(vec,dim):
    '''
    Constructs the diagonal cov matrix of given vec of vals AND dimensions
    '''
    cov = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            try:
                cov[i][j] = vec[i]**2.
            except IndexError:
                cov[i][j] = 1.0
    return cov

def mod_aik(xtetr,k,N):
    return xtetr+2.*k + 2.*k*(k+1.)/(N-k-1.)

def bic(xtetr,k,N):
    return xtetr + k*np.log(N)


def dev_ic(sampler,b = 0.3):
    '''
    Returns the deviance information criterion (DIC, :) )
    INPUT: A sampler (sampler object) OR an object from importing .npz file (dict, with 'arr_1' containing 
        the lnprob vals and the 'arr_2' the states), b is the burn in parameter (default == 0.3)
    OUTPUT: The value of the crit., float
        Reference:
    https://en.wikipedia.org/wiki/Deviance_information_criterion
    ----May be generalized to take as input different forms
    '''
#     func = lnL_lcdm
    try:
        n = len(sampler.chain[0][0])
        burn_in = int(sampler.chain.shape[1] * b)
        sample = sampler.get_chain(discard=burn_in, thin=1, flat=True)
        all_l_vals = sampler.get_log_prob(discard=burn_in, thin=1, flat=True)
#         D_mean = -2.*np.mean(np.ma.masked_invalid(all_l_vals))
#     #     n = len(sampler.chain[0][0])
#         D_mthet = -2.*sampler.get_lnprob([np.mean(sample[:,ii]) for ii in range(0,n)])
#         n = len(sample)

        D_mthet = -2.*sampler.log_prob_fn([np.mean(sample[:,ii]) for ii in range(0,n)])
#         count = 0
#         for ln_vals in all_l_vals:
            
#             if not(np.isfinite(ln_vals)):
#                 count += 1
#         print(count)
    except (AttributeError):
        try:
            walkers, states, dim = sampler['arr_1'].shape
            
            burn_in = int(b * states)
            sample = sampler['arr_1'][:, burn_in:].reshape(-1,dim)
#             print(sample.shape)
#             n = len(sample)
            all_l_vals = sampler['arr_0'][:,burn_in:].reshape(-1,dim)
#             print([np.mean(sample[:,ii]) for ii in range(0,dim)])
           
            D_mthet = -2.*sampler.log_prob_fn([np.mean(sample[:,ii]) for ii in range(0,dim)])
#             print(D_mthet)
        except (IndexError):
            print('input is of {0} type. I can not handle it'.format(type(sampler)))
            return 0
    D_mean = -2.*np.mean(np.ma.masked_invalid(all_l_vals))
    
    #     n = len(sampler.chain[0][0])
#     print(D_mean)
    DIC = 2.*D_mean - D_mthet
    return DIC
def Gel_Rub(chains):
    '''
    For a given emcee.chain calculates the Gelman & Rubin criterion.
    (Desription elsewhere)
    input: (numpy.ndarray)
    output: (numpy.array), the value of GR criterion for each parameter
    '''
    M,n,params_num = chains.shape
    theta_bar_m = np.zeros((M,params_num))
    sm = np.zeros((M,params_num))
    Bs = np.zeros(params_num)
    Ws = np.zeros(params_num)
    Vs = np.zeros(params_num)
    for param_val in range(params_num):
        for single_chain in range(M):
            ### means of each chain ###
            suma = sum(chains[single_chain,:,param_val])
            theta_bar_m[single_chain][param_val] = suma/n
            ### s_{\mu}^2 ###
            sm_single = sum([(chains[single_chain,indexi,param_val] - theta_bar_m[single_chain][param_val])**2. for indexi in range(n)]) #kala os edo
            sm[single_chain][param_val] = sm_single* (1./(n-1.))
        theta_bar = sum(theta_bar_m[:,param_val])*(1./M)
        Bs[param_val] = n/(M - 1) * sum([(theta_bar - theta_bar_m[i][param_val])**2. for i in range(M)])
        Ws[param_val] = (1./M) *sum(sm[:,param_val])
        Vs[param_val] = ((n-1.)/n ) * Ws[param_val] + ((M+1.)/(n*M)) * Bs[param_val]
    res = (Vs/Ws)**2.
    return res

def log_maker(sampler,nn,multi_time,fname,labels,sxolia= '',b=0.3,fname_auto = ''):
    '''
    Creates a .txt file containing all the information for the given emcee analysis.
    INPUT: sampler object, a string containing the parameter names, the dataset length (int), a string containing
    relevant comments, the % amount of the burn-in states and another string that is the relevant filename.
    (A try structure to gather possible errors could be nice. Also, the likelihood function is called within the current
    function along with the various inf. criteria and GR diagnostics. We need to save the chain also.)
    '''

    date = tm.localtime()
    burn_in = int(sampler.chain.shape[1] * b)
    prob = sampler.flatlnprobability
    pos = sampler.flatchain
    nwalkers,states, nparams = sampler.chain.shape
    # print(sampler.chain.shape)
    par_num = len(labels)
    labels = [elem.replace('$','') for elem in labels]
    date = '{0}_{1}_{2}_hrs {3} {4} '\
            .format(date[2],date[1],date[0],date[3],date[4])
    if fname_auto:
        filename = fname_auto
    else:
        filename = 'res_' + date + '_' + fname[0:10] + '.txt'
    np.savetxt('res_chain'+ fname[0:10] +'.txt',sampler.get_chain(discard=burn_in, thin=1, flat=True))
    np.savetxt('res_pos'+ fname[0:10] +'.txt',sampler.get_log_prob(discard=burn_in, thin=1, flat=True))
    with open(filename,'w') as fp:
    #     fp.write('Results of the parameters \n:'+ str(pos[np.argmax(prob)])+'\n' )
        fp.write('Parameter estimation results at '+ date +'\n')
        fp.write('The free parameters: \n')
        fp.write('------------------------------------ \n')
        points = []
        try:
            for i in range(nparams):
                mcmc = np.percentile(sampler.get_chain(discard=burn_in, thin=1, flat=True)[:, i], [15.87, 50, 84.13])
                points.append(mcmc[1])
                q = np.diff(mcmc)
                txt = "\t {3} = {0:.5f}_{{-{1:.5f}}}^{{+{2:.5f}}}  \n"
                txt = txt.format(mcmc[1], q[0], q[1],labels[i])
                fp.write(txt)
            min_chisq = -2.*sampler.log_prob_fn(points)
            red_chisq = min_chisq/(nn-par_num*1.)
            fp.write('------------------------------ \n')    
            fp.write('The chi-square at minimum is: \n \t \chi_{{min}}^2 = {0:.4f} \n'.format(min_chisq))
            fp.write('The reduced chi-square at minimum is: \n \t \chi^2_{{min}}/dof = {0:.4f} \n'.format(red_chisq))
            fp.write('Inf.criteria \n \t AIC: {0:.4f} \n \t BIC: {1:.4f} \n \t DIC: {2:.4f} \n '\
            .format(mod_aik(min_chisq,par_num*1.,nn),bic(min_chisq,par_num*1.,nn), dev_ic(sampler)))
        except IndexError as error:
            sxolia = sxolia + '\n The MCMC sampler did not gather enough points which means that the RESULTS ARE NOT TO BE TRUSTED'
        
        fp.write('MCMC parameters:\n \t nwalkers = {0}\n \t number of states = {1} \n \t GR crit. = {2} \n'.\
                 format(nwalkers,states,Gel_Rub(sampler.chain[:, burn_in:, :])))
        fp.write('------------------------------------ \n')
        fp.write('Time needed: \n' + '{0:.3f}'.format(multi_time/3600)+ ' hrs\n')
        fp.write('the file: ' + str(fname) + '\n \n' )
        fp.write('------------------------------------ \n')
        fp.write('Comments: \n' + sxolia)
# def corner_res(sampler,burn=int(0.3*states),truths=None,fpars=labels,ranges=None,sigmas=[0.5,1,2,3],smooth=0):
#     # samples=samples[burn:,:]
#     samples = sampler.chain.reshape((-1, ndim))
#     # print(samples.shape)
#     med = np.quantile(samples,q=[0.25,0.5,0.75],axis=0)[1]
#     truths = med if truths is None else truths
#     levels=[1-np.exp(-s**2/2) for s in sigmas]#(1-np.exp(-0.5**2/2),1-np.exp(-1**2/2),1-np.exp(-2**2/2),1-np.exp(-3**2/2))
#     fpars = ['p{}'.format(i) for i in range(samples.shape[1])] if fpars is None else fpars
#     Q=np.quantile(samples,q=[0.16,0.5,0.84],axis=0)
#     if ranges is None:
#         Q10=np.quantile(samples,q=[0.03,0.97],axis=0)
#         ranges=Q10.T
#     for fpar,q in zip(fpars,Q.T):
#         print(f"Parameter {fpar} quantiles  ({q[0]:.2f} - {q[1]:.2f} - {q[2]:.2f})")
#     fig = corner.corner(samples, labels=fpars,range=ranges,show_titles=True,truths=truths,quantiles=[0.16,0.5,0.84],levels=levels,smooth=smooth)#,truths=list(fit.best_values.values())
#     return med