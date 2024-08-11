# Intro 
Here we write/discuss the next steps for our analysis.
The general idea is (a) to compile a large number of modified gravity-based MOND models
and (b) to fit them on the SPARC dataset along with DM models, aiming to (c) compare their fitting quality - model selection.


# a. Compilation of Modified gravity models
-Fotis: This is mine, essentially a list of papers where velocity curves are extracted/presented-
1. Make a list of papers and start writing some details for each model
2. Implement each of them.
3. Test the implementation.


# b. Fitting on SPARC dataset

1. I encountered abnormal stop of the for loop on the einasto code (from emcee: Prob. function returned NaN). 
We need to make sure that this do not happen. An "easy" solution could be to put an if condition, however this could significantly slow down the code.
2. In order to do the model selection stuff properly, we need to keep the chains from each run. a proper saving method must be implemented. The utils.py contains relevant stuff, modification is needed though.
3. We need to test thoroughly all models.
This can be done by setting small number of steps/walkers
4. We need also an automated way (funtion) that will take the chains and plot the contours for each model/galaxy. In the paper, we will need to provide these plots. Also, we need to give indicative rotation curve fitts of all models for a given set of galaxies with the corresponding error bars. See Fig. 
5. An automated way to check if emcee (or whatever) has converged must be implemented also. For the case of not convergence, we must auromatically drop this particular galaxy OR re-run the sampler with increased number of steps/walkers. For the case of emcee, we can use the Gelman-rubin criterion values. 
6. In general, I anticipate that the models will fail in different occassions and for different causes and we would like to have the ability to provide relevant plots (i.e. have a dataframe that would contain model, dataset and fitting results, which are Gelman-rubin, information criteria etc.)

# Model selection
1. We can use initially the already implemented AIC, BIC, DIC (these exist on utils.py)
2. We could add Bayesian Evidence calculation - either using https://github.com/yabebalFantaye/MCEvidence
or by using another sampler (instead of emcee). Both choices must be explored.
