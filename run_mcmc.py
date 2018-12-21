# -*- encoding: utf-8 -*-
#!/usr/bin/python

import sys, os

#import mpi4py

import matplotlib
#matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import phoebe
from phoebe import u # units
logger = phoebe.logger(clevel='DEBUG', flevel='DEBUG')
#logger = phoebe.logger(clevel='DEBUG', flevel='DEBUG', filename='tutorial.log')

import numpy as np

#from mpi4py import MPI

#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()
#print "hello world from process ", rank, " out of ", size
#nproc = size

import emcee
#from schwimmbad import MPIPool
from emcee.utils import MPIPool


import gc
import scipy.stats as st
import traceback

import copy_reg
import types



def make_binary():
    global b
    b = phoebe.default_binary()
    
    mag0 = 10.0

    #t0 = 2457865.52276
    #period = 1.93974127612

    #mag0 = 0
    #f0 = 3640
    times, magnitudes, sigmas = np.loadtxt('phot_V.txt', unpack=True, dtype='float')
    b.add_dataset('lc', times=times, dataset='lc01', passband='Johnson:V')
    fluxes = 10**(-0.4*(magnitudes-mag0)) #* f0
    plt.scatter(range(len(fluxes)), fluxes, c='green')
    sigmas_up = np.abs(10**(-0.4*((magnitudes+sigmas)-mag0))-fluxes)
    sigmas_down = np.abs(10**(-0.4*((magnitudes-sigmas)-mag0))-fluxes)
    sigmas = np.average(zip(sigmas_up, sigmas_down), axis=1)  
    #phases = [((x-t0)%period)/period for x in times]
    #plt.scatter(phases, fluxes, s=3)
    #plt.errorbar(phases, fluxes, yerr=sigmas, fmt='.')
    b['lc01@dataset']['fluxes'] = fluxes
    b['lc01@dataset']['sigmas'] = sigmas

    #mag0 = 0
    #f0 = 4760
    times, magnitudes, sigmas = np.loadtxt('phot_I.txt', unpack=True, dtype='float')
    b.add_dataset('lc', times=times, dataset='lc02', passband='Cousins:I')
    fluxes = 10**(-0.4*(magnitudes-mag0)) #* f0
    plt.scatter(range(len(fluxes)), fluxes)
    sigmas_up = np.abs(10**(-0.4*((magnitudes+sigmas)-mag0))-fluxes)
    sigmas_down = np.abs(10**(-0.4*((magnitudes-sigmas)-mag0))-fluxes)
    sigmas = np.average(zip(sigmas_up, sigmas_down), axis=1)    
    #phases = [((x-t0)%period)/period for x in times]
    #plt.scatter(phases, fluxes, s=3)
    #plt.errorbar(phases, fluxes, yerr=sigmas, fmt='.')
    b['lc02@dataset']['fluxes'] = fluxes
    b['lc02@dataset']['sigmas'] = sigmas

    plt.show()
    
    #phs = np.array(np.linspace(0, 1.0, 1.0/0.1 + 1))
    #times = [x*period+t0 for x in phs]
    #b.add_dataset('lc', times=times, dataset='ex01', passband='Johnson:V'
   
    
    b['t0_supconj'] = 2457003.3261
    b['period@binary'] = 4.306192
    b['sma@binary'] = 13.5
    b['q@binary'] = 0.877
    b['incl@binary'] = 90
    b['teff@primary'] = 6000
    b['teff@secondary'] = 5000
    b['requiv@primary@component'] = 1
    b['requiv@secondary@component'] = 0.9
    
    
    
    b.set_value_all('ld_func', 'logarithmic')
    b.set_value_all('ld_coeffs', [0.5,0.5])
    #b.set_value('irrad_method', 'none')
    b['pblum_ref@primary@lc01@lc_dep@dataset'] = 'self'
    b['pblum_ref@primary@lc02@lc_dep@dataset'] = 'self'
    
    b['pblum_ref@secondary@lc01@lc_dep@dataset'] = 'primary'
    b['pblum_ref@secondary@lc02@lc_dep@dataset'] = 'primary'
    

    b.add_compute('phoebe', compute='preview', irrad_method='none')

    print "check1"

    #axs, artists = b['lc@dataset'].plot(x='phases', xlim=(-0.6, 0.6))
    #plt.show()
    
    b.save('solution_5')


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
        return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

comp_pars = ["value@mass@primary@component", "value@mass@secondary@component"]

def cost_func(dataset):
    global b

    times = b['value@times@%s@dataset' % dataset]
    fluxes = b['value@fluxes@%s@dataset' % dataset]
    sigmas = b['value@sigmas@%s@dataset' % dataset]

    times2 = b['value@times@%s@latest@model' % dataset]    
    fluxes2 = b['value@fluxes@%s@latest@model' % dataset]    
        
    times, fluxes, sigmas = zip(*sorted(zip(times, fluxes, sigmas)))
    times2, fluxes2 = zip(*sorted(zip(times2, fluxes2)))
       
    cost = np.sum((np.array(fluxes2) - np.array(fluxes))**2 / np.array(sigmas)**2)
    return cost

def lnprob(x, adjpars, priors):   
    global b

    # Check to see that all values are within the allowed limits:   
    if not np.all([priors[i][0] < x[i] < priors[i][1] for i in range(len(priors))]):
        return -np.inf, []

    # x[] is an array of sampled values from parameter priors:
    try:
        for i, j in enumerate(adjpars):
            #print j, x[i]
            b[j] = x[i]
        
        b.run_compute(compute='preview')
        #b.run_compute(compute='detailed')

        #axs, artists = b['lc@dataset'].plot(x='phases', xlim=(-0.6, 0.6))
        #axs, artists = b['lc@latest@model'].plot(x='phases', xlim=(-0.6, 0.6))
        #plt.show()

        cost = 0
        for dat in ['lc01', 'lc02']:#, 'lc03', 'lc04']:
            cost += cost_func(dat)
        
        lnp = - 0.5 * cost
        #print lnp, [b[par] for par in comp_pars]
        return lnp, [b[par] for par in comp_pars]
    except:
        print_exception()
        return -np.inf, [0 for par in comp_pars]

def run_sampler(sampler, niter, ntot_iter, p0, phoebe_file, prefix):

    for result in sampler.sample(p0, iterations=niter, storechain=False):        
        ntot_iter += 1
        
        position = result[0]
        computed = result[3]

        f = open(phoebe_file + prefix + '.mcmc', "a")
        
        # for each walker      
        for k in range(position.shape[0]):            
            
            f.write("%d %s %s %f\n" % ( k, 
                                    " ".join(['%.12f' % i for i in position[k]]),   # phoebe adjusted parameters
                                    " ".join(['%.12f' % i for i in computed[k]]),   # phoebe computed parameters    
                                    result[1][k]                                    # lnprob value
                                  )
            )

        f.close()

    return ntot_iter

def run(phoebe_file, adjpars, priors, state, nwalkers, niter, prefix):    
    global b
    b = phoebe.Bundle.open(phoebe_file)
    
    #b.add_compute('phoebe', compute='preview', irrad_method='none')
    #b.add_compute('phoebe', compute='detailed', irrad_method='wilson')

    ndim = len(adjpars)
    
    
    pool = MPIPool(loadbalance=True) #, debug=True
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    print "start"   

    if state is not None:
        p0 = np.genfromtxt(state)[-nwalkers:, 1:ndim+1]
        print "state file loaded p0"
    else:
        p0 = np.array([[p[0] + (p[1]-p[0])*np.random.rand() for p in priors] for i in xrange(nwalkers)])

    #print p0

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[adjpars, priors], pool=pool)
    #sampler.run_mcmc(p0, niter)
    ntot_iter = 0

    ntot_iter = run_sampler(sampler, niter, ntot_iter, p0, phoebe_file, prefix)  

    pool.close()

def onoff_parameters(init_sets, old, new, priors):
    """
    Grabs the latest chain, extracts the state, adds/removes a parameter and makes a new state.
    """

    chain_old = np.loadtxt(init_sets[old][2]+old+'.mcmc')[-init_sets[new][0]:,]
    chain_new = []
    for i, j in enumerate(chain_old):
        # Add walker index
        chain_new.append([i])
        count = 0
        # Add adjusted parameters
        for k, h in enumerate(init_sets[old][3]):
            if h:
                count += 1

            if h and init_sets[new][3][k]:
                chain_new[i].append(j[count])
            elif init_sets[new][3][k]:
                chain_new[i].append(priors[k][0] + (priors[k][1]-priors[k][0])*np.random.rand())
        # Add phoebe computed parameters
        for co in comp_pars:
            count += 1
            chain_new[i].append(j[count])
        # Add lnprob
        chain_new[i].append(j[-1])



    

    np.savetxt(init_sets[new][2]+new+'.mcmc', np.array(chain_new))

def print_exception():
    e = sys.exc_info()
    exc_type, exc_value, exc_traceback = e
    a, j = (traceback.extract_tb(exc_traceback, 1))[0][0:2]
    k = (traceback.format_exception_only(exc_type, exc_value))[0]
    print a, j, k


if __name__ == '__main__':
    
    #make_binary()
    #sys.exit()
    # set, nwalkers, niters
    

    init_sets = {
      'a':[32, 500, 'solution_1', [1,1,0,0,1,0,1,1,1,1,1,1]],
      'b':[32, 500, 'solution_2', [0,0,0,0,1,0,1,1,1,1,1,1]],
      'c':[32, 500, 'solution_3', [1,1,0,0,1,0,1,1,1,1,1,1]],
      'd':[32, 500, 'solution_3', [0,0,0,0,1,0,1,1,1,1,1,1]],
      'e':[32, 500, 'solution_3', [0,0,0,0,1,0,1,1,1,1,1,1]], #color constrained, not really
      'f':[32, 1500, 'solution_3', [0,0,0,0,1,0,1,1,1,1,1,0]], #color constrained!
      'g':[64, 1500, 'solution_3', [0,0,0,0,1,0,1,1,1,1,1,0]], #color constrained!
      'h':[64, 1500, 'solution_3', [0,0,0,0,1,0,1,1,1,1,1,0]], #color constrained!
      'i':[64, 1500, 'solution_4', [0,0,0,0,1,0,1,1,1,1,1,0]], #color constrained!
      'j':[64, 1500, 'solution_5', [0,0,0,0,1,0,1,1,1,1,1,0]], #color constrained!
    }
    set = 'j'
    state =  None#'../solution_3h.mcmc'#'state'
    
    

    adjpars = [
        ['t0_supconj', 2457003.125000, 2457003.525000], 
        ['period@binary', 4.106193, 4.506193], 
        ['sma@binary', 4., 15.], 
        ['q@binary', 0.2, 1.0], 
        ['incl@binary', 60., 90.], 
        ['ecc@binary', 0.0, 0.2], 
        ['teff@primary', 3000, 10000], 
        ['teff@secondary', 3000, 10000],
        ['requiv@primary@component', 0, 4.],
        ['requiv@secondary@component', 0, 4.], 
        ['pblum@primary@lc01@lc_dep@dataset', 0, 10.], 
        ['pblum@primary@lc02@lc_dep@dataset', 0, 10.]
        #b['gravb_bol@primary'] = 1.0
        #b['gravb_bol@secondary'] = 0.32
    ]
    priors = [[x[1], x[2]] for x in adjpars]

    #onoff_parameters(init_sets, 'h', 'j', priors)
    #sys.exit()

    mask = np.array(init_sets[set][3]) == 1
    adjpars = np.array(adjpars)[mask]
    priors = np.array(priors)[mask]
    adjpars = [x[0] for x in adjpars]
    priors = [tuple(x) for x in priors]

    
    try:
        os.makedirs(set)
    except:
        pass
    

    os.chdir(set)
    run('../'+init_sets[set][2], adjpars, priors, state, init_sets[set][0], init_sets[set][1], set)
    os.chdir('../')
    