"""
Fit joint flux-size distribution
"""
from __future__ import print_function

import os
import numpy
from numpy import log10, sqrt, zeros

from . import files

NGAUSS_DEFAULT=20
N_ITER_DEFAULT=5000
MIN_COVAR=1.0e-12


def fit_joint_run(run, model, do_subfields=False, **keys):
    import fitsio
    conf=files.read_config(run)

    pars_name='%s_pars' % model

    field_list=[]
    for subid in xrange(5):
        conf['subid']=subid
        data=files.read_output(**conf)

        pars=data[pars_name][:,2:]
        field_list.append(pars)

        if do_subfields:
            fits_name=files.get_prior_file(ext='fits', **conf)
            eps_name=files.get_prior_file(ext='eps', **conf)
            fit_joint([pars],
                      fname=fits_name,
                      eps=eps_name,
                      **keys)

    del conf['subid']

    dolog=keys.get('dolog',True)
    if dolog:
        conf['partype']='logpars'
    else:
        conf['partype']='linpars'
    fits_name=files.get_prior_file(ext='fits', **conf)
    eps_name=files.get_prior_file(ext='eps', **conf)
    print(fits_name)
    print(eps_name)

    fit_joint(field_list,
              fname=fits_name,
              eps=eps_name,
              **keys)

def fit_joint(field_list,
              ngauss=NGAUSS_DEFAULT,
              n_iter=N_ITER_DEFAULT,
              min_covar=MIN_COVAR,
              show=False,
              eps=None,
              fname=None,
              dolog=True,
              **keys):
    """
    pars should be [nobj, ndim]

    for a simple model this would be
        g1,g2,T,flux
    for bdf this would be
        g1,g2,T,flux_b,flux_d
    """

    print("ngauss:   ",ngauss)
    print("n_iter:   ",n_iter)
    print("min_covar:",min_covar)
    if dolog:
        print("using log pars")
        usepars = make_logpars_and_subtract_mean_shape(field_list)
    else:
        print("using linear pars")
        usepars=make_combined_pars(field_list)

        T_max=keys.get('T_max',1.5)
        Flux_max=keys.get('Flux_max',10.0)
        Ftot=usepars[:,3:].sum(axis=1)
        w,=numpy.where( (usepars[:,2] < T_max) & (Ftot < Flux_max) )
        print("keeping %s/%s" % (w.size,usepars.shape[0]))
        usepars=usepars[w,:]

    ndim = usepars.shape[1]
    assert (ndim==4 or ndim==5),"ndim should be 4 or 5"

    if dolog:
        par_labels=_par_labels_log[ndim]
    else:
        par_labels=_par_labels_lin[ndim]

    gmm=fit_gmix(usepars, ngauss, n_iter, min_covar=min_covar)

    output=zeros(ngauss, dtype=[('means','f8',ndim),
                                ('covars','f8',(ndim,ndim)),
                                ('icovars','f8',(ndim,ndim)),
                                ('weights','f8'),
                                ('norm','f8')])
    output['means']=gmm.means_
    output['covars']=gmm.covars_
    output['weights']=gmm.weights_

    for i in xrange(ngauss):
        output['icovars'][i] = numpy.linalg.inv( output['covars'][i] )

    plot_fits(usepars, gmm, eps=eps, par_labels=par_labels, show=show,
              dolog=dolog)

    if fname is not None:
        import fitsio
        print('writing:',fname)
        fitsio.write(fname, output, clobber=True)
    return output


def make_joint_gmm(weights, means, covars):
    """
    Make a GMM object from the inputs
    """
    from sklearn.mixture import GMM
    # we will over-ride values, pars here shouldn't matter except
    # for consistency
    ngauss=weights.size
    gmm=GMM(n_components=ngauss,
            n_iter=N_ITER_DEFAULT,
            min_covar=MIN_COVAR,
            covariance_type='full')
    gmm.means_ = means.copy()
    gmm.covars_ = covars.copy()
    gmm.weights_ = weights.copy()

    return gmm

_par_labels_log={}
_par_labels_log[4] = [r'$\eta_1$',
                      r'$\eta_2$',
                      r'$log_{10}(T)$',
                      r'$log_{10}(F)$']
_par_labels_log[5] = [r'$\eta_1$',
                      r'$\eta_2$',
                      r'$log_{10}(T)$',
                      r'$log_{10}(F_b)$',
                      r'$log_{10}(F_d)$']

_par_labels_lin={}
_par_labels_lin[4] = [r'$g_1$',
                      r'$g_2$',
                      r'$T$',
                      r'$F$']
_par_labels_lin[5] = [r'$g_1$',
                      r'$g_2$',
                      r'$T$',
                      r'$F_b$',
                      r'$F_d$']



def make_combined_pars(field_list):
    """
    just make the combined pars
    """
    import lensing

    ndim = field_list[0].shape[1]

    nobj=0
    for f in field_list:
        nobj += f.shape[0]

    print("nobj:",nobj)
    print("ndim:",ndim)
    pars = zeros( (nobj, ndim) )

    start=0
    for f in field_list:

        nf = f.shape[0]
        end = start + nf

        print(start,end)

        pars[start:end, :] = f[:,:]

        start += nf

    return pars


def make_logpars_and_subtract_mean_shape(field_list):
    """
    Subtract the mean g1,g2 from each field.

    Store pars as eta1,eta2 and log of T and fluxes
    """
    import lensing

    ndim = field_list[0].shape[1]

    nobj=0
    for f in field_list:
        nobj += f.shape[0]

    print("nobj:",nobj)
    print("ndim:",ndim)
    logpars = zeros( (nobj, ndim) )

    start=0
    for f in field_list:

        nf = f.shape[0]
        end = start + nf

        print(start,end)

        g1=f[:,0]
        g2=f[:,1]

        g1 -= g1.mean()
        g2 -= g2.mean()

        eta1, eta2 = lensing.util.g1g2_to_eta1eta2(g1,g2)

        logpars[start:end, 0] = eta1
        logpars[start:end, 1] = eta2

        logpars[start:end, 2] = log10( f[:,2] )

        logpars[start:end, 3:] = log10( f[:,3:] )

        start += nf

    return logpars


def fit_gmix(data, ngauss, n_iter, min_covar=MIN_COVAR):
    """
    For g1,g2,T,flux send logarithic versions:
        eta1, eta2, log10(T), log10(flux)
    
    data is shape
        [npoints, ndim]
    """
    from sklearn.mixture import GMM

    gmm=GMM(n_components=ngauss,
            n_iter=n_iter,
            min_covar=MIN_COVAR,
            covariance_type='full')

    gmm.fit(data)

    if not gmm.converged_:
        print("DID NOT CONVERGE")

    return gmm


def plot_fits(pars, gmm, dolog=True, show=False, eps=None, par_labels=None):
    """
    """
    import esutil as eu
    import biggles
    import images

    biggles.configure('screen','width', 1400)
    biggles.configure('screen','height', 800)

    num=pars.shape[0]
    ndim=pars.shape[1]

    samples=gmm.sample(num*100)

    nrow,ncol = images.get_grid(ndim+1) 

    tab=biggles.Table(nrow,ncol)

    gtot=sqrt( pars[:,0]**2 + pars[:,1]**2 )
    rgtot=sqrt( samples[:,0]**2 + samples[:,1]**2 )

    plt = _plot_single(gtot, rgtot)
    if dolog:
        plt.xlabel = r'$|\eta|$'
    else:
        plt.xlabel = r'$|g|$'
    tab[0,0] = plt

    for dim in xrange(ndim):
        plt = _plot_single(pars[:,dim], samples[:,dim])
        if par_labels is not None:
            plt.xlabel=par_labels[dim]
        else:
            plt.xlabel=r'$P_%s$' % dim

        row=(dim+1)/ncol
        col=(dim+1) % ncol

        tab[row,col] = plt

    tab.aspect_ratio=nrow/float(ncol)

    if show:
        tab.show()

    if eps:
        print(eps)
        d=os.path.dirname(eps)
        if not os.path.exists(d):
            os.makedirs(d)
        tab.write_eps(eps)

def log_T_to_log_sigma(log_T):
    T = 10.0**log_T
    sigma = sqrt(T/2.0 )

    log_sigma = log10( sigma )

    return log_sigma

def _plot_single(data, samples):
    import biggles

    valmin=data.min()
    valmax=data.max()

    std = data.std()
    binsize=0.1*std

    hdict = get_norm_hist(data, min=valmin, max=valmax, binsize=binsize)
    sample_hdict = get_norm_hist(samples, min=valmin, max=valmax, binsize=binsize)

    hist=hdict['hist_norm']
    sample_hist=sample_hdict['hist_norm']


    ph = biggles.Histogram(hist, x0=valmin, width=4, binsize=binsize)
    ph.label = 'data'
    sample_ph = biggles.Histogram(sample_hist, x0=valmin, 
                                  width=1, color='red', binsize=binsize)
    sample_ph.label = 'joint fit'


    key = biggles.PlotKey(0.1, 0.9, [ph, sample_ph], halign='left')

    plt = biggles.FramedPlot()

    plt.add( ph, sample_ph, key )

    return plt


def get_norm_hist(data, min=None, max=None, binsize=1):
    import esutil as eu

    hdict = eu.stat.histogram(data, min=min, max=max, binsize=binsize, more=True)

    hist_norm = hdict['hist']/float(hdict['hist'].sum())
    hdict['hist_norm'] = hist_norm

    return hdict
