"""
Fit joint flux-size distribution
"""

import os
import numpy

from . import analysis
from . import files

NGAUSS_DEFAULT=8
N_ITER_DEFAULT=1000
MIN_COVAR=1.0e-6
COVARIANCE_TYPE='full'

def make_joint_gmm(means, covars, weights):
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
            covariance_type=COVARIANCE_TYPE)
    gmm.means_ = means.copy()
    gmm.covars_ = covars.copy()
    gmm.weights_ = weights.copy()

    return gmm

_par_labels={}
_par_labels[4] = [r'$\eta_1$',
                  r'$\eta_2$',
                  r'$log_{10}(T)$',
                  r'$log_{10}(F)$']
_par_labels[4] = [r'$\eta_1$',
                  r'$\eta_2$',
                  r'$log_{10}(T)$',
                  r'$log_{10}(F_b)$',
                  r'$log_{10}(F_d)$']

def fit_joint(pars,
              ngauss=NGAUSS_DEFAULT,
              n_iter=N_ITER_DEFAULT,
              eps=None):
    """
    pars should be [nobj, ndim]

    for a simple model this would be
        g1,g2,T,flux
    for bdf this would be
        g1,g2,T,flux_b,flux_d
    """
    import lensing

    ndim = pars.shape[1]
    assert (ndim==4 or ndim==5),"ndim should be 4 or 5"

    par_labels=_par_labels[ndim]

    logpars = pars.copy()

    eta1, eta2 = lensing.util.g1g2_to_eta1eta2(pars[:,0], pars[:,1])

    log_T = numpy.log10( pars[:,2] )

    logpars[:,0] = eta1
    logpars[:,1] = eta2
    logpars[:,2] = log_T

    logpars[:,3:] = numpy.log10( pars[:,3:] )

    gmm=fit_gmix(logpars,ngauss,n_iter)

    output=numpy.zeros(ngauss, dtype=[('means','f8',ndim),
                                      ('covars','f8',(ndim,ndim)),
                                      ('weights','f8')])
    output['means']=gmm.means_
    output['covars']=gmm.covars_
    output['weights']=gmm.weights_

    plot_fits(logpars, gmm, eps=eps, par_labels=par_labels)

    """
    extra='ngauss%d' % ngauss
    eps=files.get_dist_path(version,model,'joint-dist',ext='eps',
                            extra=extra)
    gmmtest=make_joint_gmm(output['means'], output['covars'], output['weights'])


    flux_mode = 10.0**log_flux_mode
    T_near = 10.0**log_T_near

    print
    print 'log_flux_mode:',log_flux_mode
    print 'log_T_near:   ',log_T_near
    print 'flux_mode:    ',flux_mode
    print 'T_near:       ',T_near

    h={'logfmode':log_flux_mode,
       'fmode':flux_mode,
       'logTnear':log_T_near,
       'Tnear':T_near}
    files.write_dist(version, model, 'joint-dist', output,header=h,
                     extra=extra)

    """
    return output, h


def fit_gmix(data, ngauss, n_iter):
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
            covariance_type=COVARIANCE_TYPE)

    gmm.fit(data)

    if not gmm.converged_:
        raise ValueError("did not converge")

    return gmm

_good_ranges={}
_good_ranges['exp'] = {'flux':[0.0, 100.0]}
_good_ranges['dev'] = {'flux':[0.0, 100.0]}
 
def select_by_flux(data, model):
    """
    Very loose cuts
    """

    flux, flux_err, T, T_err = analysis.get_flux_T(data, model)

    rng = _good_ranges[model]['flux']
    w,=numpy.where( (flux > rng[0]) & (flux < rng[1]) )
    return w


def plot_fits(pars, gmm, show=False, eps=None, par_labels=None):
    """
    """
    import esutil as eu
    import biggles

    num=pars.shape[0]
    ndim=pars.shape[1]

    samples=gmm.sample(num*100)

    tab=biggles.Table(1, ndim)

    for dim in xrange(ndim):
        plt = _plot_single(pars[:,dim], samples[:,dim])
        if par_labels is not None:
            plt.xlabel=par_labels[dim]
        else:
            plt.xlabel=r'$P_%s$' % dim

        tab[0,dim] = plt

    tab.aspect_ratio=0.5

    if show:
        tab.show()

    if eps:
        print eps
        d=os.path.dirname(eps)
        if not os.path.exists(d):
            os.makedirs(d)
        tab.write_eps(eps)

def log_T_to_log_sigma(log_T):
    T = 10.0**log_T
    sigma = numpy.sqrt(T/2.0 )

    log_sigma = numpy.log10( sigma )

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


    key = biggles.PlotKey(0.9, 0.9, [ph, sample_ph], halign='right')

    plt = biggles.FramedPlot()

    plt.add( ph, sample_ph, key )

    return plt


def get_norm_hist(data, min=None, max=None, binsize=1):
    import esutil as eu

    hdict = eu.stat.histogram(data, min=min, max=max, binsize=binsize, more=True)

    hist_norm = hdict['hist']/float(hdict['hist'].sum())
    hdict['hist_norm'] = hist_norm

    return hdict
