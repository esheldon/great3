"""
Fit joint flux-size distribution
"""
from __future__ import print_function

import os
import numpy
from numpy import log10, sqrt, ones, zeros, exp, array, diag, where

from . import files

NGAUSS_DEFAULT=20
N_ITER_DEFAULT=5000
MIN_COVAR=1.0e-12

S2N_RANGE=[50.0, 1000.0]

def read_field_list(run, model, **keys):
    """
    read in results from a deep field
    """
    conf=files.read_config(run)
    s2n_range=keys.get('s2n_range',S2N_RANGE)
    noshape=keys.get('noshape',False)

    print("s2n_range:",s2n_range)
    print("noshape:",noshape)

    pars_name='%s_pars' % model
    s2n_name='%s_s2n_w' % model
    g_name='%s_g' % model

    field_list=[]
    for subid in xrange(5):
        conf['subid']=subid
        data=files.read_output(**conf)

        g=sqrt( data[g_name][:,0]**2 + data[g_name][:,1]**2 )
        w,=where(  (data[s2n_name] > s2n_range[0])
                 & (data[s2n_name] < s2n_range[1])
                 & (g < 1.0))

        if noshape:
            pars=data[pars_name][w,4:]
        else:
            pars=data[pars_name][w,2:]

        field_list.append(pars)

    return field_list

def fit_joint_run(run, model, **keys):
    import fitsio
    conf=files.read_config(run)

    keys['noshape']=keys.get('noshape',True)
    keys['dolog']=keys.get('dolog',True)

    field_list=read_field_list(run, model, **keys)
    usepars = make_logpars(field_list)

    if keys['dolog']:
        if keys['noshape']:
            conf['partype']='hybrid'
        else:
            conf['partype']='logpars'
    else:
        conf['partype']='linpars'

    fits_name=files.get_prior_file(ext='fits', **conf)
    eps_name=files.get_prior_file(ext='eps', **conf)
    print(fits_name)
    print(eps_name)

    if keys['noshape']:
        fit_joint_noshape(usepars,model,
                          fname=fits_name,
                          eps=eps_name,
                          **keys)


    else:
        fit_joint_all(usepars,model,
                      fname=fits_name,
                      eps=eps_name,
                      **keys)


def get_par_labels(model, ndim, dolog):
    if model=='bdf':
        if dolog:
            par_labels=_par_labels_bdf_log[ndim]
        else:
            raise ValueError("only log for bdf")
    elif model=='sersic':
        if dolog:
            par_labels=_par_labels_sersic_log[ndim]
        else:
            raise ValueError("only log for sersic")
    else:
        if dolog:
            par_labels=_par_labels_log[ndim]
        else:
            par_labels=_par_labels_lin[ndim]


    return par_labels

def fit_joint_noshape(usepars,
                      model,
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
        T,flux
    for bdf this would be
        T,flux_b,flux_d
    """

    print("ngauss:   ",ngauss)
    print("n_iter:   ",n_iter)
    print("min_covar:",min_covar)
    if dolog:
        print("using log pars")
    else:
        raise ValueError("no lin pars")

    ndim = usepars.shape[1]
    assert (ndim==2 or ndim==3 or ndim==4),"ndim should be 2,3,4, got %s" % ndim

    par_labels=get_par_labels(model, ndim, dolog)

    gmm0=fit_gmix(usepars, ngauss, n_iter, min_covar=min_covar)

    output=zeros(ngauss, dtype=[('means','f8',ndim),
                                ('covars','f8',(ndim,ndim)),
                                ('icovars','f8',(ndim,ndim)),
                                ('weights','f8'),
                                ('norm','f8')])
    output['means']=gmm0.means_
    output['covars']=gmm0.covars_
    output['weights']=gmm0.weights_

    for i in xrange(ngauss):
        output['icovars'][i] = numpy.linalg.inv( output['covars'][i] )

    # make sure our constructor works
    gmm_plot=make_joint_gmm(gmm0.weights_, gmm0.means_, gmm0.covars_)

    samples=gmm_plot.sample(usepars.shape[0]*100)
    plot_fits(usepars, samples, eps=eps, par_labels=par_labels, show=show,
              dolog=dolog)

    if fname is not None:
        import fitsio
        print('writing:',fname)
        fitsio.write(fname, output, clobber=True)
    return output


def fit_joint_all(field_list,model,
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
        usepars=make_combined_pars_subtract_mean_shape(field_list)

        # usepars are [gtot,T,Flux1,Flux2,...]
        T_max=keys.get('T_max',1.5)
        Flux_max=keys.get('Flux_max',10.0)
        T=usepars[:,1]
        Ftot=usepars[:,2:].sum(axis=1)
        w,=numpy.where( (T < T_max) & (Ftot < Flux_max) )
        print("keeping %s/%s" % (w.size,usepars.shape[0]))
        usepars=usepars[w,:]

    ndim = usepars.shape[1]
    assert (ndim==3 or ndim==4),"ndim should be 3 or 4"

    par_labels=get_par_labels(model, ndim, dolog)

    gmm0=fit_gmix(usepars, ngauss, n_iter, min_covar=min_covar)


    output=zeros(ngauss, dtype=[('means','f8',ndim),
                                ('covars','f8',(ndim,ndim)),
                                ('icovars','f8',(ndim,ndim)),
                                ('weights','f8'),
                                ('norm','f8')])
    output['means']=gmm0.means_
    output['covars']=gmm0.covars_
    output['weights']=gmm0.weights_

    for i in xrange(ngauss):
        output['icovars'][i] = numpy.linalg.inv( output['covars'][i] )

    # make sure our constructore works
    gmm_plot=make_joint_gmm(gmm0.weights_, gmm0.means_, gmm0.covars_)

    samples=gmm_plot.sample(usepars.shape[0]*100)
    plot_fits(usepars, samples, eps=eps, par_labels=par_labels, show=show,
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

_par_labels_bdf_log={}
_par_labels_bdf_log[3] = [r'$log_{10}(T)$',
                          r'$log_{10}(F_b)$',
                          r'$log_{10}(F_d)$']
_par_labels_bdf_log[4] = [r'$|\eta|$',
                          r'$log_{10}(T)$',
                          r'$log_{10}(F_b)$',
                          r'$log_{10}(F_d)$']


_par_labels_sersic_log={}
_par_labels_sersic_log[3] = [r'$log_{10}(T)$',
                             r'$log_{10}(F)$',
                             r'$log_{10}(n)$']



_par_labels_log={}
_par_labels_log[2] = [r'$log_{10}(T)$',
                      r'$log_{10}(F)$']

_par_labels_log[3] = [r'$|\eta|$',
                      r'$log_{10}(T)$',
                      r'$log_{10}(F)$']
_par_labels_log[4] = [r'$|\eta|$',
                      r'$log_{10}(T)$',
                      r'$log_{10}(F_b)$',
                      r'$log_{10}(F_d)$']

_par_labels_lin={}
_par_labels_lin[2] = [r'$T$',
                      r'$F$']

_par_labels_lin[3] = [r'$|g|$',
                      r'$T$',
                      r'$F$']
_par_labels_lin[4] = [r'$|g|$',
                      r'$g_2$',
                      r'$T$',
                      r'$F_b$',
                      r'$F_d$']



def make_combined_pars_subtract_mean_shape(field_list):
    """
    just make the combined pars
    """
    import lensing

    ndim = field_list[0].shape[1]
    ndim_keep=ndim-1

    nobj=0
    for f in field_list:
        nobj += f.shape[0]



    print("nobj:",nobj)
    print("ndim:",ndim)
    print("ndim_keep:",ndim_keep)

    # make pars with mag of ellipticity parameter
    pars = zeros( (nobj, ndim_keep) )

    start=0
    for f in field_list:

        nf = f.shape[0]
        end = start + nf

        print(start,end)

        f[:,0] -= f[:,0].mean()
        f[:,1] -= f[:,1].mean()

        ecomb = sqrt( f[:,0]**2 + f[:,1]**2 )
        pars[start:end, 0] = ecomb
        pars[start:end, 1:] = f[:,2:]

        start += nf

    return pars


def make_logpars_and_subtract_mean_shape(field_list):
    """
    Subtract the mean g1,g2 from each field.

    Store pars as eta1,eta2 and log of T and fluxes
    """
    import lensing

    ndim = field_list[0].shape[1]
    ndim_keep=ndim-1

    nobj=0
    for f in field_list:
        nobj += f.shape[0]

    print("nobj:",nobj)
    print("ndim:",ndim)
    print("ndim_keep:",ndim_keep)

    logpars = zeros( (nobj, ndim_keep) )

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
        eta=sqrt(eta1**2 + eta2**2)

        logpars[start:end, 0] = eta
        logpars[start:end, 1] = log10( f[:,2] )
        logpars[start:end, 2:] = log10( f[:,3:] )

        start += nf

    return logpars

def make_logpars(field_list):
    """
    Make log version of pars
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

        logpars[start:end, :] = log10( f[:,:] )

        start += nf

    return logpars



def fit_gmix(data, ngauss, n_iter, min_covar=MIN_COVAR):
    """
        gtot, T, flux
        etatot, log10(T), log10(flux)
    
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


def plot_fits(pars, samples, dolog=True, show=False, eps=None, par_labels=None):
    """
    """
    import esutil as eu
    import biggles
    import images

    biggles.configure('screen','width', 1400)
    biggles.configure('screen','height', 800)

    num=pars.shape[0]
    ndim=pars.shape[1]

    nrow,ncol = images.get_grid(ndim) 

    tab=biggles.Table(nrow,ncol)


    for dim in xrange(ndim):
        plt = _plot_single(pars[:,dim], samples[:,dim])
        if par_labels is not None:
            plt.xlabel=par_labels[dim]
        else:
            plt.xlabel=r'$P_%s$' % dim

        row=(dim)/ncol
        col=(dim) % ncol

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

    hsum=float(hdict['hist'].sum())

    hist_norm = hdict['hist']/hsum
    hist_norm_err = sqrt(hdict['hist'])/hsum

    hdict['hist_norm'] = hist_norm
    hdict['hist_norm_err'] = hist_norm_err

    return hdict


def fit_g_prior(run, model, **keys):
    """
    Fit only the g prior
    """
    import esutil as eu
    import biggles
    import mcmc

    fl=read_field_list(run, model, **keys)
    comb=make_combined_pars_subtract_mean_shape(fl)

    g=comb[:,0]

    binsize=0.01

    hdict=get_norm_hist(g, min=0, binsize=binsize)

    yrange=[0.0, 1.1*(hdict['hist_norm']+hdict['hist_norm_err']).max()]
    plt=eu.plotting.bscatter(hdict['center'],
                             hdict['hist_norm'],
                             yerr=hdict['hist_norm_err'],
                             yrange=yrange,
                             show=False)

    ivar = ones(hdict['center'].size)
    w,=where(hdict['hist_norm_err'] > 0.0)
    if w.size > 0:
        ivar[w] = 1.0/hdict['hist_norm_err'][w]**2

    bdf=keys.get('bdf',False)
    if bdf:
        print("bdf")
        gpfitter=GPriorFitterAlt(hdict['center'],
                                 hdict['hist_norm'],
                                 ivar)
    else:
        print("exp")
        gpfitter=GPriorFitterExp(hdict['center'],
                                 hdict['hist_norm'],
                                 ivar)

    gpfitter.do_fit()
    gpfitter.print_result()
    res=gpfitter.get_result()

    gvals=numpy.linspace(0.0, 1.0)
    model=gpfitter.get_model_val(res['pars'], g=gvals)
    crv=biggles.Curve(gvals, model, color='red')
    plt.add(crv)

    mcmc.plot_results(gpfitter.trials)
    plt.show()


class GPriorFitterExp(object):
    def __init__(self, xvals, yvals, ivar, nwalkers=100, burnin=1000, nstep=1000, gmax=1.0):
        """
        Fit with gmax fixed
        Input is the histogram data
        """
        self.xvals=xvals
        self.yvals=yvals
        self.ivar=ivar

        self.nwalkers=nwalkers
        self.burnin=burnin
        self.nstep=nstep

        #self.npars=6
        self.npars=3
        self.gmax=gmax

    def get_result(self):
        return self.result

    def do_fit(self):
        import emcee

        print("getting guess")
        guess=self.get_guess()
        sampler = emcee.EnsembleSampler(self.nwalkers, 
                                        self.npars,
                                        self.get_lnprob,
                                        a=2)


        print("burnin:",self.burnin)
        pos, prob, state = sampler.run_mcmc(guess, self.burnin)
        sampler.reset()
        print("steps:",self.nstep)
        pos, prob, state = sampler.run_mcmc(pos, self.nstep)

        self.trials  = sampler.flatchain

        arates = sampler.acceptance_fraction
        self.arate = arates.mean()

        self._calc_result()

    def _calc_result(self):
        import mcmc
        pars,pcov=mcmc.extract_stats(self.trials)

        d=diag(pcov)
        perr = sqrt(d)

        self.result={'arate':self.arate,
                     'A':pars[0],
                     'A_err':perr[0],
                     'a':pars[1],
                     'a_err':perr[1],
                     'g0':pars[2],
                     'g0_err':perr[2],
                     #'s':pars[4],
                     #'s_err':perr[4],
                     #'r':pars[5],
                     #'r_err':perr[5],
                     'pars':pars,
                     'pcov':pcov,
                     'perr':perr}

    def print_result(self):

        fmt="""    A:    %(A).6g +/- %(A_err).6g
    a:    %(a).6g +/- %(a_err).6g
    g0:   %(g0).6g +/- %(g0_err).6g
    gmax: %(gmax).6g +/- %(gmax_err).6g
    s:    %(s).6g +/- %(s_err).6g
    r:    %(r).6g +/- %(r_err).6g\n"""

        fmt="""    arate: %(arate)s
    A:    %(A).6g +/- %(A_err).6g
    a:    %(a).6g +/- %(a_err).6g
    g0:   %(g0).6g +/- %(g0_err).6g\n"""

        print( fmt % self.result )


    def get_guess(self):
        xstep=self.xvals[1]-self.xvals[0]

        self.Aguess = self.yvals.sum()*xstep
        aguess=1.11
        g0guess=0.052
        #s_guess=2.0
        #r_guess=1.0

        pcen=array( [self.Aguess, aguess, g0guess])
        print("pcen:",pcen)

        guess=zeros( (self.nwalkers,self.npars) )
        width=0.1

        nwalkers=self.nwalkers
        guess[:,0] = pcen[0]*(1.+width*srandu(nwalkers))
        guess[:,1] = pcen[1]*(1.+width*srandu(nwalkers))
        guess[:,2] = pcen[2]*(1.+width*srandu(nwalkers))
        #guess[:,3] = pcen[3]*(1.+width*srandu(nwalkers))
        #guess[:,4] = pcen[4]*(1.+width*srandu(nwalkers))
        #guess[:,5] = pcen[5]*(1.+width*srandu(nwalkers))

        return guess


    def get_lnprob(self, pars):
        w,=where(pars < 0)
        if w.size > 0:
            return -9.999e20

        A=pars[0]
        a=pars[1]
        g0=pars[2]

        if a > 1000:
            return -9.999e20

        model=self.get_model_val(pars)

        chi2 = (model - self.yvals)**2
        chi2 *= self.ivar
        lnprob = -0.5*chi2.sum()

        ap = -0.5*( (A-self.Aguess)/(self.Aguess*0.1) )**2

        lnprob += ap

        return lnprob


    def get_model_val(self, pars, g=None):
        from numpy import pi


        A=pars[0]
        a=pars[1]
        g0=pars[2]
        gmax=self.gmax

        if g is None:
            g=self.xvals

        model=zeros(g.size)

        gsq = g**2

        w,=where(g < self.gmax)
        if w.size > 0:
            omgsq=self.gmax-gsq[w]
            omgsq_sq = omgsq[w]*omgsq[w]

            gw=g[w]
            numer = 2*pi*gw*A*(1-exp( (gw-gmax)/a )) * omgsq_sq
            denom = (1+gw)*sqrt(gw**2 + g0**2)

            model[w]=numer/denom


        """
        w,=where(g < gmax)
        if w.size > 0:
            gw=g[w]
            numer = 2*pi*gw*A*(1-exp( (gw-gmax)/a )) * omgsq_sq[w]
            denom = (1+gw)*sqrt(gw**2 + g0**2)

            model[w]=numer/denom
        """
        return model


class GPriorFitterAlt(GPriorFitterExp):
    def __init__(self, xvals, yvals, ivar, nwalkers=100, burnin=1000, nstep=1000, **keys):
        super(GPriorFitterAlt,self).__init__(xvals, yvals, ivar,
                                              nwalkers=100, burnin=1000, nstep=1000, **keys)
        self.npars=3

    def get_guess(self):
        xstep=self.xvals[1]-self.xvals[0]

        self.Aguess = self.yvals.sum()*xstep
        bguess=2.3
        cguess=6.7

        pcen=array( [self.Aguess, bguess, cguess] )
        print("pcen:",pcen)

        guess=zeros( (self.nwalkers,self.npars) )
        width=0.1

        nwalkers=self.nwalkers
        guess[:,0] = pcen[0]*(1.+width*srandu(nwalkers))
        guess[:,1] = pcen[1]*(1.+width*srandu(nwalkers))
        guess[:,2] = pcen[2]*(1.+width*srandu(nwalkers))

        return guess


    def get_lnprob(self, pars):
        w,=where(pars < 0)
        if w.size > 0:
            return -9.999e20


        model=self.get_model_val(pars)

        chi2 = (model - self.yvals)**2
        chi2 *= self.ivar
        lnprob = -0.5*chi2.sum()

        A=pars[0]
        ap = -0.5*( (A-self.Aguess)/(self.Aguess*0.1) )**2
        #ap = -0.5*( (A-self.Aguess)/(self.Aguess) )**2
        #lnprob += ap
        #print(pars)
        #print(lnprob)

        return lnprob


    def get_model_val(self, pars, g=None):
        from numpy import pi


        A=pars[0]
        b=pars[1]
        c=pars[2]

        if g is None:
            g=self.xvals

        gsq=g**2
        model=2*pi*g*A*exp( -b*g - c*gsq )*(1-gsq)**2
        #model=2*pi*g*A*exp( -0.5*gsq/sigma**2 )*(1-gsq)**2

        return model

    def _calc_result(self):
        import mcmc
        pars,pcov=mcmc.extract_stats(self.trials)

        d=diag(pcov)
        perr = sqrt(d)

        self.result={'arate':self.arate,
                     'A':pars[0],
                     'A_err':perr[0],
                     'b':pars[1],
                     'b_err':perr[1],
                     'c':pars[2],
                     'c_err':perr[2],
                     'pars':pars,
                     'pcov':pcov,
                     'perr':perr}

    def print_result(self):

        fmt="""    arate: %(arate)s
    A:     %(A).6g +/- %(A_err).6g
    b:     %(b).6g +/- %(b_err).6g
    c:     %(c).6g +/- %(c_err).6g\n"""

        print( fmt % self.result )




def srandu(num=None):
    """
    Generate random numbers in the symmetric distribution [-1,1]
    """
    return 2*(numpy.random.random(num)-0.5)

