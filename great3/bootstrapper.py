from __future__ import print_function

import numpy
from numpy import array

import ngmix
from .generic import srandu

class Bootstrapper(object):
    def __init__(self, psf_obs, gal_obs, use_logpars=False):
        """
        the observations will be mutated on successful fitting of the psf
        """

        self.psf_obs=psf_obs
        self.gal_obs=gal_obs

        self.use_logpars=use_logpars

        self.model_fits={}

    def get_psf_fitter(self):
        """
        get the fitter for the psf
        """
        if not hasattr(self,'psf_fitter'):
            raise RuntimeError("you need to fit with the psf first")
        return self.psf_fitter

    def get_max_fitter(self):
        """
        get the maxlike fitter for the galaxy
        """
        if not hasattr(self,'max_fitter'):
            raise RuntimeError("you need to run fit_max successfully first")
        return self.max_fitter


    def fit_psf(self, psf_model, Tguess=None, ntry=4):
        """
        fit the psf using a PSFRunner

        TODO: add bootstrapping T guess as well, from unweighted moments
        """
        assert Tguess is not None,"send a Tguess"

        lm_pars={'maxfev': 4000}
        runner=PSFRunner(self.psf_obs, psf_model, Tguess, lm_pars)
        runner.go(ntry=ntry)

        psf_fitter = runner.fitter

        res=psf_fitter.get_result()

        if res['flags']==0:
            self.psf_fitter=psf_fitter
            gmix=self.psf_fitter.get_gmix()

            self.psf_obs.set_gmix(gmix)
            self.gal_obs.set_psf(self.psf_obs)

        else:
            raise PSFFailure("failed to fit psf")

    def fit_max(self, gal_model, pars, prior=None, ntry=1):
        """
        fit the galaxy.  You must run fit_psf() successfully first
        """

        self._fit_gal_psf_flux()

        guesser=self._get_max_guesser(prior=prior)

        runner=MaxRunner(self.gal_obs, gal_model, pars, guesser,
                         prior=prior,
                         use_logpars=self.use_logpars)

        runner.go(ntry=ntry)

        self.max_fitter=runner.fitter

        res=self.max_fitter.get_result()

        if res['flags'] != 0:
            raise GalFailure("failed to fit galaxy")

    def _fit_gal_psf_flux(self):
        """
        use psf as a template, measure flux (linear)
        """
        if not hasattr(self,'psf_fitter'):
            raise RuntimeError("you need to fit with the psf first")

        fitter=ngmix.fitting.TemplateFluxFitter(self.gal_obs, do_psf=True)
        fitter.go()

        res=fitter.get_result()

        self.psf_flux = res['flux']
        self.psf_flux_err = res['flux_err']

        self.psf_flux_fitter=fitter


    def _get_max_guesser(self, prior=None):
        """
        get a guesser that uses the psf T and galaxy psf flux to
        generate a guess, drawing from priors on the other parameters
        """
        from ngmix.guessers import TFluxGuesser, TFluxAndPriorGuesser

        psf_T = self.psf_obs.gmix.get_T()

        psf_flux=self.psf_flux

        if self.use_logpars:
            scaling='log'
        else:
            scaling='linear'

        if prior is None:
            guesser=TFluxGuesser(psf_T,
                                 self.psf_flux,
                                 scaling=scaling)
        else:
            guesser=TFluxAndPriorGuesser(psf_T,
                                         self.psf_flux,
                                         prior,
                                         scaling=scaling)
        return guesser




class PSFRunner(object):
    """
    wrapper to generate guesses and run the psf fitter a few times
    """
    def __init__(self, obs, model, Tguess, lm_pars):
        self.obs=obs

        mess="psf model should be turb or gauss,got '%s'" % model
        assert model in ['turb','gauss'],mess

        self.model=model
        self.lm_pars=lm_pars
        self.set_guess0(Tguess)

    def go(self, ntry=1):
        from ngmix.fitting import LMSimple

        for i in xrange(ntry):
            guess=self.get_guess()
            fitter=LMSimple(self.obs,self.model,lm_pars=self.lm_pars)
            fitter.go(guess)

            res=fitter.get_result()
            if res['flags']==0:
                break

        self.fitter=fitter

    def get_guess(self):
        guess=self.guess0.copy()

        guess[0:0+2] + 0.01*srandu(2)
        guess[2:2+2] + 0.1*srandu(2)
        guess[4] = guess[4]*(1.0 + 0.1*srandu())
        guess[5] = guess[5]*(1.0 + 0.1*srandu())

        return guess

    def set_guess0(self, Tguess):
        Fguess = self.obs.image.sum()
        Fguess *= self.obs.jacobian.get_scale()**2
        self.guess0=array( [0.0, 0.0, 0.0, 0.0, Tguess, Fguess] )


class MaxRunner(object):
    """
    wrapper to generate guesses and run the psf fitter a few times
    """
    def __init__(self, obs, model, pars, guesser, prior=None, use_logpars=False):
        self.obs=obs

        self.pars=pars
        self.method=pars['method']
        if self.method == 'lm':
            self.send_pars=pars['lm_pars']
        else:
            self.send_pars=pars

        mess="model should be exp or dev,got '%s'" % model
        assert model in ['exp','dev'],mess

        self.model=model
        self.prior=prior
        self.use_logpars=use_logpars

        self.bestof = pars.get('bestof',1)

        self.guesser=guesser

    def go(self, ntry=1):
        if self.method=='lm':
            method=self._go_lm
        elif self.method=='nm':
            method=self._go_nm
        else:
            raise ValueError("bad method '%s'" % self.method)

        lnprob_max=-numpy.inf
        fitter_best=None
        for i in xrange(self.bestof):
            method(ntry=ntry)

            res=self.fitter.get_result()
            if res['flags']==0:
                if res['lnprob'] > lnprob_max:
                    lnprob_max = res['lnprob']
                    fitter_best=self.fitter
        
        if fitter_best is not None:
            self.fitter=fitter_best

    def _go_lm(self, ntry=1):
        from ngmix.fitting import LMSimple

        for i in xrange(ntry):
            guess=self.guesser()
            fitter=LMSimple(self.obs,
                            self.model,
                            lm_pars=self.send_pars,
                            use_logpars=self.use_logpars,
                            prior=self.prior)

            fitter.go(guess)

            res=fitter.get_result()
            if res['flags']==0:
                break

        self.fitter=fitter

    def _go_nm(self, ntry=1):
        from ngmix.fitting import MaxSimple

        for i in xrange(ntry):
            guess=self.guesser()
            fitter=MaxSimple(self.obs, self.model,
                             method='Nelder-Mead',
                             use_logpars=self.use_logpars,
                             prior=self.prior)

            fitter.run_max(guess, **self.send_pars)

            res=fitter.get_result()
            if res['flags']==0:
                break

        self.fitter=fitter


