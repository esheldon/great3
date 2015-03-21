from __future__ import print_function

import numpy
from numpy import array, sqrt, exp, log

import ngmix
from ngmix import Observation
from .generic import srandu, PSFFailure, GalFailure


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

    def get_isampler(self):
        """
        get the importance sampler
        """
        if not hasattr(self,'isampler'):
            raise RuntimeError("you need to run isample() successfully first")
        return self.isampler


    def fit_psf(self, psf_model, Tguess=None, ntry=4):
        """
        fit the psf using a PSFRunner or EMRunner

        TODO: add bootstrapping T guess as well, from unweighted moments
        """
        assert Tguess is not None,"send a Tguess"

        if 'em' in psf_model:
            runner=self._fit_psf_em(psf_model, Tguess, ntry)
        else:
            runner=self._fit_psf_max(psf_model, Tguess, ntry)

        psf_fitter = runner.fitter
        res=psf_fitter.get_result()

        if res['flags']==0:
            self.psf_fitter=psf_fitter
            gmix=self.psf_fitter.get_gmix()

            self.psf_obs.set_gmix(gmix)
            self.gal_obs.set_psf(self.psf_obs)

        else:
            raise PSFFailure("failed to fit psf")

    def _fit_psf_em(self, psf_model, Tguess, ntry):
        from .nfit import get_em_ngauss

        ngauss=get_em_ngauss(psf_model)
        em_pars={'tol': 5.0e-6, 'maxiter': 50000}

        runner=EMRunner(self.psf_obs, Tguess, ngauss, em_pars)
        runner.go(ntry=ntry)

        return runner


    def _fit_psf_max(self, psf_model, Tguess, ntry):
        lm_pars={'maxfev': 4000}
        runner=PSFRunner(self.psf_obs, psf_model, Tguess, lm_pars)
        runner.go(ntry=ntry)

        return runner


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
            raise GalFailure("failed to fit galaxy with maxlike")

    def isample(self, ipars, prior=None):
        """
        bootstrap off the maxlike run
        """

        self._try_replace_cov(ipars['cov_pars'])

        max_fitter=self.max_fitter
        use_fitter=max_fitter

        niter=len(ipars['nsample'])
        for i,nsample in enumerate(ipars['nsample']):
            sampler=self._make_sampler(use_fitter, ipars)
            if sampler is None:
                raise GalFailure("isampling failed")

            sampler.make_samples(nsample)

            sampler.set_iweights(max_fitter.calc_lnprob)
            sampler.calc_result()

            tres=sampler.get_result()

            print("    eff iter %d: %.2f" % (i,tres['efficiency']))
            use_fitter = sampler

        self.isampler=sampler

    def _make_sampler(self, fitter, ipars):
        from ngmix.fitting import ISampler
        from numpy.linalg import LinAlgError

        res=fitter.get_result()
        icov = res['pars_cov']*ipars['ifactor']**2

        try:
            sampler=ISampler(res['pars'],
                             icov,
                             ipars['df'],
                             min_err=ipars['min_err'],
                             max_err=ipars['max_err'])
        except LinAlgError:
            print("        bad cov")
            sampler=None

        return sampler



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



    def _try_replace_cov(self, cov_pars):
        """
        the lm cov sucks, try to replace it
        """
        if not hasattr(self,'max_fitter'):
            raise RuntimeError("you need to fit with the max like first")

        fitter=self.max_fitter

        # reference to res
        res=fitter.get_result()

        print("        replacing cov")
        fitter.calc_cov(cov_pars['h'],cov_pars['m'])

        if res['flags'] != 0:
            print("        replacement failed")
            res['flags']=0


class CompositeBootstrapper(Bootstrapper):
    def fit_max(self, model, pars, prior=None, ntry=1):
        """
        fit the galaxy.  You must run fit_psf() successfully first
        """

        assert model=='cm','model must be cm'

        self._fit_gal_psf_flux()

        print("    fitting exp")
        exp_fitter=self._fit_one_model_max('exp',pars,prior=prior,ntry=ntry)
        print("    fitting dev")
        dev_fitter=self._fit_one_model_max('dev',pars,prior=prior,ntry=ntry)

        print("    fitting fracdev")
        fres=self._fit_fracdev(exp_fitter, dev_fitter, ntry=ntry)
        fracdev = fres['fracdev']

        TdByTe = self._get_TdByTe(exp_fitter, dev_fitter)

        guesser=self._get_max_guesser(prior=prior)

        print("    fitting composite")
        runner=CompositeMaxRunner(self.gal_obs,
                                  pars,
                                  guesser,
                                  fracdev,
                                  TdByTe,
                                  prior=prior,
                                  use_logpars=self.use_logpars)

        runner.go(ntry=ntry)

        self.max_fitter=runner.fitter

        res=self.max_fitter.get_result()

        if res['flags'] != 0:
            raise GalFailure("failed to fit galaxy with maxlike")

        res['TdByTe'] = TdByTe
        res['fracdev'] = fres['fracdev']
        res['fracdev_err'] = fres['fracdev_err']

    def isample(self, ipars, prior=None):
        super(CompositeBootstrapper,self).isample(ipars,prior=prior)
        maxres=self.max_fitter.get_result()
        ires=self.isampler.get_result()

        ires['TdByTe']=maxres['TdByTe']
        ires['fracdev']=maxres['fracdev']
        ires['fracdev_err']=maxres['fracdev_err']

    def _fit_one_model_max(self, gal_model, pars, prior=None, ntry=1):
        """
        fit the galaxy.  You must run fit_psf() successfully first
        """

        guesser=self._get_max_guesser(prior=prior)

        runner=MaxRunner(self.gal_obs, gal_model, pars, guesser,
                         prior=prior,
                         use_logpars=self.use_logpars)

        runner.go(ntry=ntry)

        fitter=runner.fitter

        res=fitter.get_result()

        if res['flags'] != 0:
            raise GalFailure("failed to fit galaxy with maxlike")

        return fitter

    def _fit_fracdev(self, exp_fitter, dev_fitter, ntry=1):
        from ngmix.fitting import FracdevFitter

        epars=exp_fitter.get_result()['pars']
        dpars=dev_fitter.get_result()['pars']

        ffitter = FracdevFitter(self.gal_obs, epars, dpars,
                                use_logpars=self.use_logpars)
        res=ffitter.get_result()

        if res['flags'] != 0:
            raise GalFailure("failed to fit fracdev")

        mess='        fracdev: %(fracdev).3f +/- %(fracdev_err).3f'
        mess = mess % res
        print(mess)

        self.fracdev_fitter=ffitter
        return res


    def _get_TdByTe(self, exp_fitter, dev_fitter):
        epars=exp_fitter.get_result()['pars']
        dpars=dev_fitter.get_result()['pars']

        if self.use_logpars:
            Te = exp(epars[4])
            Td = exp(dpars[4])
        else:
            Te = epars[4]
            Td = dpars[4]
        TdByTe = Td/Te

        print('        Td/Te: %.3f' % TdByTe)
        return TdByTe


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

class EMRunner(object):
    """
    wrapper to generate guesses and run the psf fitter a few times
    """
    def __init__(self, obs, Tguess, ngauss, em_pars):

        self.ngauss = ngauss
        self.Tguess = Tguess
        self.sigma_guess = sqrt(Tguess/2)
        self.set_obs(obs)

        self.em_pars=em_pars

    def set_obs(self, obsin):
        """
        set a new observation with sky
        """
        im_with_sky, sky = ngmix.em.prep_image(obsin.image)

        self.obs   = Observation(im_with_sky, jacobian=obsin.jacobian)
        self.sky   = sky


    def go(self, ntry=1):

        fitter=ngmix.em.GMixEM(self.obs)
        for i in xrange(ntry):
            guess=self.get_guess()

            fitter.go(guess, self.sky, **self.em_pars)

            res=fitter.get_result()
            if res['flags']==0:
                break

        self.fitter=fitter

    def get_guess(self):
        """
        Guess for the EM algorithm
        """

        if self.ngauss==1:
            return self._get_em_guess_1gauss()
        elif self.ngauss==2:
            return self._get_em_guess_2gauss()
        elif self.ngauss==3:
            return self._get_em_guess_3gauss()
        else:
            raise ValueError("bad ngauss: %d" % self.ngauss)

    def _get_em_guess_1gauss(self):

        sigma2 = self.sigma_guess**2
        pars=array( [1.0 + 0.1*srandu(),
                     0.1*srandu(),
                     0.1*srandu(), 
                     sigma2*(1.0 + 0.1*srandu()),
                     0.2*sigma2*srandu(),
                     sigma2*(1.0 + 0.1*srandu())] )

        return ngmix.gmix.GMix(pars=pars)

    def _get_em_guess_2gauss(self):
        from .nfit import _em2_pguess, _em2_fguess

        sigma2 = self.sigma_guess**2

        pars=array( [_em2_pguess[0],
                     0.1*srandu(),
                     0.1*srandu(),
                     _em2_fguess[0]*sigma2*(1.0 + 0.1*srandu()),
                     0.0,
                     _em2_fguess[0]*sigma2*(1.0 + 0.1*srandu()),

                     _em2_pguess[1],
                     0.1*srandu(),
                     0.1*srandu(),
                     _em2_fguess[1]*sigma2*(1.0 + 0.1*srandu()),
                     0.0,
                     _em2_fguess[1]*sigma2*(1.0 + 0.1*srandu())] )


        return ngmix.gmix.GMix(pars=pars)

    def _get_em_guess_3gauss(self):
        from .nfit import _em3_pguess, _em3_fguess

        sigma2 = self.sigma_guess**2

        pars=array( [_em3_pguess[0]*(1.0+0.1*srandu()),
                     0.1*srandu(),
                     0.1*srandu(),
                     _em3_fguess[0]*sigma2*(1.0 + 0.1*srandu()),
                     0.01*srandu(),
                     _em3_fguess[0]*sigma2*(1.0 + 0.1*srandu()),

                     _em3_pguess[1]*(1.0+0.1*srandu()),
                     0.1*srandu(),
                     0.1*srandu(),
                     _em3_fguess[1]*sigma2*(1.0 + 0.1*srandu()),
                     0.01*srandu(),
                     _em3_fguess[1]*sigma2*(1.0 + 0.1*srandu()),

                     _em3_pguess[2]*(1.0+0.1*srandu()),
                     0.1*srandu(),
                     0.1*srandu(),
                     _em3_fguess[2]*sigma2*(1.0 + 0.1*srandu()),
                     0.01*srandu(),
                     _em3_fguess[2]*sigma2*(1.0 + 0.1*srandu())]

                  )


        return ngmix.gmix.GMix(pars=pars)



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

class CompositeMaxRunner(MaxRunner):
    """
    wrapper to generate guesses and run the psf fitter a few times
    """
    def __init__(self, obs, pars, guesser, fracdev, TdByTe,
                 prior=None, use_logpars=False):
        self.obs=obs

        self.pars=pars
        self.fracdev=fracdev
        self.TdByTe=TdByTe

        self.method=pars['method']
        if self.method == 'lm':
            self.send_pars=pars['lm_pars']
        else:
            self.send_pars=pars

        self.prior=prior
        self.use_logpars=use_logpars

        self.bestof = pars.get('bestof',1)

        self.guesser=guesser

    def _go_lm(self, ntry=1):
        from ngmix.fitting import LMComposite

        for i in xrange(ntry):
            guess=self.guesser()
            fitter=LMComposite(self.obs,
                               self.fracdev,
                               self.TdByTe,
                               lm_pars=self.send_pars,
                               use_logpars=self.use_logpars,
                               prior=self.prior)

            fitter.go(guess)

            res=fitter.get_result()
            if res['flags']==0:
                break

        self.fitter=fitter

