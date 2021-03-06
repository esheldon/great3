"""
"""

from __future__ import print_function

import numpy
from numpy import where, array, sqrt, exp, log, linspace, zeros
from numpy import isfinite, median

import ngmix
from ngmix import Observation
from .generic import srandu, PSFFailure, GalFailure
from ngmix.gexceptions import GMixRangeError


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

    def set_round_s2n(self, prior=None, ntry=4):
        """
        set the s/n and (s/n)_T for the round model

        the s/n measure is stable, the size will require a prior and may
        need retries
        """

        obs=self.gal_obs

        max_fitter=self.get_max_fitter()
        res=max_fitter.get_result()

        pars, pars_lin = self._get_round_pars(res['pars'])

        gm0_round = self._get_gmix_round(res, pars_lin)

        gmpsf_round = obs.psf.gmix.make_round()

        gm_round = gm0_round.convolve(gmpsf_round)

        # first the overall s/n, this is stable
        res['round_pars'] = pars
        res['s2n_r'] = gm_round.get_model_s2n(obs)

        # now the covariance matrix, which can be more unstable
        cov=self._sim_cov_round(obs,
                                gm_round, gmpsf_round,
                                res, pars,
                                prior=prior,
                                ntry=ntry)

        if cov is None:
            print("    failed to fit round (S/N)_T")
            res['T_s2n_r'] = -9999.0
        else:
            if self.use_logpars:
                Ts2n_round = sqrt(1.0/cov[4,4])
            else:
                Ts2n_round = pars_lin[4]/sqrt(cov[4,4])
            res['T_s2n_r'] = Ts2n_round

    def _get_gmix_round(self, res, pars):
        gm_round = ngmix.GMixModel(pars, res['model'])
        return gm_round

    def _sim_cov_round(self,
                       obs,
                       gm_round, gmpsf_round,
                       res, pars_round,
                       prior=None,
                       ntry=4):
        """
        gm_round is convolved
        """
        from numpy.linalg import LinAlgError
        
        im0=gm_round.make_image(obs.image.shape,
                                jacobian=obs.jacobian)
        
        
        # image here is not used
        psf_obs = Observation(im0, gmix=gmpsf_round)

        noise=1.0/sqrt( median(obs.weight) )

        cov=None
        for i in xrange(ntry):
            nim = numpy.random.normal(scale=noise,
                                      size=im0.shape)

            im = im0 + nim
            

            newobs = Observation(im,
                                 weight=obs.weight,
                                 jacobian=obs.get_jacobian(),
                                 psf=psf_obs)
            
            fitter=self._get_round_fitter(newobs, res, prior=prior)

            # we can't recover from this error
            try:
                fitter._setup_data(pars_round)
            except GMixRangeError:
                break

            try:
                tcov=fitter.get_cov(pars_round, 1.0e-3, 5.0)
                if tcov[4,4] > 0:
                    cov=tcov
                    break
            except LinAlgError:
                pass

        return cov

    def _get_round_fitter(self, obs, res, prior=None):
        fitter=ngmix.fitting.LMSimple(obs, res['model'],
                                      prior=prior,
                                      use_logpars=self.use_logpars)
        return fitter

    def _get_round_pars(self, pars_in):
        from ngmix.shape import get_round_factor

        pars=pars_in.copy()
        pars_lin=pars.copy()

        if self.use_logpars:
            pars_lin[4:4+2] = exp(pars[4:4+2])

        g1,g2,T = pars_lin[2],pars_lin[3],pars_lin[4]

        f = get_round_factor(g1, g2)
        T = T*f

        pars[2]=0.0
        pars[3]=0.0
        pars_lin[2]=0.0
        pars_lin[3]=0.0
        pars_lin[4]=T

        if self.use_logpars:
            pars[4] = log(T)
        else:
            pars[4] = T

        return pars, pars_lin


    def get_isampler(self):
        """
        get the importance sampler
        """
        if not hasattr(self,'isampler'):
            raise RuntimeError("you need to run isample() successfully first")
        return self.isampler

    def find_cen(self):
        """
        run a single-gaussian em fit, just to find the center

        Modify the jacobian center accordingly.

        If it fails, don't modify anything
        """
        from ngmix.em import GMixEM, prep_image
        from ngmix import GMix

        jacob=self.gal_obs.jacobian

        row,col=jacob.get_cen()

        guess=array([1.0, # p
                     row, # row in current jacobian coords
                     col, # col in current jacobian coords
                     4.0,
                     0.0,
                     4.0])

        gm_guess = GMix(pars=guess)

        im,sky = prep_image(self.gal_obs.image)
        obs = Observation(im)
        fitter=GMixEM(obs)
        fitter.go(gm_guess, sky, maxiter=4000) 
        res=fitter.get_result()
        if res['flags']==0:
            gm=fitter.get_gmix()
            row,col=gm.get_cen()
            print("        setting jacobian cen to:",row,col,
                  "numiter:",res['numiter'])
            jacob.set_cen(row,col)
        else:
            print("        failed to find cen")



    def fit_psf(self, psf_model, Tguess=None, ntry=4, **keys):
        """
        fit the psf using a PSFRunner or EMRunner

        TODO: add bootstrapping T guess as well, from unweighted moments
        """
        assert Tguess is not None,"send a Tguess"

        if 'em' in psf_model:
            runner=self._fit_psf_em(psf_model, Tguess, ntry, **keys)
        else:
            runner=self._fit_psf_max(psf_model, Tguess, ntry, **keys)

        psf_fitter = runner.fitter
        res=psf_fitter.get_result()

        if res['flags']==0:
            self.psf_fitter=psf_fitter
            gmix=self.psf_fitter.get_gmix()

            self.psf_obs.set_gmix(gmix)
            self.gal_obs.set_psf(self.psf_obs)

        else:
            raise PSFFailure("failed to fit psf")

    def _fit_psf_em(self, psf_model, Tguess, ntry, em_pars=None):
        from .nfit import get_em_ngauss

        ngauss=get_em_ngauss(psf_model)

        if em_pars is None:
            em_pars={'tol': 5.0e-6, 'maxiter': 50000}

        runner=EMRunner(self.psf_obs, Tguess, ngauss, em_pars)
        runner.go(ntry=ntry)

        return runner


    def _fit_psf_max(self, psf_model, Tguess, ntry):
        lm_pars={'maxfev': 4000}
        runner=PSFRunner(self.psf_obs, psf_model, Tguess, lm_pars)
        runner.go(ntry=ntry)

        return runner


    def fit_max(self, gal_model, pars, prior=None, extra_priors=None, ntry=1):
        """
        fit the galaxy.  You must run fit_psf() successfully first

        extra_priors is ignored here but used in composite
        """

        self.max_fitter = self._fit_one_model_max(gal_model,
                                                  pars,
                                                  prior=prior,
                                                  ntry=ntry)
        res=self.max_fitter.get_result()
        res['psf_flux'] = self.psf_flux
        res['psf_flux_err'] = self.psf_flux_err


    def _fit_one_model_max(self, gal_model, pars, prior=None, ntry=1):
        """
        fit the galaxy.  You must run fit_psf() successfully first
        """

        if not hasattr(self,'psf_flux'):
            self.fit_gal_psf_flux()

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

        maxres=max_fitter.get_result()
        tres['model'] = maxres['model']
        tres['psf_flux']=self.psf_flux
        tres['psf_flux_err']=self.psf_flux_err

        if 's2n_r' in maxres:
            tres['s2n_r'] = maxres['s2n_r']
            tres['T_s2n_r'] = maxres['T_s2n_r']

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



    def fit_gal_psf_flux(self):
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

        print("    psf flux: %.3f +/- %.3f" % (res['flux'],res['flux_err']))

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
    def __init__(self, psf_obs, gal_obs,
                 use_logpars=False,
                 fracdev_prior=None,
                 fracdev_grid=None):
        super(CompositeBootstrapper,self).__init__(psf_obs,
                                                   gal_obs,
                                                   use_logpars=use_logpars)
        self.fracdev_prior=fracdev_prior
        #self.fracdev_tests=linspace(-0.5,1.1,17)
        #self.fracdev_tests=linspace(-1.0,1.5,26)
        #self.fracdev_tests=linspace(-1.0,1.1,22)
        if fracdev_grid is not None:
            #print("loading fracdev grid:",fracdev_grid)
            self.fracdev_tests=linspace(fracdev_grid['min'],
                                        fracdev_grid['max'],
                                        fracdev_grid['num'])
        else:
            self.fracdev_tests=linspace(-1.0,1.5,26)

    def fit_max(self,
                model,
                pars,
                prior=None,
                extra_priors=None,
                ntry=1):
        """
        fit the galaxy.  You must run fit_psf() successfully first
        """

        assert model=='cm','model must be cm'
        #assert extra_priors != None,"send extra_priors="
        if extra_priors is None:
            #print("using regular prior for exp and dev")
            exp_prior=prior
            dev_prior=prior
        else:
            exp_prior=extra_priors['exp']
            dev_prior=extra_priors['dev']

        if not hasattr(self,'psf_flux'):
            self.fit_gal_psf_flux()

        print("    fitting exp")
        exp_fitter=self._fit_one_model_max('exp',pars,
                                           prior=exp_prior,ntry=ntry)
        print("    fitting dev")
        dev_fitter=self._fit_one_model_max('dev',pars,
                                           prior=dev_prior,ntry=ntry)

        print("    fitting fracdev")
        use_grid=pars.get('use_fracdev_grid',False)
        fres=self._fit_fracdev(exp_fitter, dev_fitter, use_grid=use_grid)

        fracdev = fres['fracdev']
        fracdev_clipped = self._clip_fracdev(fracdev,pars)

        mess='        nfev: %d fracdev: %.3f +/- %.3f clipped: %.3f'
        print(mess % (fres['nfev'],fracdev,fres['fracdev_err'],fracdev_clipped))


        TdByTe = self._get_TdByTe(exp_fitter, dev_fitter)

        guesser=self._get_max_guesser(prior=prior)

        print("    fitting composite")
        for i in [1,2]:
            try:
                runner=CompositeMaxRunner(self.gal_obs,
                                          pars,
                                          guesser,
                                          fracdev_clipped,
                                          TdByTe,
                                          prior=prior,
                                          use_logpars=self.use_logpars)
                runner.go(ntry=ntry)
                break
            except GMixRangeError:
                #if i==1:
                #    print("caught GMixRange, clipping [-1.0,1.5]")
                #    fracdev_clipped = fracdev_clipped.clip(min=-1.0, max=1.5)
                #elif i==2:
                #    print("caught GMixRange, clipping [ 0.0,1.0]")
                #    fracdev_clipped = fracdev_clipped.clip(min=0.0, max=1.0)
                print("caught GMixRange, clipping [ 0.0,1.0]")
                fracdev_clipped = fracdev_clipped.clip(min=0.0, max=1.0)



        self.max_fitter=runner.fitter

        res=self.max_fitter.get_result()

        if res['flags'] != 0:
            raise GalFailure("failed to fit galaxy with maxlike")

        res['TdByTe'] = TdByTe
        res['fracdev_nfev'] = fres['nfev']
        res['fracdev'] = fracdev_clipped
        res['fracdev_noclip'] = fracdev
        res['fracdev_err'] = fres['fracdev_err']
        res['psf_flux'] = self.psf_flux
        res['psf_flux_err'] = self.psf_flux_err

    def _maybe_clip(self, efitter, dfitter, pars, fracdev):
        """
        allow the user to send a s/n above which the clip
        is applied.  Default is effectively no clipping,
        since fracdev_s2n_clip_min defaults to 1.0e9
        """
        eres=efitter.get_result()
        dres=dfitter.get_result()

        s2n_max=max( eres['s2n_w'], dres['s2n_w'] )

        clip_min = pars.get('fracdev_s2n_clip_min',1.e9)
        if s2n_max > clip_min:
            print("        clipping")
            frange=pars.get('fracdev_range',[-2.0, 2.0])
            fracdev_clipped = fracdev.clip(min=frange[0],max=frange[1])
        else:
            fracdev_clipped = 0.0 + fracdev

        if False and s2n_max > 50:
            import images
            images.multiview(self.gal_obs.image)
            key=raw_input('hit a key: ')
            if key=='q':
                stop
        return fracdev_clipped

    def _clip_fracdev(self, fracdev, pars):
        """
        clip according to parameters
        """
        frange=pars.get('fracdev_range',[-2.0, 2.0])
        fracdev_clipped = fracdev.clip(min=frange[0],max=frange[1])
        return fracdev_clipped


    def _get_gmix_round(self, res, pars):
        gm_round = ngmix.gmix.GMixCM(res['fracdev'],
                                     res['TdByTe'],
                                     pars)
        return gm_round

    def _get_round_fitter(self, obs, res, prior=None):
        fitter=ngmix.fitting.LMComposite(obs,
                                         res['fracdev'],
                                         res['TdByTe'],
                                         prior=prior,
                                         use_logpars=self.use_logpars)

        return fitter



    def isample(self, ipars, prior=None):
        super(CompositeBootstrapper,self).isample(ipars,prior=prior)
        maxres=self.max_fitter.get_result()
        ires=self.isampler.get_result()

        ires['TdByTe']=maxres['TdByTe']
        ires['fracdev']=maxres['fracdev']
        ires['fracdev_noclip']=maxres['fracdev_noclip']
        ires['fracdev_err']=maxres['fracdev_err']
        ires['psf_flux']=maxres['psf_flux']
        ires['psf_flux_err']=maxres['psf_flux_err']
        ires['round_pars']=maxres['round_pars']

    def _fit_fracdev(self, exp_fitter, dev_fitter, use_grid=False):
        from ngmix.fitting import FracdevFitter, FracdevFitterMax

        eres=exp_fitter.get_result()
        dres=dev_fitter.get_result()
        epars=eres['pars']
        dpars=dres['pars']

        #s2n_max=max( eres['s2n_w'], dres['s2n_w'] )
        #print("s2n exp:",eres['s2n_w'],"dev:",dres['s2n_w'])
        #s2n_max=min( eres['s2n_w'], dres['s2n_w'] )

        #if s2n_max > 35.0 or self.fracdev_prior is None:

        fprior=self.fracdev_prior
        if fprior is None:
            ffitter = FracdevFitter(self.gal_obs, epars, dpars,
                                    use_logpars=self.use_logpars)
            res=ffitter.get_result()
        else:

            ffitter = FracdevFitterMax(self.gal_obs, epars, dpars,
                                       use_logpars=self.use_logpars,
                                       prior=fprior)
            if use_grid:
                res=self._fit_fracdev_grid(ffitter)
            else:

                guess=self._get_fracdev_guess(ffitter)

                print("        fracdev guess:",guess)
                if guess is None:
                    raise GalFailure("failed to fit fracdev")

                ffitter.go(guess)

                res=ffitter.get_result()

        if res['flags'] != 0:
            raise GalFailure("failed to fit fracdev")


        self.fracdev_fitter=ffitter
        return res

    def _fit_fracdev_grid(self, ffitter):
        """
        just use the grid
        """
        #print("    fitting fracdev on grid")
        fracdev=self._get_fracdev_guess(ffitter)

        if fracdev is None:
            raise GalFailure("failed to fit fracdev")

        res={'flags':0,
             'fracdev':fracdev,
             'fracdev_err':1.0,
             'nfev':self.fracdev_tests.size}

        return res



    def _get_fracdev_guess(self, fitter):
        tests=self.fracdev_tests
        lnps=zeros(tests.size)
        for i in xrange(tests.size):
            lnps[i] = fitter.calc_lnprob(tests[i:i+1])

        w,=where(isfinite(lnps))
        if w.size == 0:
            return None

        if False:
            from biggles import plot
            plot(tests[w], lnps[w])
            key=raw_input('hit a key: ')

        ibest=lnps[w].argmax()
        guess=tests[w[ibest]]
        return guess

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


class BestBootstrapper(Bootstrapper):
    def __init__(self, psf_obs, gal_obs,
                 use_logpars=False, fracdev_prior=None):
        super(BestBootstrapper,self).__init__(psf_obs,
                                                   gal_obs,
                                                   use_logpars=use_logpars)

    def fit_max(self, exp_prior, dev_prior, exp_rate, pars, ntry=1):
        """
        fit the galaxy.  You must run fit_psf() successfully first
        """

        if not hasattr(self,'psf_flux'):
            self.fit_gal_psf_flux()

        print("    fitting exp")
        exp_fitter=self._fit_one_model_max('exp',pars,prior=exp_prior,ntry=ntry)
        print("    fitting dev")
        dev_fitter=self._fit_one_model_max('dev',pars,prior=dev_prior,ntry=ntry)

        exp_res=exp_fitter.get_result()
        dev_res=dev_fitter.get_result()

        log_exp_rate = log(exp_rate)
        log_dev_rate = log(1.0-exp_rate)
        exp_lnprob = exp_res['lnprob'] + log_exp_rate
        dev_lnprob = dev_res['lnprob'] + log_dev_rate

        if exp_lnprob > dev_lnprob:
            self.max_fitter = exp_fitter
            self.prior=exp_prior
            res=exp_res
        else:
            self.max_fitter = dev_fitter
            self.prior=dev_prior
            res=dev_res

        mess="    exp_lnp: %.6g dev_lnp: %.6g best: %s"
        #mess=mess % (exp(exp_lnprob),exp(dev_lnprob),res['model'])
        mess=mess % (exp_lnprob,dev_lnprob,res['model'])
        print(mess)


    def isample(self, ipars):
        super(BestBootstrapper,self).isample(ipars,prior=self.prior)


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
        elif self.ngauss==4:
            return self._get_em_guess_4gauss()
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

    def _get_em_guess_4gauss(self):
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
                     _em3_fguess[2]*sigma2*(1.0 + 0.1*srandu()),

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
    wrapper to generate guesses and run the fitter a few times
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


