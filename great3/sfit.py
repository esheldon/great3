from __future__ import print_function

from . import files
from .generic import *
from .constants import *
from .nfit import *

class LMFitter(NGMixFitter):
    def _process_object(self, sub_index):
        """
        run fitting
        """
        self.res={'flags':0}

        self.index = self.index_list[sub_index]
        self._set_image_data()

        try:
            self._dofits()
        except PSFFailure:
            self.res['flags'] = PSF_FIT_FAILURE

        self._copy_to_output(sub_index, self.res)

    def _dofits(self):
        boot=Bootstrapper(self.psf_obs,
                          self.gal_obs,
                          use_logpars=True)

        sigma_guess=self.conf['psf_fwhm_guess']/2.35
        Tguess=2*sigma_guess**2
        boot.fit_psf(self.conf['psf_model'],
                     Tguess=Tguess,
                     ntry=self.conf['psf_ntry'])

        # this is a copy
        self.res['psf_gmix'] = boot.psf_obs.get_gmix()
        self._print_psf_res()

        if self.make_plots:
            self._compare_psf(boot.psf_fitter, self.conf['psf_model'])

        max_pars=self.conf['max_pars']
        for model in self.conf['model_pars']:
            try:

                prior=self.priors[model]
                boot.fit_max(model,
                             max_pars,
                             prior=prior,
                             ntry=max_pars['ntry'])

                fitter=boot.get_max_fitter()
                self.res[model] = {'fitter':fitter,
                                   'res':fitter.get_result()}

                self._print_galaxy_res(model)

                if self.make_plots:
                    self._do_gal_plots(model, fitter)


            except GalFailure:
                print("failed to fit galaxy with model: %s" % model)
                self.res['flags'] = 2**(i+1)

    def _set_image_data(self):
        """
        Get all the data we need to do our processing
        """
        gal_image, gal_cen_guess = \
                self.field.get_gal_image(self.index)

        weight_image = 0*gal_image + self.sky_ivar

        if self.conf['use_random_psf']:
            rint=numpy.random.randint(9)
            print("    random psf:",rint)
            psf_image, psf_cen_guess = \
                    self.field.get_star_image(rint)
        else:
            # otherwise we just use the first one. Make
            # sure you have parameters set so that you really
            # fit it well
            if not hasattr(self,'_psf_image'):
                self._psf_image,self._psf_cen_guess = \
                        self.field.get_star_image(0)
            psf_image=self._psf_image
            psf_cen_guess=self._psf_cen_guess

        psf_jacob=self._get_jacobian(psf_cen_guess)
        self.psf_obs=Observation(psf_image,
                                 jacobian=psf_jacob)

        gal_jacob=self._get_jacobian(gal_cen_guess)
        self.gal_obs=Observation(gal_image,
                                 weight=weight_image,
                                 jacobian=gal_jacob)


class ISampleFitter(LMFitter):
    def _dofits(self):

        boot=Bootstrapper(self.psf_obs,
                          self.gal_obs,
                          use_logpars=True)

        sigma_guess=self.conf['psf_fwhm_guess']/2.35
        Tguess=2*sigma_guess**2
        boot.fit_psf(self.conf['psf_model'],
                     Tguess=Tguess,
                     ntry=self.conf['psf_ntry'])

        # this is a copy
        self.res['psf_gmix'] = boot.psf_obs.get_gmix()
        self._print_psf_res()

        if self.make_plots:
            self._compare_psf(boot.psf_fitter, self.conf['psf_model'])

        max_pars=self.conf['max_pars']
        ipars=self.conf['isample_pars']

        models=list(self.conf['model_pars'].keys())
        for i,model in enumerate(models):
            try:

                prior=self.priors[model]
                boot.fit_max(model,
                             max_pars,
                             prior=prior,
                             ntry=max_pars['ntry'])

                boot.isample(model,
                             ipars,
                             prior=prior)

                sampler=boot.get_isampler()
                max_fitter=boot.get_max_fitter()
                self._add_shear_info(sampler, max_fitter, model)

                self.res[model] = {'fitter':sampler,
                                   'res':sampler.get_result()}
                
                self._print_galaxy_res(model)

                if self.make_plots:
                    self._compare_gal(model, boot.max_fitter)
                    self._make_trials_plot(model, sampler)

            except GalFailure:
                print("failed to fit galaxy with model: %s" % model)
                self.res['flags'] = 2**(i+1)

    def _add_shear_info(self, sampler, max_fitter, model):
        """
        lensfit and pqr

        calc result *before* calling this method
        """

        # this is the full prior
        prior=self.priors[model]
        g_prior=prior.g_prior

        iweights = sampler.get_iweights()
        samples = sampler.get_samples()
        g_vals=samples[:,2:2+2]

        res=sampler.get_result()

        # keep for later if we want to make plots
        self.weights=iweights

        # we are going to mutate the result dict owned by the sampler
        stats = max_fitter.get_fit_stats(res['pars'])
        res.update(stats)

        ls=ngmix.lensfit.LensfitSensitivity(g_vals,
                                            g_prior,
                                            weights=iweights,
                                            remove_prior=True)
        g_sens = ls.get_g_sens()
        g_mean = ls.get_g_mean()

        res['g_sens'] = g_sens
        res['nuse'] = ls.get_nuse()

        # not able to use extra weights yet
        '''
        pqrobj=ngmix.pqr.PQR(g, g_prior,
                             shear_expand=self.shear_expand,
                             remove_prior=remove_prior)


        P,Q,R = pqrobj.get_pqr()
        res['P']=P
        res['Q']=Q
        res['R']=R
        '''

    def _make_trials_plot(self, model, fitter):
        """
        Plot the trials
        """
        width,height=800,800
        pdict=fitter.make_plots(title=model,
                                weights=fitter.get_iweights(),
                                nsigma=4)


        trials_pname='trials-%06d-%s.png' % (self.index,model)
        print("          ",trials_pname)
        pdict['trials'].write_img(width,height,trials_pname)

        wtrials_pname='wtrials-%06d-%s.png' % (self.index,model)
        print("          ",wtrials_pname)
        pdict['wtrials'].write_img(width,height,wtrials_pname)