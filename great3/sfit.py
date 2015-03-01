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


