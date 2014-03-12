from __future__ import print_function
import numpy

from .generic import *
from .constants import *

import admom


class RGFitter(object):

    def _process_object(self, sub_index):
        """
        run re-gauss
        """
        index = self.index_list[sub_index]

        gal_image,gal_cen = self.fields.get_gal_image(index)
        psf_image,psf_cen = self.fields.get_star_image(index)

        res=self._run_regauss(gal_image, gal_cen,
                              psf_image, psf_cen)
        return res

    def _run_regauss(self,
                     gal_image, gal_cen,
                     psf_image, psf_cen):

        ntry=self.ntry
        for i in xrange(ntry):

            cen_guess, psf_cen_guess, psf_irr_guess, gal_irr_guess = \
                    self._get_guesses(gal_cen, psf_cen)

            rg = admom.ReGauss(gal_image,
                               cen_guess[0],
                               cen_guess[1],
                               psf_image,
                               psf_cen_guess[0],
                               psf_cen_guess[1],
                               guess_psf=psf_irr_guess,
                               guess=gal_irr_guess,
                               sigsky=self.skysig)
            rg.do_all()

            res = rg['rgcorrstats']
            if res is not None and res['flags'] == 0:
                # error accounting for the 1/R scaling
                res['err_corr'] = rg['rgstats']['uncer']/R
                res['flags'] = 0
                break
        
        if res is None:
            print("    regauss failed")
            res={'flags':RG_FAILURE}

        return res

    def _get_guesses(self, gal_cen, psf_cen):
        cen_guess = gal_cen + self.guess_width_cen*srandu(2)
        psf_cen_guess = psf_cen + self.guess_width_cen*srandu(2)

        psf_irr_guess = self.guess_psf_irr*(1.0+0.01*srandu())
        # guess gal bigger
        gal_irr_guess = 1.4*self.guess_psf_irr*(1.0+0.01*srandu())

        return cen_guess, psf_cen_guess, psf_irr_guess, gal_irr_guess

    def print_res(self,res):
        """
        Print some stats from the fit
        """
        mess='    %(e1).6g %(e2).6g +/- %(err).6g'
        mess = mess % res
        print(mess)

    def _finish_setup(self):
        """
        Process the rest of the input
        """

        conf=self.conf

        # pixels
        self.guess_width_cen = conf.get('guess_width_cen',0.1)

        # the central guess for the PSF FWHM in arcsec
        guess_psf_fwhm  = conf.get('guess_psf_fwhm',0.9)
        guess_psf_fwhm = guess_psf_fwhm/PIXEL_SCALE

        # convert from fwhm to sigma, then square and double
        self.guess_psf_irr = (guess_psf_fwhm/2.35)**2

        # how many times to retry the fit
        self.ntry = conf.get('ntry',5)

    def _copy_to_output(self, sub_index, res):
        """
        Copy to the output structure
        """
        data=self.data

        data['flags'][sub_index] = res['flags']

        if res['flags']==0:
            data['e1'][sub_index]  = res['e1']
            data['e2'][sub_index]  = res['e2']
            data['err'][sub_index] = res['err_corr']
            data['R'][sub_index]   = res['R']

    def _make_struct(self):
        """
        Make the output structure
        """

        num=self.index_list.size

        dt = self._get_default_dtype()
        
        dt += [('e1','f8'),
               ('e2','f8'),
               ('err','f8'),
               ('R','f8')]

        data=numpy.zeros(num, dtype=dt)

        # from default dtype.
        data['flags'] = NO_ATTEMPT

        data['e1']  = DEFVAL
        data['e2']  = DEFVAL
        data['err'] = PDEFVAL
        data['R']   = DEFVAL

        self.data=data

