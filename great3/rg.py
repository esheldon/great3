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

    def _run_regauss(self,
                     gal_image, gal_cen,
                     psf_image, psf_cen):

        ntry=self.ntry
        for i in xrange(ntry):
            cen_guess = gal_cen + self.guess_width_cen*srandu(2)
            psf_cen_guess = psf_cen + self.guess_width_cen*srandu(2)

            psf_irr_guess = self.guess_psf_irr*(1.0+0.01*srandu())
            # guess gal bigger
            gal_irr_guess = 1.4*self.guess_psf_irr*(1.0+0.01*srandu())

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
                e1,e2,R=res['e1'],res['e2'],res['R']
                if (abs(e1) < RG_MAX_ELLIP
                        and abs(e2) < RG_MAX_ELLIP
                        and R > RG_MIN_R
                        and R <  RG_MAX_R):

                res['err'] = rg['rgstats']['uncer']/R
                break
        
        if res is None:
            res={'flags':1}

        if res['flags'] == 0: 
            res['err']
            res['err']=rg['rgstats']['uncer']/R
            e1,e2,R=res['e1'],res['e2'],res['R']
            if (abs(e1) > MAX_ELLIP
                    or abs(e2) > MAX_ELLIP
                    or R <= MIN_R
                    or R >  MAX_R):
                res['flags']=1
            else:
                # err is per component.  Boost by 1/R
                res['err']=rg['rgstats']['uncer']/R
                weight,ssh=self._get_weight_and_ssh(res['e1'],
                                                    res['e2'],
                                                    res['err'])

                res['weight'] = weight
                res['ssh'] = ssh

        return res


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

    def _make_struct(self):
        """
        Make the output structure
        """

        num=self.index_list.size

        dt = self._get_default_dtype()
        
        dt += [('e1','f8'),
               ('e2','f8'),
               ('err','f8'),
               ('ssh','f8'),
               ('R','f8'),
               ('weight','f8'),
               ('rg_flags','i4')]

        data=numpy.zeros(num, dtype=dt)

        # from default dtype.
        data['flags'] = NO_ATTEMPT

        data['e1']  = DEFVAL
        data['e2']  = DEFVAL
        data['err'] = PDEFVAL
        data['ssh'] = DEFVAL
        data['R']   = DEFVAL
        data['weight']   = 0.0
        data['rg_flags'] = NO_ATTEMPT

        self.data=data

