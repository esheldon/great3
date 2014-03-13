from __future__ import print_function
import numpy
from numpy.random import random as randu

from . import files
from .generic import *
from .constants import *

class PSFFailure(Exception):
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)


class NGMixFitter(FitterBase):
    def _process_object(self, sub_index):
        """
        run B&A fitting
        """
        index = self.index_list[sub_index]

        gal_image,gal_cen = self.field.get_gal_image(index)
        rint=numpy.random.randint(9)
        psf_image,psf_cen = self.field.get_star_image(rint)

        try:
            psf_fitter = self._fit_psf(psf_image, psf_cen)
        except PSFFailure:
            print("psf failure at",sub_index)
            self.data['flags'][sub_index] = PSF_FIT_FAILURE
            return

        psf_gmix=psf_fitter.get_gmix()
        stop
        self._copy_psf_info(sub_index, psf_gmix)

        gal_res = self._fit_galaxy(gal_image, gal_cen, psf_gmix)

    def _fit_psf(self, psf_image, psf_cen): 
        """
        Fit the psf image
        """
        
        conf=self.conf
        sigma_guess = conf['psf_fwhm_guess']/2.35

        model=conf['psf_model']
        if 'em' in model:
            ngauss=_em_ngauss_map[model]
            if ngauss==1:
                fitter=_fit_em_1gauss(psf_image, psf_cen, sigma_guess)
            else:
                fitter=_fit_em_2gauss(psf_image, psf_cen, sigma_guess)

        else:
            raise ValueError("unsupported psf model: '%s'" % model)

        return fitter

    def _fit_em_1gauss(self, im, cen, sigma_guess):
        """
        Just run the fitter
        """
        return _fit_with_em(self, im, cen, sigma_guess, 1)

    def _fit_em_2gauss(self, im, cen, sigma_guess):
        """
        First fit 1 gauss and use it for guess
        """
        fitter1=_fit_with_em(self, im, cen, sigma_guess, 1)

        gmix=fitter1.get_gmix()
        sigma_guess_new = sqrt( gmix.get_T()/2. )

        fitter2=_fit_with_em(self, im, cen, sigma_guess_new, 2)

        return fitter2

    def _fit_with_em(self, im, cen, sigma_guess, ngauss):
        """
        Fit the image using EM
        """

        if ngauss <= 0 or ngauss > 2:
            raise ValueError("unsupported em ngauss: %d" % ngauss)

        conf=self.conf

        im_with_sky, sky = ngmix.em.prep_image(im)
        jacob = self._get_jacobian(cen)

        ntry=conf['psf_ntry']

        for i in xrange(ntry):
            guess = self._get_em_guess(sigma, ngauss)
            try:
                fitter=self._do_fit_em_with_full_guess(im_with_sky,
                                                       sky,
                                                       guess,
                                                       jacob)
                break
            except GMixMaxIterEM:
                fitter=None

        return fitter

    def _do_fit_em_with_full_guess(self,
                                   image,
                                   sky,
                                   guess,
                                   jacob):

        maxiter=conf['psf_maxiter']
        tol=conf['psf_tol']
        fitter=ngmix.em.GMixEM(image, jacobian=jacob)
        fitter.go(guess, sky, maxiter=maxiter, tol=tol)

        return fitter


    def _get_em_guess(self, sigma, ngauss):
        """
        Guess for the EM algorithm
        """

        if ngauss==1:
            return self._get_em_guess_1gauss(sigma)
        elif ngauss==2:
            return self._get_em_guess_2gauss(sigma)
        else:
            raise ValueError("1 or 2 em gauss")

    def _get_em_guess_1gauss(self, sigma):
        import ngmix
        from ngmix import srandu

        sigma2 = sigma**2
        pars=numpy.array( [1.0 + 0.01*srandu(),
                           0.1*srandu(),
                           0.1*srandu(), 
                           sigma2*(1.0 + 0.1*srandu()),
                           0.01*srandu(),
                           sigma2*(1.0 + 0.1*srandu())] )

        return ngmix.gmix.GMix(pars=pars)

    def _get_em_guess_2gauss(self, sigma):
        import ngmix
        from ngmix import srandu

        sigma2 = sigma**2

        pars=numpy.array( [_em2_pguess[0],
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


    def _fit_galaxy(self, gal_image, gal_cen, psf_gmix):
        pass

    def _finish_setup(self):
        """
        Process the rest of the input
        """

        conf=self.conf
        self._unpack_priors()

    def _get_jacobian(self, cen):
        """
        Get a simple jacobian at the specified location
        """
        import ngmix

        j = ngmix.jacobian.Jacobian(cen[0],
                                    cen[1],
                                    PIXEL_SCALE,
                                    0.0,
                                    0.0,
                                    PIXEL_SCALE)
        return j

    def _unpack_priors(self):
        conf=self.conf

        nmod=len(self.simple_models)

        self.cen_prior=get_cen_prior(conf)

        T_priors=get_T_priors(conf)
        counts_priors=get_counts_priors(conf)
        g_priors=get_g_priors(conf)

        if (len(T_priors) != nmod
                or len(g_priors) != nmod
                or len(g_priors) != nmod):
            raise ValueError("models and T,counts,g priors must be same length")

        priors={}
        models=self.simple_models
        for i in xrange(nmod):
            model=models[i]

            T_prior=T_priors[i]

            # note it is a list
            counts_prior=[ counts_priors[i] ]

            g_prior=g_priors[i]
            
            modlist={'T':T_prior, 'counts':counts_prior,'g':g_prior}
            priors[model] = modlist

        self.priors=priors
        self.draw_g_prior=conf.get('draw_g_prior',True)


def get_T_priors(conf):
    import ngmix

    T_prior_types=conf['T_prior_types']

    T_priors=[]
    for i,typ in enumerate(T_prior_types):
        if typ == 'flat':
            pars=conf['T_prior_pars'][i]
            T_prior=ngmix.priors.FlatPrior(pars[0], pars[1])
        elif typ =='lognormal':
            pars=conf['T_prior_pars'][i]
            T_prior=ngmix.priors.LogNormal(pars[0], pars[1])
        elif typ=="cosmos_exp":
            T_prior=ngmix.priors.TPriorCosmosExp()
        elif typ=="cosmos_dev":
            T_prior=ngmix.priors.TPriorCosmosDev()
        else:
            raise ValueError("bad T prior type: %s" % T_prior_type)

        T_priors.append(T_prior)

    return T_priors

def get_counts_priors(conf):
    import ngmix

    counts_prior_types=conf['counts_prior_types']

    counts_priors=[]
    for i,typ in enumerate(counts_prior_types):
        if typ == 'flat':
            pars=conf['counts_prior_pars'][i]
            counts_prior=ngmix.priors.FlatPrior(pars[0], pars[1])
        else:
            raise ValueError("bad counts prior type: %s" % counts_prior_type)

        counts_priors.append(counts_prior)

    return counts_priors


def get_g_priors(conf):
    import ngmix
    g_prior_types=conf['g_prior_types']

    g_priors=[]
    for i,typ in enumerate(g_prior_types):
        if typ =='exp':
            pars=conf['g_prior_pars'][i]
            parr=numpy.array(pars,dtype='f8')
            g_prior = ngmix.priors.GPriorM(parr)
        elif typ=='cosmos-galfit':
            g_prior = ngmix.priors.make_gprior_cosmos_galfit()
        elif typ=='cosmos-exp':
            g_prior = ngmix.priors.make_gprior_cosmos_exp()
        elif typ=='cosmos-dev':
            g_prior = ngmix.priors.make_gprior_cosmos_dev()
        elif typ =='ba':
            sigma=conf['g_prior_pars'][i]
            g_prior = ngmix.priors.GPriorBA(sigma)
        else:
            raise ValueError("implement gprior '%s'")
        g_priors.append(g_prior)

    return g_priors

def get_cen_prior(conf):
    import ngmix
    use_cen_prior=conf.get('use_cen_prior',False)
    if use_cen_prior:
        width=conf.get('cen_width',1.0)
        return ngmix.priors.CenPrior(0.0, 0.0, width,width)
    else:
        return None

_em2_fguess=numpy.array([0.5793612389470884,1.621860687127999])
_em2_pguess=numpy.array([0.596510042804182,0.4034898268889178])
#_em2_fguess=numpy.array([12.6,3.8])
#_em2_fguess[:] /= _em2_fguess.sum()
#_em2_pguess=numpy.array([0.30, 0.70])
_em_ngauss_map = {'em1':1, 'em2':2}

