from __future__ import print_function
import numpy
from numpy import sqrt, array, zeros, log10
from numpy.random import random as randu
from pprint import pprint

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
        self.res={'flags':0}

        self.index = self.index_list[sub_index]
        self._set_image_data()

        self._fit_psf()
        if self.res['flags'] != 0:
            return res

        self._fit_galaxy()

        self._copy_to_output(sub_index, self.res)

    def _set_image_data(self):
        """
        Get all the data we need to do our processing
        """
        self.gal_image,self.gal_cen_guess = \
                self.field.get_gal_image(self.index)

        self.weight_image = 0*self.gal_image + self.sky_ivar

        if self.conf['use_random_psf']:
            rint=numpy.random.randint(9)
            self.psf_image,self.psf_cen_guess = \
                    self.field.get_star_image(rint)
        else:
            # otherwise we just use the first one. Make
            # sure you have parameters set so that you really
            # fit it well
            if not hasattr(self,'_psf_image'):
                self.psf_image,self.psf_cen_guess = \
                        self.field.get_star_image(0)


    def _fit_psf(self): 
        """
        Fit the psf image
        """
        
        self.fitting_galaxy=False

        # if not using a random psf, just do the fit once
        if not self.conf['use_random_psf']:
            if hasattr(self,'psf_gmix'):
                print("    re-using psf fit")
                self.res['psf_gmix1']=self.psf_gmix1
                self.res['psf_gmix']=self.psf_gmix
                return

        conf=self.conf
        sigma_guess = conf['psf_fwhm_guess']/2.35

        fitter1=self._fit_em_1gauss(self.psf_image,
                                    self.psf_cen_guess,
                                    sigma_guess)
        if fitter1 is None:
            self.res['flags'] = PSF_FIT_FAILURE
            print("psf em1 failure at object",index)
            return

        psf_gmix1=fitter1.get_gmix()
        self.psf_gmix1=psf_gmix1
        self.res['psf_gmix_em1']=psf_gmix1

        model=conf['psf_model']
        if 'em' in model:
            # update the fiducial
            cen_guess = self.psf_cen_guess + array(psf_gmix1.get_cen())

            ngauss=get_em_ngauss(model)
            fitter=self._fit_em_ngauss(self.psf_image,
                                       cen_guess,
                                       psf_gmix1,
                                       ngauss)
        else:
            raise ValueError("unsupported psf model: '%s'" % model)

        if fitter is None:
            self.res['flags'] = PSF_FIT_FAILURE
            print("psf failure at object",index)
        else:
            psf_gmix = fitter.get_gmix()
            print("psf fit:")
            print(psf_gmix)
            print("psf fwhm:",2.35*sqrt( psf_gmix.get_T()/2. ))

            self.psf_gmix=psf_gmix
            self.res['psf_gmix']=psf_gmix

            if self.make_plots:
                self._compare_psf(fitter, model)


    def _fit_em_1gauss(self, im, cen, sigma_guess):
        """
        Just run the fitter

        cen is the global cen not within jacobian
        """
        return self._fit_with_em(im, cen, sigma_guess, 1)

    def _fit_em_ngauss(self, im, cen, gmix1, ngauss):
        """
        Start with fit from using 1 gauss, which must be entered

        cen is the global cen not within jacobian
        """

        sigma_guess = sqrt( gmix1.get_T()/2. )
        fitter=self._fit_with_em(im, cen, sigma_guess, ngauss)

        return fitter



    def _fit_with_em(self, im, cen, sigma_guess, ngauss):
        """
        Fit the image using EM

        cen is the global cen not within jacobian
        """
        import ngmix
        from ngmix.gexceptions import GMixMaxIterEM, GMixRangeError
        conf=self.conf

        im_with_sky, sky = ngmix.em.prep_image(im)
        jacob = self._get_jacobian(cen)

        ntry,maxiter,tol = self._get_em_pars()
        for i in xrange(ntry):
            guess = self._get_em_guess(sigma_guess, ngauss)
            print("em guess:")
            print(guess)
            try:
                fitter=self._do_fit_em_with_full_guess(im_with_sky,
                                                       sky,
                                                       guess,
                                                       jacob)
                tres=fitter.get_result()
                print("em numiter:",tres['numiter'])
                break
            except GMixMaxIterEM:
                fitter=None
            except GMixRangeError:
                fitter=None

        return fitter

    def _do_fit_em_with_full_guess(self,
                                   image,
                                   sky,
                                   guess,
                                   jacob):
        import ngmix

        ntry,maxiter,tol = self._get_em_pars()

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
        elif ngauss==3:
            return self._get_em_guess_3gauss(sigma)
        else:
            return self._get_em_guess_ngauss(sigma,ngauss)

    def _get_em_guess_1gauss(self, sigma):
        import ngmix

        sigma2 = sigma**2
        pars=array( [1.0 + 0.1*srandu(),
                     0.2*srandu(),
                     0.2*srandu(), 
                     sigma2*(1.0 + 0.5*srandu()),
                     0.2*sigma2*srandu(),
                     sigma2*(1.0 + 0.5*srandu())] )

        return ngmix.gmix.GMix(pars=pars)

    def _get_em_guess_2gauss(self, sigma):
        import ngmix

        sigma2 = sigma**2

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

    def _get_em_guess_3gauss(self, sigma):
        import ngmix

        sigma2 = sigma**2

        glist=[]

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


    def _get_em_guess_ngauss(self, sigma, ngauss):
        import ngmix

        sigma2 = sigma**2

        glist=[]

        for i in xrange(ngauss):
            if i > 2:
                ig=2
            else:
                ig=i

            glist += [_em3_pguess[ig]*(1.0 + 0.1*srandu()),
                      0.1*srandu(),
                      0.1*srandu(),
                      _em3_fguess[ig]*sigma2*(1.0 + 0.1*srandu()),
                      0.1*srandu(),
                      _em3_fguess[ig]*sigma2*(1.0 + 0.1*srandu())]

        pars=array(glist)

        return ngmix.gmix.GMix(pars=pars)


    def _get_em_pars(self):
        conf=self.conf
        if self.fitting_galaxy:
            return conf['gal_em_ntry'], conf['gal_em_maxiter'], conf['gal_em_tol']
        else:
            return conf['psf_em_ntry'], conf['psf_em_maxiter'], conf['psf_em_tol']

    def _compare_psf(self, fitter, model):
        """
        compare psf image to best fit model
        """
        import images

        model_image = fitter.make_image(counts=self.psf_image.sum())

        plt=images.compare_images(self.psf_image,
                                  model_image,
                                  label1='psf',
                                  label2=self.conf['psf_model'],
                                  show=False)

        pname='psf-resid-%s-%06d.png' % (model, self.index)
        print("          ",pname)
        plt.write_img(1400,800,pname)

    def _do_gal_plots(self, model, fitter):
        """
        Make residual plot and trials plot
        """
        self._compare_gal(model, fitter)
        self._make_trials_plot(model, fitter)

    def _make_trials_plot(self, model, fitter):
        """
        Plot the trials
        """
        width,height=800,800
        tup=fitter.make_plots(title=model)

        if isinstance(tup, tuple):
            p,wp = tup
            wtrials_pname='wtrials-%06d-%s.png' % (self.index,model)
            print("          ",wtrials_pname)
            wp.write_img(width,height,wtrials_pname)
        else:
            p = tup

        trials_pname='trials-%06d-%s.png' % (self.index,model)
        print("          ",trials_pname)
        p.write_img(width,height,trials_pname)

    def _compare_gal(self, model, fitter):
        """
        compare psf image to best fit model
        """
        import images

        gmix = fitter.get_gmix()

        res=self.res
        psf_gmix = res['psf_gmix']
        gmix_conv = gmix.convolve(psf_gmix)

        model_image = gmix_conv.make_image(self.gal_image.shape,
                                           jacobian=res['jacob'])

        plt=images.compare_images(self.gal_image,
                                  model_image,
                                  label1='galaxy',
                                  label2=model,
                                  show=False)
        pname='gal-resid-%06d-%s.png' % (self.index,model)
        print("          ",pname)
        plt.write_img(1400,800,pname)


    def _fit_galaxy(self):
        """
        Fit the galaxy to the models

        First fit a single gaussian with em to get a good center. Also fit that
        model to the image with fixed everything to get a flux, which is linear
        and always gives some answer.

        Then fit the galaxy models.
        """

        self.fitting_galaxy=True

        print('    fitting gal em 1gauss')

        self._fit_galaxy_em()
        self._fit_galaxy_models()

    def _fit_galaxy_em(self):
        """

        Fit a single gaussian with em to find a decent center.  We don't get a
        flux out of that, but we can get flux using the _fit_flux routine

        """
        #import images
        #images.multiview(self.gal_image)

        # first the structural fit
        sigma_guess = sqrt( self.res['psf_gmix'].get_T()/2.0 )
        print('      sigma guess:',sigma_guess)
        fitter=self._fit_em_1gauss(self.gal_image,
                                   self.gal_cen_guess,
                                   sigma_guess)

        em_gmix = fitter.get_gmix()
        print("      em gmix:",em_gmix)

        row_rel, col_rel = em_gmix.get_cen()
        em_cen = self.gal_cen_guess + array([row_rel,col_rel])

        print("      em gauss cen:",em_cen)

        jacob=self._get_jacobian(em_cen)

        # now get a flux
        print('    fitting robust gauss flux')
        flux, flux_err = self._fit_flux(self.gal_image,
                                        self.weight_image,
                                        jacob,
                                        em_gmix)

        self.res['em_gauss_flux'] = flux
        self.res['em_gauss_flux_err'] = flux_err
        self.res['em_gauss_cen'] = jacob.get_cen()
        self.res['em_gmix'] = em_gmix
        self.res['jacob'] = jacob

    def _fit_flux(self, image, weight_image, jacob, gmix):
        """
        Fit the flux from a fixed gmix model.  This is linear and always
        succeeds.
        """
        import ngmix

        fitter=ngmix.fitting.PSFFluxFitter(image,
                                           weight_image,
                                           jacob,
                                           gmix)
        fitter.go()
        res=fitter.get_result()

        flux=res['flux']
        flux_err=res['flux_err']
        mess='         %s +/- %s' % (flux,flux_err)
        print(mess)

        return flux, flux_err


    def _fit_galaxy_models(self):
        """
        Run through and fit all the models
        """

        for model in self.fit_models:
            print('    fitting:',model)

            if model=='bdf':
                self._fit_bdf()
            elif model=='sersic':
                self._fit_sersic()
            else:
                self._fit_simple(model)

            self._print_galaxy_res(model)

            if self.make_plots:
                self._do_gal_plots(model, self.res[model]['fitter'])

    def _fit_sersic(self):
        """
        Fit a sersic model, taking guesses from our previous em fits
        """
        import ngmix
        from ngmix.fitting import print_pars

        res=self.res

        priors=self.priors['sersic']

        cen_prior=self.cen_prior
        g_prior=priors['g']
        T_prior=priors['T']
        counts_prior=priors['counts']
        n_prior=priors['n']

        res=self.res
        conf=self.conf

        #image,wt,jacobian=self._get_image_data()

        fitter=ngmix.fitting.MCMCSersic(self.gal_image,
                                        self.weight_image,
                                        res['jacob'],
                                        psf=res['psf_gmix'],

                                        nwalkers=conf['nwalkers'],
                                        mca_a=conf['mca_a'],

                                        shear_expand=self.shear_expand,

                                        cen_prior=cen_prior,
                                        T_prior=T_prior,
                                        counts_prior=counts_prior,
                                        g_prior=g_prior,
                                        n_prior=n_prior,
                                        do_pqr=conf['do_pqr'])

        full_guess=self._get_guess_sersic(conf['nwalkers'])
        pos=fitter.run_mcmc(full_guess,conf['burnin'])
        pos=fitter.run_mcmc(pos,conf['nstep'])
        fitter.calc_result()

        self.res['sersic'] = {'fitter':fitter,
                              'res':fitter.get_result()}

        return fitter




    def _fit_simple(self, model):
        """
        Fit the simple model, taking guesses from our
        previous em fits
        """
        import ngmix

        if self.joint_prior is not None:
            return self._fit_simple_joint(model)


        priors=self.priors[model]
        g_prior=priors['g']
        T_prior=priors['T']
        counts_prior=priors['counts']
        cen_prior=self.cen_prior

        res=self.res
        conf=self.conf

        full_guess=self._get_guess_simple()

        fitter=ngmix.fitting.MCMCSimple(self.gal_image,
                                        self.weight_image,
                                        res['jacob'],
                                        model,
                                        psf=res['psf_gmix'],

                                        nwalkers=conf['nwalkers'],
                                        burnin=conf['burnin'],
                                        nstep=conf['nstep'],
                                        mca_a=conf['mca_a'],

                                        full_guess=full_guess,

                                        shear_expand=self.shear_expand,

                                        cen_prior=cen_prior,
                                        T_prior=T_prior,
                                        counts_prior=counts_prior,
                                        g_prior=g_prior,
                                        do_pqr=conf['do_pqr'])
        fitter.go()

        self.res[model] = {'fitter':fitter,
                           'res':fitter.get_result()}


    def _fit_simple_joint(self, model):
        """
        Fit the simple model, taking guesses from our
        previous em fits
        """
        import ngmix
        from ngmix.fitting import MCMCSimpleJointHybrid

        cen_prior=self.cen_prior

        res=self.res
        conf=self.conf


        ntry=conf['mcmc_ntry']
        for i in xrange(ntry):

            full_guess=self._get_guess_simple_joint()

            fitter=MCMCSimpleJointHybrid(self.gal_image,
                                         self.weight_image,
                                         res['jacob'],
                                         model,
                                         psf=res['psf_gmix'],

                                         nwalkers=conf['nwalkers'],
                                         burnin=conf['burnin'],
                                         nstep=conf['nstep'],
                                         mca_a=conf['mca_a'],

                                         full_guess=full_guess,
                                         shear_expand=self.shear_expand,

                                         cen_prior=cen_prior,
                                         joint_prior=self.joint_prior,

                                         do_pqr=conf['do_pqr'])
            fitter.go()

            # guard against rare mcmc problems
            tres=fitter.get_result()
            if (tres['pars'] is not None
                    and tres['flags'] == 0
                    and abs(tres['pars'][2]) <1 ):
                break
            else:
                print("          bad mcmc result:")
                pprint(tres)
                print("          trying again")

        self.res[model] = {'fitter':fitter,
                           'res':tres}


    def _get_guess_simple(self,
                          widths=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01]):
        """
        Get a guess centered on the truth

        width is relative for T and counts
        """

        

        nwalkers = self.conf['nwalkers']

        res=self.res
        gmix = res['em_gmix']
        g1,g2,T = gmix.get_g1g2T()
        flux = res['em_gauss_flux']

        guess=numpy.zeros( (nwalkers, 6) )

        # centers relative to jacobian center
        guess[:,0] = widths[0]*srandu(nwalkers)
        guess[:,1] = widths[1]*srandu(nwalkers)

        guess_shape=get_shape_guess(g1, g2, nwalkers, width=widths[2])
        guess[:,2]=guess_shape[:,0]
        guess[:,3]=guess_shape[:,1]

        guess[:,4] = get_positive_guess(T,nwalkers,width=widths[4])
        guess[:,5] = get_positive_guess(flux,nwalkers,width=widths[5])

        return guess


    def _get_guess_sersic(self,
                          num,
                          pars=None,
                          random_n=True,
                          widths=[0.05, 0.05, 0.05, 0.05]):
        """
        Guess based on the em fit, with n drawn from prior
        """

        res=self.res

        if pars is None:
            gmix = res['em_gmix']
            cen1,cen2=gmix.get_cen()
            g1,g2,T = gmix.get_g1g2T()
            flux = res['em_gauss_flux']
        else:
            cen1=pars[0]
            cen2=pars[1]
            g1=pars[2]
            g2=pars[3]
            T=pars[4]
            flux=pars[5]

        shapes=get_shape_guess(g1, g2, num, width=widths[1])

        n_prior=self.priors['sersic']['n']

        if random_n:
            nvals = n_prior.sample(num)
        else:
            nvals=zeros(num)
            nvals[:] = 0.5*(n_prior.minval+n_prior.maxval)

        guess=zeros( (num, 7) )

        guess[:,0] = cen1 + widths[0]*srandu(num)
        guess[:,1] = cen2 + widths[0]*srandu(num)

        guess[:,2]=shapes[:,0]
        guess[:,3]=shapes[:,1]

        guess[:,4] = get_positive_guess(T,num,width=widths[2])
        guess[:,5] = get_positive_guess(flux,num,width=widths[3])

        guess[:,6] = nvals

        return guess



    def _get_guess_simple_joint(self):
        """
        Get a guess centered on the truth

        width is relative for T and counts
        """

        width = 0.01

        nwalkers = self.conf['nwalkers']

        res=self.res
        gmix = res['em_gmix']
        g1,g2,T = gmix.get_g1g2T()
        F = res['em_gauss_flux']


        guess=numpy.zeros( (nwalkers, 6) )

        guess[:,0] = width*srandu(nwalkers)
        guess[:,1] = width*srandu(nwalkers)

        guess_shape=get_shape_guess(g1,g2,nwalkers,width=width)
        guess[:,2]=guess_shape[:,0]
        guess[:,3]=guess_shape[:,1]

        guess[:,4] = log10( get_positive_guess(T,nwalkers,width=width) )

        # got anything better?
        guess[:,5] = log10( get_positive_guess(F,nwalkers,width=width) )

        return guess





    def _fit_bdf(self):
        """
        Fit the simple model, taking guesses from our
        previous em fits
        """
        import ngmix

        if self.joint_prior is not None:
            return self._fit_bdf_joint()

        priors=self.priors['bdf']

        g_prior=priors['g']
        T_prior=priors['T']
        counts_prior=priors['counts']
        cen_prior=self.cen_prior
        bfrac_prior=self.bfrac_prior


        res=self.res
        conf=self.conf

        full_guess=self._get_guess_bdf()

        fitter=ngmix.fitting.MCMCBDF(self.gal_image,
                                     self.weight_image,
                                     res['jacob'],
                                     psf=res['psf_gmix'],

                                     nwalkers=conf['nwalkers'],
                                     burnin=conf['burnin'],
                                     nstep=conf['nstep'],
                                     mca_a=conf['mca_a'],

                                     full_guess=full_guess,

                                     shear_expand=self.shear_expand,

                                     cen_prior=cen_prior,
                                     T_prior=T_prior,
                                     counts_prior=counts_prior,
                                     g_prior=g_prior,
                                     bfrac_prior=bfrac_prior,

                                     do_pqr=conf['do_pqr'])
        fitter.go()

        self.res['bdf'] = {'fitter':fitter,
                           'res':fitter.get_result()}

    def _get_guess_bdf(self,
                       widths=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01]):
        """
        Get a guess centered on the truth

        width is relative for T and counts
        """

        nwalkers = self.conf['nwalkers']

        res=self.res
        gmix = res['em_gmix']
        g1,g2,T = gmix.get_g1g2T()
        flux = res['em_gauss_flux']

        guess=numpy.zeros( (nwalkers, 7) )

        # centers relative to jacobian center
        guess[:,0] = widths[0]*srandu(nwalkers)
        guess[:,1] = widths[1]*srandu(nwalkers)

        guess_shape=get_shape_guess(g1, g2, nwalkers, width=widths[2])
        guess[:,2]=guess_shape[:,0]
        guess[:,3]=guess_shape[:,1]

        guess[:,4] = get_positive_guess(T,nwalkers,width=widths[4])

        counts_guess = get_positive_guess(flux,nwalkers,width=widths[5])

        bfracs=numpy.zeros(nwalkers)
        nhalf=nwalkers/2
        bfracs[0:nhalf] = 0.01*randu(nhalf)
        bfracs[nhalf:] = 0.99+0.01*randu(nhalf)

        dfracs = 1.0 - bfracs

        # bulge flux
        guess[:,5] = bfracs*counts_guess
        # disk flux
        guess[:,6] = dfracs*counts_guess

        return guess


    def _fit_bdf_joint(self):
        """
        Fit the bdf model using joint prior, taking guesses from our previous
        em fits
        """
        import ngmix
        from ngmix.fitting import MCMCBDFJointHybrid

        cen_prior=self.cen_prior

        res=self.res
        conf=self.conf

        nwalkers=conf['nwalkers']
        ntry=conf['mcmc_ntry']

        which_arate=1
        for i in xrange(ntry):


            full_guess=self._get_guess_bdf_joint(nwalkers=nwalkers)

            fitter=MCMCBDFJointHybrid(self.gal_image,
                                      self.weight_image,
                                      res['jacob'],
                                      psf=res['psf_gmix'],

                                      nwalkers=nwalkers,
                                      burnin=conf['burnin'],
                                      nstep=conf['nstep'],
                                      mca_a=conf['mca_a'],

                                      #Tfracdiff_max=conf['Tfracdiff_max'],

                                      shear_expand=self.shear_expand,

                                      full_guess=full_guess,

                                      cen_prior=cen_prior,
                                      joint_prior=self.joint_prior,

                                      do_pqr=conf['do_pqr'])


            fitter.go()

            tres=fitter.get_result()

            if (tres['arate'] < conf['min_arate']
                    and i < (ntry-1) and which_arate==1):
                nwalkers = nwalkers*conf['nwalkers_multiply']
                which_arate=2
                print("low arate:",tres['arate'],"trying with nwalkers=",nwalkers)
                continue

            # guard against rare mcmc problems
            if (tres['pars'] is not None
                    and tres['flags'] == 0
                    and abs(tres['pars'][2]) <1 ):
                break
            else:
                print("          bad mcmc result:")
                pprint(tres)
                print("          trying again")


        self.res['bdf'] = {'fitter':fitter,
                           'res':fitter.get_result()}


    def _get_guess_bdf_joint(self, nwalkers=1):
        """
        Get a guess centered on the truth

        width is relative for T and counts
        """

        width = 0.01


        res=self.res
        gmix = res['em_gmix']
        g1,g2,T = gmix.get_g1g2T()
        flux = res['em_gauss_flux']


        guess=numpy.zeros( (nwalkers, 7) )

        guess[:,0] = width*srandu(nwalkers)
        guess[:,1] = width*srandu(nwalkers)

        guess_shape=get_shape_guess(g1,g2,nwalkers,width=width)
        guess[:,2]=guess_shape[:,0]
        guess[:,3]=guess_shape[:,1]

        guess[:,4] = log10( get_positive_guess(T,nwalkers,width=width) )

        # got anything better?
        Fb = 0.1*flux
        Fd = 0.9*flux
        guess[:,5] = log10( get_positive_guess(Fb,nwalkers,width=width) )
        guess[:,6] = log10( get_positive_guess(Fd,nwalkers,width=width) )

        return guess





    def _print_galaxy_res(self, model):
        from ngmix.fitting import print_pars
        res=self.res[model]['res']

        print_pars(res['pars'],     front="    pars: ")
        print_pars(res['pars_err'], front="    err:  ")
 
        print('        arate:',res['arate'])


    def _finish_setup(self):
        """
        Process the rest of the input
        """

        conf=self.conf
        self.fit_models=conf['fit_models']
        self.make_plots = conf['make_plots']
        if self.make_plots:
            print("will make plots!")

        self.shear_expand=conf.get('shear_expand')
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

        nmod=len(self.fit_models)

        self.cen_prior=get_cen_prior(conf)
        self.joint_prior=get_joint_prior(conf)

        if self.joint_prior is not None:
            self.priors=None
        else:
            T_priors=get_T_priors(conf)
            counts_priors=get_counts_priors(conf)
            g_priors=get_g_priors(conf)
            n_priors=get_n_priors(conf)

            if (len(T_priors) != nmod
                    or len(counts_priors) != nmod
                    or len(n_priors) != nmod
                    or len(g_priors) != nmod):
                raise ValueError("models and T,counts,g priors must be same length")

            priors={}
            for i in xrange(nmod):
                model=self.fit_models[i]

                T_prior=T_priors[i]

                # note it is a list
                counts_prior=counts_priors[i]

                g_prior=g_priors[i]
                n_prior=n_priors[i]
                
                modlist={'T':T_prior, 'counts':counts_prior, 'g':g_prior, 'n':n_prior}
                priors[model] = modlist

            self.priors=priors

            # bulge+disk fixed size ratio
            self.bfrac_prior=get_bfrac_prior(conf)

    def _print_res(self, res):
        pass


    def _copy_to_output(self, sub_index, res):
        """
        Copy the galaxy fits
        """

        data=self.data
        conf=self.conf

        # overall flags, model flags copied below
        data['flags'][sub_index] = res['flags']

        if 'psf_gmix_em1' in res:
            self._copy_psf_em1_to_output(res['psf_gmix_em1'],sub_index)

        if 'psf_gmix' in res:
            self._copy_psf_to_output(res['psf_gmix'],sub_index)

            # em fit to convolved galaxy
            data['em_gauss_flux'][sub_index] = res['em_gauss_flux']
            data['em_gauss_flux_err'][sub_index] = res['em_gauss_flux_err']
            data['em_gauss_cen'][sub_index] = res['em_gauss_cen']

            for model in self.fit_models:
                self._copy_pars(sub_index, model, res)

    def _copy_pars(self, sub_index, model, allres):
        """
        Copy from the result dict to the output array
        """

        conf=self.conf
        res = allres[model]['res']

        n=get_model_names(model)

        pars=res['pars']
        pars_cov=res['pars_cov']

        flux=pars[5:].sum()
        flux_err=sqrt( pars_cov[5:, 5:].sum() )

        self.data[n['flags']][sub_index] = res['flags']

        self.data[n['pars']][sub_index,:] = pars
        self.data[n['pars_cov']][sub_index,:,:] = pars_cov

        self.data[n['flux']][sub_index] = flux
        self.data[n['flux_err']][sub_index] = flux_err

        self.data[n['g']][sub_index,:] = res['g']
        self.data[n['g_cov']][sub_index,:,:] = res['g_cov']

        self.data[n['arate']][sub_index] = res['arate']
        self.data[n['tau']][sub_index] = res['tau']

        for sn in _stat_names:
            if sn in res:
                self.data[n[sn]][sub_index] = res[sn]

        if conf['do_pqr']:
            self.data[n['P']][sub_index] = res['P']
            self.data[n['Q']][sub_index,:] = res['Q']
            self.data[n['R']][sub_index,:,:] = res['R']
 
    def _copy_psf_em1_to_output(self, gmix, sub_index):
        """
        copy some psf info
        """
        g1,g2,T=gmix.get_g1g2T()

        data=self.data
        data['psf_em1_g'][sub_index,0] = g1
        data['psf_em1_g'][sub_index,1] = g2
        data['psf_em1_T'][sub_index] = T


    def _copy_psf_to_output(self, gmix, sub_index):
        """
        copy some psf info
        """
        pars=gmix.get_full_pars()
        self.data['psf_pars'][sub_index,:] = pars

    def _get_model_npars(self, model):
        """
        Get the models and number of parameters
        """
        if model in ['sersic','bdf']:
            return 7
        else:
            return 6

    def _make_struct(self):
        """
        make the output structure
        """

        conf=self.conf

        dt = self._get_default_dtype()

        psf_model=conf['psf_model']
        if 'em' in psf_model:
            ngauss=get_em_ngauss(psf_model)
            dt += [('psf_pars','f8',ngauss*6)]
        else:
            raise ValueError("unsupported psf model: '%s'" % psf_model)

        n=get_model_names('em_gauss')
        dt += [(n['flux'],    'f8'),
               (n['flux_err'],'f8'),
               (n['cen'],'f8',2)]

        models=self.fit_models
        for model in models:
            np = self._get_model_npars(model)

            n=get_model_names(model)

            dt+=[(n['flags'],'i4'),
                 (n['pars'],'f8',np),
                 (n['pars_cov'],'f8',(np,np)),
                 (n['flux'],'f8'),
                 (n['flux_err'],'f8'),
                 (n['g'],'f8',2),
                 (n['g_cov'],'f8',(2,2)),
                
                 (n['s2n_w'],'f8'),
                 (n['chi2per'],'f8'),
                 (n['dof'],'f8'),
                 (n['aic'],'f8'),
                 (n['bic'],'f8'),
                 (n['arate'],'f8'),
                 (n['tau'],'f8'),
                ]

            if conf['do_pqr']:
                dt += [(n['P'], 'f8'),
                       (n['Q'], 'f8', 2),
                       (n['R'], 'f8', (2,2))]


        num=self.index_list.size
        data=numpy.zeros(num, dtype=dt)

        data['psf_pars'] = DEFVAL
        data['em_gauss_flux'] = DEFVAL
        data['em_gauss_flux_err'] = PDEFVAL
        data['em_gauss_cen'] = DEFVAL

        for model in self.fit_models:
            n=get_model_names(model)

            data[n['flags']] = NO_ATTEMPT

            data[n['pars']] = DEFVAL
            data[n['pars_cov']] = PDEFVAL
            data[n['flux']] = DEFVAL
            data[n['flux_err']] = PDEFVAL
            data[n['g']] = DEFVAL
            data[n['g_cov']] = PDEFVAL

            data[n['s2n_w']] = DEFVAL
            data[n['chi2per']] = PDEFVAL
            data[n['aic']] = BIG_PDEFVAL
            data[n['bic']] = BIG_PDEFVAL

            data[n['tau']] = BIG_PDEFVAL

            if conf['do_pqr']:
                data[n['P']] = DEFVAL
                data[n['Q']] = DEFVAL
                data[n['R']] = DEFVAL
     
        self.data=data

def get_shear(data, model):
    """
    Calculate the shear from the pqr stats
    """
    import lensing
    Pname='%s_P' % model
    Qname='%s_Q' % model
    Rname='%s_R' % model
    sh,cov=lensing.pqr.get_pqr_shear(data[Pname],data[Qname],data[Rname])
    res={'shear':sh,
         'shear_cov':cov}
    return res

def get_joint_prior(conf):
    import ngmix
    from . import joint_prior

    jptype = conf.get('joint_prior_type',None)
    if jptype==None:
        jp=None
    elif 'bdf' in jptype:
        jp=joint_prior.make_joint_prior_bdf(type=jptype)
    else:
        jp = joint_prior.make_joint_prior_simple(type=jptype)

    return jp

def get_T_priors(conf):
    import ngmix

    T_prior_types=conf.get('T_prior_types',None)
    if T_prior_types is None:
        return None

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

    counts_prior_types=conf.get('counts_prior_types',None)
    if counts_prior_types is None:
        return None

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
    g_prior_types=conf.get('g_prior_types',None)

    if g_prior_types is None:
        return None

    g_priors=[]
    for i,typ in enumerate(g_prior_types):
        if typ =='exp':
            pars=conf['g_prior_pars'][i]
            parr=array(pars,dtype='f8')
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
        elif typ is None:
            g_prior = None
        else:
            raise ValueError("implement gprior '%s'")
        g_priors.append(g_prior)

    return g_priors

def get_n_priors(conf):
    import ngmix
    n_prior_types=conf.get('n_prior_types',None)

    if n_prior_types is None:
        return None

    n_priors=[]
    for i,typ in enumerate(n_prior_types):
        if typ =='flat':
            pars=conf['n_prior_pars'][i]
            n_prior=ngmix.priors.FlatPrior(pars[0], pars[1])
        else:
            raise ValueError("implement n prior '%s'")
        n_priors.append(n_prior)

    return n_priors



def get_cen_prior(conf):
    import ngmix
    use_cen_prior=conf.get('use_cen_prior',False)
    if use_cen_prior:
        width=conf.get('cen_width',1.0)
        return ngmix.priors.CenPrior(0.0, 0.0, width,width)
    else:
        return None

def get_bfrac_prior(conf):
    
    bptype = conf.get('bfrac_prior_type',None)

    if bptype == 'default':
        import ngmix
        # use the miller and im3shape style
        bfrac_prior=ngmix.priors.BFrac()
    elif bptype ==None:
        bfrac_prior=None
    else:
        raise ValueError("bad bfrac_prior_type: '%s'" % bptype)

    return bfrac_prior

_em2_fguess =array([0.5793612389470884,1.621860687127999])
_em2_pguess =array([0.596510042804182,0.4034898268889178])

_em3_pguess = array([0.596510042804182,0.4034898268889178,1.303069003078001e-07])
_em3_fguess = array([0.5793612389470884,1.621860687127999,7.019347162356363],dtype='f8')

#_em2_fguess=array([12.6,3.8])
#_em2_fguess[:] /= _em2_fguess.sum()
#_em2_pguess=array([0.30, 0.70])
#_em_ngauss_map = {'em1':1, 'em2':2, 'em3':3,'em4':4,'em5':5,'em6':6}
def get_em_ngauss(name):
    ngauss=int( name[2:] )
    return ngauss

_stat_names=['s2n_w',
             'chi2per',
             'dof',
             'aic',
             'bic']


def get_model_names(model):
    names=['flags',
           'pars',
           'pars_cov',
           'cen',
           'flux',
           'flux_err',
           'g',
           'g_cov',
           'P',
           'Q',
           'R',
           'tries',
           'arate',
           'tau']

    names += _stat_names

    ndict={}
    for n in names:
        ndict[n] = '%s_%s' % (model,n)

    return ndict

def get_shape_guess(g1, g2, n, width=0.01):
    """
    Get guess, making sure in range
    """
    import ngmix
    from ngmix.gexceptions import GMixRangeError

    gtot = sqrt(g1**2 + g2**2)
    if gtot > 0.98:
        g1 = g1*0.9
        g2 = g2*0.9

    guess=numpy.zeros( (n, 2) )
    shape=ngmix.Shape(g1, g2)

    for i in xrange(n):

        while True:
            try:
                g1_offset = width*srandu()
                g2_offset = width*srandu()
                shape_new=shape.copy()
                shape_new.shear(g1_offset, g2_offset)
                break
            except GMixRangeError:
                pass

        guess[i,0] = shape_new.g1
        guess[i,1] = shape_new.g2

    return guess

def get_positive_guess(val, n, width=0.01):
    """
    Get guess, making sure positive
    """
    from ngmix.gexceptions import GMixRangeError

    if val <= 0.0:
        print("val <= 0: %s" % val)
        print("using arbitrary value of 0.1")
        val=0.1
        #raise GMixRangeError("val <= 0: %s" % val)

    vals=numpy.zeros(n)-9999.0
    while True:
        w,=numpy.where(vals <= 0)
        if w.size == 0:
            break
        else:
            vals[w] = val*(1.0 + width*srandu(w.size))

    return vals


