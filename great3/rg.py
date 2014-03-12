from __future__ import print_function
import numpy
from numpy.random import random as randu

from . import files
from .generic import *
from .constants import *



class RGFitter(FitterBase):

    def _process_object(self, sub_index):
        """
        run re-gauss
        """
        index = self.index_list[sub_index]

        gal_image,gal_cen = self.field.get_gal_image(index)
        psf_image,psf_cen = self.field.get_star_image(0)

        res=self._run_regauss(gal_image, gal_cen,
                              psf_image, psf_cen)
        return res

    def _run_regauss(self,
                     gal_image, gal_cen,
                     psf_image, psf_cen):

        import admom

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
                R = res['R']
                res['err'] = rg['rgstats']['uncer']/R
                res['flags'] = 0
                break
        
        if res is None:
            print("    regauss failed")
            res={'flags':RG_FAILURE}

        res['ntry'] = i+1
        return res

    def _get_guesses(self, gal_cen, psf_cen):
        cen_guess = gal_cen + self.guess_width_cen*srandu(2)
        psf_cen_guess = psf_cen + self.guess_width_cen*srandu(2)

        psf_irr_guess = self.guess_psf_irr*(1.0+0.01*srandu())
        # guess gal bigger.  Note random here is [0,1]
        gal_irr_guess = self.guess_psf_irr*(1.0+1.0*randu())

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
        data['ntry'][sub_index] = res['ntry']

        if res['flags']==0:
            data['e1'][sub_index]  = res['e1']
            data['e2'][sub_index]  = res['e2']
            data['err'][sub_index] = res['err']
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
               ('R','f8'),
               ('ntry','i4')]

        data=numpy.zeros(num, dtype=dt)

        # from default dtype.
        data['flags'] = NO_ATTEMPT

        data['e1']  = DEFVAL
        data['e2']  = DEFVAL
        data['err'] = PDEFVAL
        data['R']   = DEFVAL

        self.data=data


def select_and_calc_shear(data, **cuts):
    """
    Make selections and calculate the mean shear
    """
    w=select(data, **cuts)
    res=get_shear(data[w], **cuts)
    return res

def select(data,**keys):
    """
    parameters
    ----------
    data:
        The output of regauss, e.g. from using read_output
    max_ellip:
        Maximum ellipticity in either component
    max_err:
        Maximum ellipticity error.
    R_range:
        Two-element sequence for the range
    """

    max_ellip=keys['max_ellip']
    max_err=keys['max_err']
    R_range=keys['R_range']

    w,=numpy.where(  (data['flags'] == 0)
                   & (numpy.abs(data['e1']) < max_ellip)
                   & (numpy.abs(data['e2']) < max_ellip)
                   & (data['err'] < max_err)
                   & (data['R'] > R_range[0])
                   & (data['R'] < R_range[1]) )
    return w


def get_shear(data, **keys):
    """
    parameters
    ----------
    data:
        Outputs from running regauss.
    include_shape_noise_err: bool, optional
        If True, just use the weighted variance of shapes, otherwise
        use the errors only.
    shape_noise_type: string, optional
        What type of fits from cosmos to use for shape noise. 
    """
    from esutil.stat import wmom

    wt, ssh = get_weight_and_ssh(data['e1'], data['e2'], data['err'],
                                 shape_noise_type=shape_noise_type)

    e1mean,e1err = wmom(data['e1'], wt, calcerr=True)
    e2mean,e2err = wmom(data['e2'], wt, calcerr=True)
    R,Rerr = wmom(data['R'], wt, calcerr=True)
    ssh,ssherr = wmom(ssh, wt, calcerr=True)

    if not include_shape_noise_err:
        err2=err**2
        e1ivar = ( 1.0/err2 ).sum()
        e1err = numpy.sqrt( 1.0/e1ivar )
        e2err = e1err

    g1=0.5*e1mean/ssh
    g2=0.5*e2mean/ssh

    g1err = 0.5*e1err/ssh
    g2err = 0.5*e2err/ssh

    shear=numpy.array([g1,g2])
    shear_err=numpy.array([g1err,g2err])
    shear_cov=numpy.zeros( (2,2) )
    shear_cov[0,0]=g1err**2
    shear_cov[1,1]=g2err**2

    out={'shear':shear,
         'shear_cov':shear_cov,
         'ssh':ssh,
         'R':R,
         'Rerr':Rerr}
    return out

def get_weight_and_ssh(e1, e2, err, shape_noise_type='exp'):
    """
    err is per component, as is the shape noise
    """
    esq = e1**2 + e2**2
    err2=err**2

    # for responsivity. 
    #   Shear = 0.5 * sum(w*e1)/sum(w)/R
    # where
    #   ssh = sum(w*ssh)/sum(w)
    # this shape noise is per component

    sn2 = get_shape_noise(shape_noise_type)

    # coefficients (eq 5-35 Bern02) 

    f = sn2/(sn2 + err2)

    ssh = 1.0 - (1-f)*sn2 - 0.5*f**2 * esq

    weight = 1.0/(sn2 + err2)

    return weight, ssh

_SN2={'exp':None,'dev':None}
def get_shape_noise(type):
    """
    get the shape noise per component
    """
    if _SN2[type] is None:

        import ngmix
        
        if type=='exp':
            g_prior=ngmix.priors.make_gprior_cosmos_exp()
        elif type=='dev':
            g_prior=ngmix.priors.make_gprior_cosmos_dev()
        else:
            raise ValueError("bad shape noise type: '%s'" % type)

        n=100000
        g1,g2 = g_prior.sample2d(n)
        e1=g1.copy()
        e2=g2.copy()
        for i in xrange(n):
            e1[i], e2[i] = ngmix.shape.g1g2_to_e1e2(g1[i],g2[i])
        _SN2[type] = e1.var()

        print("SN2['%s']: %s" % (type,_SN2[type]))

    return _SN2[type]


