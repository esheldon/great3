import numpy
from . import files
from .constants import *

def cal_sky_noise_fields(**keys):
    """
    Calculate the sky noise in all images from the specified branch
    """
    
    deep=keys.get('deep',False)
    if deep:
        nsub=NSUB_DEEP
    else:
        nsub=NSUB

    out=numpy.zeros(NSUB,dtype=[('subid','f8'), ('skysig','f8')])

    for subid in xrange(NSUB):
        keys['subid']=subid
        if deep:
            im=files.read_deep_gal_image(**keys)
        else:
            im=files.read_gal_image(**keys)

        gf = get_sky_noise(im)
        res=gf.get_result()

        out['subid']=subid
        out['skysig']=res['pars'][1]
        out['skysig_err']=res['perr'][2]

def get_sky_noise(image):
    import fitting

    fmin=-0.3
    fmax=0.15
    binsize=0.005

    imravel=image.ravel()
    w,=numpy.where( (imravel > fmin) & (imravel < fmax) )
    
    gf=fitting.GaussFitter(imravel[w],
                           min=fmin,
                           max=fmax
                           binsize=binsize)
    gf.dofit([0.0, 0.08, image.sum()])

    return gf
