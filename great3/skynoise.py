from __future__ import print_function

import os
import numpy
from . import files
from .constants import *

def calc_branch_sky_noise(**keys):
    """
    Calculate the sky noise in all images from the specified branch
    """
    import fitsio

    nsub=files.get_nsub(**keys)

    deep=keys.get('deep',False)
    if deep:
        outfile=files.get_deep_skynoise_file(**keys)

        fmin=-0.08
        fmax= 0.035
        binsize=0.001

    else:
        outfile=files.get_skynoise_file(**keys)
        fmin=-0.3
        fmax= 0.15
        binsize=0.005



    print("will write to:",outfile)
    d=files.get_skynoise_plot_dir(**keys)
    if not os.path.exists(d):
        os.makedirs(d)

    out=numpy.zeros(1,dtype=[('subid','f8'),
                             ('skysig','f8'),
                             ('skysig_err','f8')])

    if deep:
        im=files.read_deep_gal_image(**keys)
        plot_file=files.get_deep_skynoise_plot_file(**keys)
    else:
        im=files.read_gal_image(**keys)
        plot_file=files.get_skynoise_plot_file(**keys)

    gf = get_sky_noise(im,fmin,fmax,binsize)
    res=gf.get_result()
    plt=gf.make_plot(show=False)

    print("    writing:",plot_file)
    plt.write_eps(plot_file)

    pars=res['pars']
    perr=res['perr']

    print('    %.3g +/- %.3g' % (pars[1],perr[1]))
    out['subid']=keys['subid']
    out['skysig']=pars[1]
    out['skysig_err']=perr[1]

    print("writing:",outfile)
    fitsio.write(outfile, out, clobber=True)

def get_sky_noise(image,fmin,fmax,binsize):
    import fitting

    imravel=image.ravel()
    w,=numpy.where( (imravel > fmin) & (imravel < fmax) )
    
    gf=fitting.GaussFitter(imravel[w],
                           min=fmin,
                           max=fmax,
                           binsize=binsize)
    gf.dofit([0.0, 0.08, image.sum()])

    return gf


