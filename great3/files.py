"""
File locations and reading for great3
"""
from __future__ import print_function
import os

def get_dir():
    """
    The GREAT3_DATA_DIR environment variable must be set
    """
    d=os.environ['GREAT3_DATA_DIR']
    return d

def get_branch_dir(**keys):
    """
    $GREAT3_DATA_DIR/experiment/obs_type/shear_type
    e.g.      /control/ground/constant

    parameters
    ----------
    experiment: string
        Required keyword.  e.g. control, real
    obs_type: string
        Required keyword. e.g. ground, space
    shear_type: string
        Required keyword.  e.g. constant
    """
    d=get_dir()

    experiment=keys['experiment']
    obs_type=keys['obs_type']
    shear_type=keys['shear_type']

    d = os.path.join(d, experiment, obs_type, shear_type)
    return d

def get_file(**keys):
    """
    parameters
    ----------
    experiment: string
        Required keyword.  e.g. control, real
    obs_type: string
        Required keyword. e.g. ground, space
    shear_type: string
        Required keyword.  e.g. constant
    ftype: string
        Required keyword. file type, e.g. galaxy_catalog, star_image, etc.
    subid: number
        Requird keyword. e.g. 12
    epoch: number
        Optional keyword.  Default 0 for image types, none for catalogs.
    """
    d=get_branch_dir(**keys)

    if 'epoch' not in keys:
        keys['epoch'] = 0

    if 'catalog' in keys['ftype']:
        fname='%(ftype)s-%(subid)03d.fits' 
    else:
        fname='%(ftype)s-%(subid)03d-%(epoch)d.fits' 

    fname = fname % keys

    fname = os.path.join(d, fname)
    return fname


def get_gal_cat_file(**keys):
    """
    Same parameters as get_file but ftype is set to 'galaxy_catalog'
    and epoch is not required
    """
    d=get_branch_dir(**keys)

    nkeys={}
    nkeys.update(keys)
    nkeys['ftype'] = 'galaxy_catalog'
    return get_file(**nkeys)

def read_gal_cat(**keys):
    """
    Same parameters as get_file but ftype is set to 'galaxy_catalog'
    and epoch is not required
    """
    import fitsio

    fname=get_gal_cat_file(**keys)
    print('reading:',fname)

    data=fitsio.read(fname, lower=True)
    return data

def get_star_cat_file(**keys):
    """
    Same parameters as get_file but ftype is set to 'star_catalog'
    and epoch is not required
    """
    d=get_branch_dir(**keys)

    nkeys={}
    nkeys.update(keys)
    nkeys['ftype'] = 'star_catalog'
    return get_file(**nkeys)

def read_star_cat(**keys):
    """
    Same parameters as get_file but ftype is set to 'star_catalog'
    and epoch is not required
    """
    import fitsio

    fname=get_star_cat_file(**keys)
    print('reading:',fname)

    data=fitsio.read(fname, lower=True)
    return data

def get_gal_image_file(**keys):
    """
    Same parameters as get_file but ftype is set to 'image'
    """
    d=get_branch_dir(**keys)

    nkeys={}
    nkeys.update(keys)
    nkeys['ftype'] = 'image'
    return get_file(**nkeys)

def read_gal_image(**keys):
    """
    Same parameters as get_file but ftype is set to 'image'
    and epoch is not required
    """
    import fitsio

    fname=get_gal_image_file(**keys)
    print('reading:',fname)

    data=fitsio.read(fname)
    return data


def get_star_image_file(**keys):
    """
    Same parameters as get_file but ftype is set to 'starfield_image'
    """
    d=get_branch_dir(**keys)

    nkeys={}
    nkeys.update(keys)
    nkeys['ftype'] = 'starfield_image'
    return get_file(**nkeys)

def read_star_image(**keys):
    """
    Same parameters as get_file but ftype is set to 'starfield_image'
    and epoch is not required
    """
    import fitsio

    fname=get_star_image_file(**keys)
    print('reading:',fname)

    data=fitsio.read(fname)
    return data


def get_deep_gal_cat_file(**keys):
    """
    Same parameters as get_file but ftype is set to 'galaxy_catalog'
    and epoch is not required
    """
    d=get_branch_dir(**keys)

    nkeys={}
    nkeys.update(keys)
    nkeys['ftype'] = 'deep_galaxy_catalog'
    return get_file(**nkeys)

def read_deep_gal_cat(**keys):
    """
    Same parameters as get_file but ftype is set to 'deep_galaxy_catalog'
    and epoch is not required
    """
    import fitsio

    fname=get_deep_gal_cat_file(**keys)
    print('reading:',fname)

    data=fitsio.read(fname, lower=True)
    return data

def get_deep_star_cat_file(**keys):
    """
    Same parameters as get_file but ftype is set to 'deep_star_catalog'
    and epoch is not required
    """
    d=get_branch_dir(**keys)

    nkeys={}
    nkeys.update(keys)
    nkeys['ftype'] = 'deep_star_catalog'
    return get_file(**nkeys)

def read_deep_star_cat(**keys):
    """
    Same parameters as get_file but ftype is set to 'deep_star_catalog'
    and epoch is not required
    """
    import fitsio

    fname=get_deep_star_cat_file(**keys)
    print('reading:',fname)

    data=fitsio.read(fname, lower=True)
    return data

def get_deep_gal_image_file(**keys):
    """
    Same parameters as get_file but ftype is set to 'deep_image'
    """
    d=get_branch_dir(**keys)

    nkeys={}
    nkeys.update(keys)
    nkeys['ftype'] = 'deep_image'
    return get_file(**nkeys)

def read_deep_gal_image(**keys):
    """
    Same parameters as get_file but ftype is set to 'deep_image'
    and epoch is not required
    """
    import fitsio

    fname=get_deep_gal_image_file(**keys)
    print('reading:',fname)

    data=fitsio.read(fname)
    return data


def get_deep_star_image_file(**keys):
    """
    Same parameters as get_file but ftype is set to 'deep_starfield_image'
    """
    d=get_branch_dir(**keys)

    nkeys={}
    nkeys.update(keys)
    nkeys['ftype'] = 'deep_starfield_image'
    return get_file(**nkeys)

def read_deep_star_image(**keys):
    """
    Same parameters as get_file but ftype is set to 'deep_starfield_image'
    and epoch is not required
    """
    import fitsio

    fname=get_deep_star_image_file(**keys)
    print('reading:',fname)

    data=fitsio.read(fname)
    return data

def get_skynoise_dir():
    """
    Get the directory holding the sky noise estimates
    """
    d = get_dir()
    return os.path.join(d, 'skynoise')

def get_skynoise_file(**keys):
    """
    Get the file holding the sky noise estimates for each subfield
    """
    d=get_skynoise_dir()

    fname='%(experiment)s-%(obs_type)s-%(shear_type)s-skynoise.fits'
    fname = fname % keys

    fname=os.path.join(d, fname)
    return fname

def read_skynoise(**keys):
    """
    Read the sky noise for the indicated field
    """
    import fitsio
    fname=get_skynoise_file(**keys)
    print("reading:",fname)
    return fitsio.read(fname)

def get_deep_skynoise_file(**keys):
    """
    Get the file holding the sky noise estimates for each subfield
    """
    fname=get_skynoise_file(**keys)
    d=os.path.dirname(fname)
    bn=os.path.basename(fname)

    fname = 'deep-%s' % bn
    fname = os.path.join(d, fname)

    return fname

def read_deep_skynoise(**keys):
    """
    Read the sky noise for the indicated field
    """
    import fitsio
    fname=get_deep_skynoise_file(**keys)
    print("reading:",fname)
    return fitsio.read(fname)


def get_skynoise_plot_dir():
    """
    dir to hold plots
    """
    d=get_skynoise_dir()
    plot_dir=os.path.join(d, 'plots')
    return plot_dir

def get_skynoise_plot_file(**keys):
    """
    A plot of the fit
    """

    d=get_skynoise_plot_dir()

    fname='%(experiment)s-%(obs_type)s-%(shear_type)s-%(subid)03d-skynoise.eps'
    fname = fname % keys

    fname=os.path.join(d, fname)
    return fname

def get_deep_skynoise_plot_file(**keys):
    """
    plot of the fit
    """
    fname=get_skynoise_plot_file(**keys)
    d=os.path.dirname(fname)
    bn=os.path.basename(fname)

    fname = 'deep-%s' % bn
    fname = os.path.join(d, fname)

    return fname
