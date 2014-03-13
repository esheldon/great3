"""
File locations and reading for great3
"""
from __future__ import print_function
import os
import numpy

from .constants import *

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

def get_run_dir(**keys):
    """
    Get the directory holding the run
    """
    d = get_dir()
    return os.path.join(d, 'processing', keys['run'])

def get_output_dir(**keys):
    """
    Get the directory holding the outputs
    """
    rd=get_run_dir(**keys)

    return os.path.join(rd, 'output')

def get_output_file(**keys):
    """
    Get output file

    parameters
    ----------
    experiment: string
        Required keyword.  e.g. control, real
    obs_type: string
        Required keyword. e.g. ground, space
    shear_type: string
        Required keyword.  e.g. constant
    run: string
        Required keyword, the run id
    start: int, optional
        Star for obj range
    end: int, optional
        End for obj range
    """
    d=get_output_dir(**keys)

    nkeys={}
    nkeys.update(keys)
    obj_range=nkeys.get('obj_range',None)


    if obj_range is not None:
        nkeys['start'] = obj_range[0]
        nkeys['end'] = obj_range[1]
        fname='%(experiment)s-%(obs_type)s-%(shear_type)s-%(subid)03d-%(run)s-%(start)05d-%(end)05d.fits'
    else:
        fname='%(experiment)s-%(obs_type)s-%(shear_type)s-%(subid)03d-%(run)s.fits'

    fname = fname % nkeys

    return os.path.join(d, fname)

def read_output(**keys):
    """
    Same parameters as for get_output_file
    """
    import fitsio

    fname=get_output_file(**keys)
    print('reading:',fname)

    data=fitsio.read(fname)
    return data


def get_condor_dir(**keys):
    """
    Get the directory holding the condor files
    """
    rd=get_run_dir(**keys)

    return os.path.join(rd, 'condor')


def get_condor_file(**keys):
    """
    Get output file

    parameters
    ----------
    run: string
        Required keyword.  run id
    filenum: number
        An ordered number
    missing: bool
        for missing outputs
    """
    missing=keys.get('missing',False)

    d=get_condor_dir(**keys)

    fname='%(run)s' % keys

    if missing:
        fname += '-missing'

    fname = fname+'-%(filenum)03d.condor' % keys

    return os.path.join(d, fname)


def get_master_script_file(**keys):
    """
    the master script run by condor
    """
    d=get_condor_dir(**keys)
    return os.path.join(d, 'master.sh')

def write_fits_clobber(fname, data):
    """
    Write the data to the file, checking for directory
    existence and over-writing
    """
    import fitsio
    d=os.path.dirname(fname)
    if not os.path.exists(d):
        os.makedirs(d)

    print("writing:",fname)
    fitsio.write(fname, data, clobber=True)

def get_shear_file(**keys):
    """
    Get the shear file

    parameters
    ----------
    experiment: string
        Required keyword.  e.g. control, real
    obs_type: string
        Required keyword. e.g. ground, space
    shear_type: string
        Required keyword.  e.g. constant
    run: string
        Required keyword, the run id
    cut: string
        Required keyword representing the cuts
    """
    d=get_output_dir(**keys)

    nkeys={}
    nkeys.update(keys)
    obj_range=nkeys.get('obj_range',None)


    fname='%(experiment)s-%(obs_type)s-%(shear_type)s-%(run)s-%(cut)s.dat'

    fname = fname % nkeys

    return os.path.join(d, fname)

def read_shear(**keys):
    """
    Same parameters as for get_shear_file
    """

    fname=get_shear_file(**keys)
    print('reading:',fname)

    dlist=[]
    with open(fname) as fobj:
        for line in fobj:
            if line[0]=='#':
                continue
            vals=line.split()
            subid=int(vals[0])
            shear1=float(vals[1])
            shear2=float(vals[2])

            dlist.append( (subid, shear1, shear2) )

    n=len(dlist)
    out=numpy.zeros(n, dtype=[('subid','i4'),
                              ('shear','f8',2)])
    for i in xrange(n):
        d=dlist[i]
        out['subid'][i] = d[0]
        out['shear'][i,0] = d[1]
        out['shear'][i,1] = d[2]
    return out


def get_nsub(**keys):
    """
    Get the number of sub-field
    """
    from . import constants

    deep=keys.get('deep',False)
    if deep:
        return constants.NSUB_DEEP
    else:
        return constants.NSUB

def get_chunk_ranges(nper):
    """
    Get the chunk definitions for gals in a sub-field
    """
    
    nchunks = NGAL_PER_SUBFIELD/nper
    nleft = NGAL_PER_SUBFIELD % nper

    if nleft != 0:
        nchunks += 1

    low=[]
    high=[]

    for i in xrange(nchunks):

        low_i = i*nper

        # minus one becuase it is inclusive
        if i == (nchunks-1) and nleft != 0:
            high_i = low_i + nleft -1
        else:
            high_i = low_i + nper  - 1

        low.append( low_i )
        high.append( high_i )

    return low,high
