"""
File locations and reading for great3
"""
from __future__ import print_function
import os
import numpy

from .constants import *

def get_config_dir():
    """
    get the config directory
    """
    d=os.environ['GREAT3_CONFIG_DIR']
    return d

def get_config_file(run):
    """
    get the config directory
    """
    d=get_config_dir()
    fname='run-%s.yaml' % run
    return os.path.join(d,fname)

def read_config(run):
    """
    get the config directory
    """
    import yaml
    fname=get_config_file(run)
    conf=yaml.load(open(fname))

    mess="mismatch between runs: '%s' vs '%s'" % (run,conf['run'])
    assert conf['run']==run,mess

    return conf


def get_dir(**keys):
    """
    get my great3 run dir

    parameters
    ----------
    great3run: string
        One of my great3 runs, e.g. run01

    The GREAT3_DATA_DIR environment variable must be set
    """

    great3run=keys['great3run']
    d=os.environ['GREAT3_DATA_DIR']

    #d=os.path.join(d,great3run,'data')
    d=os.path.join(d,great3run,'data','public')
    return d

def get_truth_dir(**keys):
    """
    get truth directory

    parameters
    ----------
    great3run: string
        One of my great3 runs, e.g. run01

    The GREAT3_DATA_DIR environment variable must be set
    """

    great3run=keys['great3run']
    d=os.environ['GREAT3_DATA_DIR']

    #d=os.path.join(d,great3run,'data')
    d=os.path.join(d,great3run,'data','truth')
    return d


def get_branch_dir(**keys):
    """
    $GREAT3_DATA_DIR/experiment/obs_type/shear_type
    e.g.      /control/ground/constant

    parameters
    ----------
    great3run: string
        One of my great3 runs, e.g. run01
    experiment: string
        Required keyword.  e.g. control, real
    obs_type: string
        Required keyword. e.g. ground, space
    shear_type: string
        Required keyword.  e.g. constant
    """
    d=get_dir(**keys)

    experiment=keys['experiment']
    obs_type=keys['obs_type']
    shear_type=keys['shear_type']

    d = os.path.join(d, experiment, obs_type, shear_type)
    return d

def get_file(**keys):
    """
    parameters
    ----------
    great3run: string
        One of my great3 runs, e.g. run01
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

    great3run: string
        One of my great3 runs, e.g. run01
    experiment: string
        Required keyword.  e.g. control, real
    obs_type: string
        Required keyword. e.g. ground, space
    shear_type: string
        Required keyword.  e.g. constant
    subid: number
        Requird keyword. e.g. 12
    epoch: number
        Optional keyword.  Default 0 for image types, none for catalogs.
    """
    d=get_branch_dir(**keys)

    nkeys={}
    nkeys.update(keys)
    nkeys['ftype'] = 'image'
    return get_file(**nkeys)

def count_gal_image_files(**keys):
    """
    count all that exist
    """
    subid=0
    while True:
        keys['subid']=subid

        fname=get_gal_image_file(**keys)
        if not os.path.exists(fname):
            break

        subid += 1

    return subid
 
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

def count_deep_gal_image_files(**keys):
    """
    count all that exist
    """
    subid=0
    while True:
        keys['subid']=subid

        fname=get_deep_gal_image_file(**keys)
        if not os.path.exists(fname):
            break

        subid += 1

    return subid
        

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

def get_skynoise_dir(**keys):
    """
    Get the directory holding the sky noise estimates

    parameters
    ----------
    great3run: string
        One of my great3 runs, e.g. run01
    """
    d = get_dir(**keys)
    return os.path.join(d, 'skynoise')

def get_skynoise_file(**keys):
    """
    Get the file holding the sky noise estimates for each subfield
    """
    d=get_skynoise_dir(**keys)

    #fname='%(experiment)s-%(obs_type)s-%(shear_type)s-skynoise.fits'
    fname='%(experiment)s-%(obs_type)s-%(shear_type)s-%(subid)06d-skynoise.fits'
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


def get_skynoise_plot_dir(**keys):
    """
    dir to hold plots
    """
    d=get_skynoise_dir(**keys)
    plot_dir=os.path.join(d, 'plots')
    return plot_dir

def get_skynoise_plot_file(**keys):
    """
    A plot of the fit
    """

    d=get_skynoise_plot_dir(**keys)

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
    d=os.environ['GREAT3_DATA_DIR']
    run=keys['run']
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

def get_prior_file(**keys):
    """
    Files concerning the distribution of parameters in the deep fields
    """
    d=get_output_dir(**keys)

    if 'subid' in keys:
        fname='%(experiment)s-%(obs_type)s-%(shear_type)s-%(subid)03d-%(run)s-%(partype)s-dist.%(ext)s'
    else:
        fname='%(experiment)s-%(obs_type)s-%(shear_type)s-%(run)s-%(partype)s-dist.%(ext)s'

    fname = fname % keys

    return os.path.join(d, fname)

def read_prior(**keys):
    """
    read the prior file
    """
    import fitsio
    fname=get_prior_file(**keys)
    print("reading:",fname)
    return fitsio.read(fname)


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

def get_wq_file(**keys):
    """
    Get the wq file, is in the condor dir

    parameters
    ----------
    run: string
        Required keyword.  run id
    obj_range: [low,high]
        Objects to process
    missing: bool
        for missing outputs
    """

    missing=keys.get('missing',False)

    d=get_condor_dir(**keys)

    fname='%(run)s' % keys

    if missing:
        fname += '-missing'

    first,last=keys['obj_range']
    fname = fname+'-%(subid)03d-%(first)04d-%(last)04d.yaml' % keys

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
    model: string
        Model fit to use
    """
    d=get_output_dir(**keys)

    obj_range=keys.get('obj_range',None)

    with_psf=keys.get('with_psf',False)

    if with_psf:
        fname='%(experiment)s-%(obs_type)s-%(shear_type)s-%(run)s-%(model)s-%(cut)s-extra.dat'
    else:
        fname='%(experiment)s-%(obs_type)s-%(shear_type)s-%(run)s-%(model)s-%(cut)s.dat'

    fname = fname % keys

    return os.path.join(d, fname)

def read_shear(**keys):
    """
    Same parameters as for get_shear_file
    """

    with_psf=keys.get('with_psf',False)

    if with_psf:
        return read_shear_with_psf(**keys)

    dt=[('subid','i4'), ('shear','f8',2)]

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
    out=numpy.zeros(n, dtype=dt)
    for i in xrange(n):
        d=dlist[i]
        out['subid'][i] = d[0]
        out['shear'][i,0] = d[1]
        out['shear'][i,1] = d[2]
    return out

def read_shear_with_psf(**keys):
    """
    Same parameters as for get_shear_file
    """

    with_psf=keys.get('with_psf',False)

    dt=[('subid','i4'),
        ('shear','f8',2),
        ('shear_err','f8',2),
        ('psf_g','f8',2)]

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

            err1=float(vals[3])
            err2=float(vals[4])
            psf_g1=float(vals[5])
            psf_g2=float(vals[6])
            dlist.append( (subid, shear1, shear2,err1,err1,psf_g1,psf_g2) )

    n=len(dlist)
    out=numpy.zeros(n, dtype=dt)
    for i in xrange(n):
        d=dlist[i]
        out['subid'][i] = d[0]
        out['shear'][i,0] = d[1]
        out['shear'][i,1] = d[2]
        out['shear_err'][i,1] = d[3]
        out['shear_err'][i,1] = d[4]
        out['psf_g'][i,0] = d[5]
        out['psf_g'][i,1] = d[6]

    return out


def read_shear_subid(**keys):
    """
    Get the shear for a particular subid
    """
    if 'subid' not in keys:
        raise ValueError("send subid")

    data=read_shear(**keys)

    w,=numpy.where(data['subid']==keys['subid'])

    return data['shear'][ w[0], :]

def read_shear_expand(**keys):
    """
    Get the shear given a shear_expand_run and shear_expand_cut
    """
    shear_expand_run = keys['shear_expand_run']
    shear_expand_cut = keys['shear_expand_cut']

    cc={}
    cc.update(**keys)

    cc['run'] = shear_expand_run
    cc['cut'] = shear_expand_cut
    shear_expand=read_shear_subid(**cc)

    return shear_expand


def get_nsub(**keys):
    """
    Get the number of sub-field
    """
    from . import constants

    deep=keys.get('deep',False)
    if deep:
        nsub=count_deep_gal_image_files(**keys)
    else:
        nsub=count_gal_image_files(**keys)

    return nsub

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

def get_storage_dir():
    tdir=os.environ['TMPDIR']
    return os.path.join(tdir,'great3-storage')



def get_plot_dir(**keys):
    """
    Get the directory holding the outputs
    """
    rd=get_run_dir(**keys)

    return os.path.join(rd, 'plots')


def get_plot_file(**keys):
    """
    Get plot file

    parameters
    ----------
    run: string
        Required keyword, the run id
    extra: string
        Extra stuff on the end of the file name
    """
    d=get_plot_dir(**keys)

    fname=['%(run)s' % keys]

    if 'extra' in keys:
        fname += [keys['extra']]

    fname='-'.join(fname)

    fname='%s.eps' % fname

    return os.path.join(d, fname)


