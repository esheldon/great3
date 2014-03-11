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
        Optional keyword.  Default 0.
    """
    d=get_branch_dir(**keys)

    if 'epoch' not in keys:
        keys['epoch'] = 0

    fname='%(ftype)s-%(subid)03d-%(epoch)d.fits' 
    fname = fname % keys

    fname = os.path.join(d, fname)
    return fname

def get_gal_cat_file(**keys):
    """
    Same parameters as get_file but ftype is set to 'galaxy_catalog'
    """
    d=get_branch_dir(**keys)

    keys['ftype'] = 'galaxy_catalog'
    return get_file(**keys)

def get_star_cat_file(**keys):
    """
    Same parameters as get_file but ftype is set to 'star_catalog'
    """
    d=get_branch_dir(**keys)

    keys['ftype'] = 'star_catalog'
    return get_file(**keys)


def get_gal_image_file(**keys):
    """
    Same parameters as get_file but ftype is set to 'image'
    """
    d=get_branch_dir(**keys)

    keys['ftype'] = 'image'
    return get_file(**keys)

def get_star_image_file(**keys):
    """
    Same parameters as get_file but ftype is set to 'starfield_image'
    """
    d=get_branch_dir(**keys)

    keys['ftype'] = 'starfield_image'
    return get_file(**keys)
