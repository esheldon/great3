# arcsec/pixel
#PIXEL_SCALE=0.2

NGAL_PER_SUBFIELD=10000

# number of "subfield"s; these are the 10x10 degree images
NSUB=200
NSUB_DEEP=5


# starting new values for these
DEFVAL      = -9999
PDEFVAL     =  9999
BIG_DEFVAL  = -9.999e9
BIG_PDEFVAL =  9.999e9

# main flag bits.  Everything from bit 1-29 is usable by the specific fitters
# but most likely they will have their more specific flag fields

PSF_FIT_FAILURE=2**0

# for different models. In the code w just set
# 2**(imodel+1)
GAL_FIT_FAILURE1=2**1
GAL_FIT_FAILURE2=2**2
GAL_FIT_FAILURE3=2**3
GAL_FIT_FAILURE4=2**4
GAL_FIT_FAILURE5=2**5

NO_ATTEMPT=2**30

# for regauss
RG_FAILURE = 2**1

# default checkpoint configuration
CHECKPOINTS_DEFAULT_MINUTES=[10,30,60,90]
