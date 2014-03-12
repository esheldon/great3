# arcsec/pixel
PIXEL_SCALE=0.2

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
NO_ATTEMPT=2**30

# for rg
RG_MAX_ELLIP = 4.0
RG_MAX_R     = 1.0
RG_MIN_R     = 0.01

RG_FAILURE = 2**1

# default checkpoint configuration
CHECKPOINTS_DEFAULT_MINUTES=[10,30,60,90]
