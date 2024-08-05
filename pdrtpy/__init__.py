"""Top level package for pdrtpy"""
#__all__ = [ "pdrutils", "measurement", "modelset"]
__version__ = "2.3.2b-95"
VERSION = __version__
AUTHORS =  'Marc W. Pound, Mark G. Wolfire'
DESCRIPTION="PhotoDissociation Region Toolbox (PDRT), astrophysics analysis tools"
KEYWORDS = "PDR photodissociation astronomy astrophysics"
# easter egg
motto = "Reliable astrophysics at everyday low, low prices! "+ u"\u00AE"

__all__ = [ "version"]

def version():
    """Version of the PDRT code

    :rtype: str
    """
    return __version__
