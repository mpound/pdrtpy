__all__ = [ "pdrutils", "measurement", "modelset"]

__version__ "2.3.2b"
VERSION = __version__
AUTHORS =  'Marc W. Pound, Mark G. Wolfire'
DESCRIPTION="PhotoDissociation Region Toolbox (PDRT), astrophysics analysis tools"
KEYWORDS = "PDR photodissociation astronomy astrophysics"

# easter egg
motto = "Reliable astrophysics at everyday low, low prices! "+ u"\u00AE"

def version():
    """Version of the PDRT code

    :rtype: str
    """
    return __version__
