
class ToolBase(object):
    """ Base class object for PDR Toolbox tools.  This class implements a simple 
        interface with a run method.  Tools will generally do some set up
        such as reading in observational data before run() can be invoked.
    """
    def __init__(self):
        pass

    def run(self):
        """Runs the tool. Each subclass Tool must implement its own run() method.
        """
        pass
