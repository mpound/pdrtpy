import numpy as np
from astropy.nddata import NDData


class FitMap(NDData):
    def __init__(self, data, *args, **kwargs):
        """ A class that can store fit objects in a data array but has all the nice WCS properties of NDData.

        :param data: the data set, an array of lmfit.model.ModelResult or lmfit.minimizer.MinimizerResult
        :type data: :class:`numpy.ndarray`-like
        :param name: an identifying name for this object
        :type name: str
        """
        debug = kwargs.pop('debug', False)
        if debug:
            print("args=",*args)
            print("kwargs=",*kwargs)
        self._name = kwargs.pop('name',None)

        # NDData wants a nddata array so give it a fake one
        # and sub our object array afterwards
        _data = np.zeros(data.shape)
        super().__init__(_data,*args,**kwargs)
        self._data = data
        if np.shape(self._data) == ():
            self._data = np.array([self._data])

    @property
    def name(self):
        """The name of this FitMap
        :rtype: str
        """
        return self._name

    def __getitem__(self,i):
        """get the value object at array index i"""
        return self._data[i]

    def get_pixel(self,world_x,world_y):
        '''Return the nearest pixel coordinates to the input world coordinates

        :param world_x: The horizontal world coordinate
        :type world_x: float
        :param world_y: The vertical world coordinate
        :type world_y: float
        '''
        if self.wcs is None:
            raise Exception(f"No wcs in this FitMap {self.name}")
        return tuple(np.round(self.wcs.world_to_pixel_values(world_x,world_y)).astype(int))
