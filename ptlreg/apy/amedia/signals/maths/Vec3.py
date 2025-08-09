import numpy as np
from .NDArray import NDArray

class Vec3(NDArray):
    def __init__(self, x=None, y=None, z=None, **kwargs):

        if(x is not None and isinstance(x,(tuple, list, np.ndarray, Vec3, NDArray))):
            super(Vec3, self).__init__(data=x, **kwargs);
        elif(y is not None and z is not None):
            super(Vec3, self).__init__(data=[x,y,z], **kwargs);
        else:
            raise NotImplementedError;

    @property
    def selfclass(self):
        return type(self);

    # <editor-fold desc="Property: 'x'">
    @property
    def x(self):
        return self._ndarray[0];
    @x.setter
    def x(self, value):
        self._ndarray[0]=value;
    # </editor-fold>

    # <editor-fold desc="Property: 'x'">
    @property
    def y(self):
        return self._ndarray[1];
    @y.setter
    def y(self, value):
        self._ndarray[1] = value;
    # </editor-fold>

    @property
    def z(self):
        return self._ndarray[2];

    @z.setter
    def z(self, value):
        self._ndarray[2]=value;

    def dot(self, other):
        return self.x*other.x+self.y*other.y+self.z*other.z;


    def cross(self, other):
        if(hasattr(other, 'z')):
            return self.selfclass(np.cross([self.x, self.y, self.z], [other.x, other.y, other.z])).get_homogenized();
        else:
            return self.selfclass(np.cross([self.x, self.y, self.z], other)).get_homogenized();

    def get_homogenized(self):
        rval=self.selfclass(self._ndarray);
        rval.homogenize();
        return rval;


    def get_normalized(self):
        rval=self.selfclass(self._ndarray);
        rval.normalize();
        return rval;


def P2D(x,y):
    return Vec3(x,y,1);

def Dist2D(A,B):
    return np.linalg.norm([A.x-B.x, A.y-B.y]);