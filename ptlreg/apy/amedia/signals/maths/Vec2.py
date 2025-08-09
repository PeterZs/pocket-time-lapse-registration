from .NDArray import *
import numpy as np


class Vec2(NDArray):
    def __init__(self, x=None, y=None, **kwargs):

        if(x is not None and isinstance(x,(tuple, list, np.ndarray, Vec2, NDArray))):
            super(Vec2, self).__init__(data=x, **kwargs);
        elif(y is not None):
            super(Vec2, self).__init__(data=[x,y], **kwargs);
        elif(x is not None):
            #  this would be weird, it would mean x is not a tuple and not none, but y is none...
            super(Vec2, self).__init__(data=[x, 0], **kwargs);
        else:
            super(Vec2, self).__init__(shape=2, **kwargs);

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
        return 1;

    def cross(self, other):
        from a3py.pointline.Vec3 import Vec3
        if(hasattr(other, 'z')):
            return Vec3(np.cross([self.x, self.y, self.z], [other.x, other.y, other.z]));
        else:
            return Vec3(np.cross([self.x, self.y, self.z], other));

def P2D(x,y=None):
    return Vec2(x,y);

def Dist2D(A,B):
    return np.linalg.norm([A.x-B.x, A.y-B.y]);
