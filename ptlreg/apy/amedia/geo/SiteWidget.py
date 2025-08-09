import numpy as np

from ptlreg.apy import *
from ptlreg.apy.core import AObject, AObjectList
from .Canvas2D import *
import drawsvg as draw
from drawsvg.widgets import DrawingWidget
from ptlreg.apy.utils.Colors import *

class Site(AObject):
    DefaultRadius = 15;

    def __init__(self, location=None, name=None, color=None, radius=None, **kwargs):
        super(Site, self).__init__(**kwargs);
        if(self.location is None):
            self.location = location;
        if(self.name is None):
            self.name = name;
        if(self.color is None):
            self.color = color;
        self.radius=radius;


    # <editor-fold desc="Property: 'location'">
    @property
    def location(self):
        return self.get_info("location");
    @location.setter
    def location(self, value):
        self.set_info('location', value);
    # </editor-fold>

    # <editor-fold desc="Property: 'name'">
    @property
    def name(self):
        return self.get_info("name");
    @name.setter
    def name(self, value):
        self.set_info('name', value);
    # </editor-fold>

    # <editor-fold desc="Property: 'color'">
    @property
    def color(self):
        return self.get_info("color");
    @color.setter
    def color(self, value):
        self.set_info('color', value);
    # </editor-fold>
    
    # <editor-fold desc="Property: 'radius'">
    @property
    def radius(self):
        r =  self.get_info("radius");
        if(r is None):
            return Site.DefaultRadius;
        else:
            return r;

    @radius.setter
    def radius(self, value):
        self.set_info('radius', value);
    # </editor-fold>
        


class SiteList(AObjectList):
    ElementClass=Site;
    def __init__(self, sites=None, **kwargs):
        if(isinstance(sites, (list, tuple, AObjectList))):
            if(isinstance(sites[0], Site)):
                super(SiteList, self).__init__(sites, **kwargs);
                return;
            else:
                sts = [];
                for s in sites:
                    sts.append(Site(s));
                super(SiteList, self).__init__(sts, **kwargs);
                return;
                # return super(FilePathList, self).__init__(fpaths, **kwargs);
        else:
            super(SiteList, self).__init__(sites, **kwargs);
            return;


class SiteWidget(DrawingWidget):
    SiteClass = Site;
    def __init__(self, sites, drawFunc, width=1080, height=720, mousedown=None, mousemove=None, mouseup=None, **kwargs):
        self.canvas = Canvas2D();
        if(isinstance(sites, (list, tuple, AObjectList))):
            self.sites = sites;
        else:
            self.sites = SiteList();
            n_sites = sites;
            colors = ColorScheme.GetSequential();
            for s in range(n_sites):
                # col = colors.getContinuousColor(s)
                col = np.array(colors.getContinuousColor(s/(n_sites-1))[:3])*255;
                col = col.astype(np.int);
                color = "rgb({},{},{})".format(int(col[0]),int(col[1]),int(col[2])),
                # col = 'rgb({},{},{})'.format(colors.color[s][0],colors.color[s][1],colors.color[s][2]);
                print(color[0])
                newsite = Site(
                    location=P2D(np.random.rand()*height, np.random.rand()*width),
                    color = color[0],
                    name="{}".format(s)
                )
                self.sites.append(newsite);


        self.drawFunc = drawFunc;
        super(SiteWidget, self).__init__(self.canvas);
        self._grabpoint = None;
        if(mousedown is not None):
            self._mouseDownFunc = mousedown;
            # self.mousedown(mousedown);
        else:
            self._mouseDownFunc = self._default_mousedown;
            # self.mousedown(SiteWidget._default_mousedown);
        if(mousemove is not None):
            self._mouseMoveFunc = mousemove;
            # self.mousemove(mousemove);
        else:
            # self.mousemove(self._default_mousemove);
            self._mouseMoveFunc = self._default_mousemove;
        if(mouseup is not None):
            self._mouseUpFunc = mouseup;
            # self.mouseup(self._default_mouseup);
        else:
            self._mouseUpFunc = self._default_mouseup;

        self.mousedown(self._mousedown);
        self.mousemove(self._mousemove);
        self.mouseup(self._mouseup);
        self.redrawWidget();

    def drawPoints(self):
        for s in self.sites:
            color = "#cccccc"
            if(s.color is not None):
                color = s.color;
            self.canvas.drawPoint(s.location, label=s.name, color=color);
            # print(s);

    def redrawWidget(self):
        self.canvas.clear();
        self.drawFunc(self);
        self.refresh();

    @staticmethod
    def _mousedown(self, x, y, info):
        self._mouseDownFunc(x, y, info)

    @staticmethod
    def _mousemove(self, x, y, info):
        self._mouseMoveFunc(x, y, info)

    @staticmethod
    def _mouseup(self, x, y, info):
        self._mouseUpFunc(x, y, info)


    def _default_mousedown(self, x, y, info, **kwargs):
        clickpoint = Vec3(x,y,1);
        for p in range(len(self.sites)):
            if((clickpoint-self.sites[p].location).L2()<(self.sites[p].radius)):
                self._grabpoint = p;
                break;
        self.redrawWidget()
        # print("{}, {}, {}".format(x,y,info));
        # self.refresh();


    def _default_mousemove(self, x, y, info, **kwargs):
        if(self._grabpoint is not None):
            self.sites[self._grabpoint].location.x = x;
            self.sites[self._grabpoint].location.y = y;
            self.redrawWidget();



    def _default_mouseup(self, x, y, info, **kwargs):
        self._grabpoint = None;
        self.redrawWidget();