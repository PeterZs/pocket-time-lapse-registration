from PIL import ImageDraw as _ImageDraw
from .Image import *

@ImageMethod
def GetWithIndexedLines(self, vlist, elist, **kwargs):
    ipil = PIM.fromarray(np.uint8(self.ipixels.copy()));
    draw = _ImageDraw.Draw(ipil)
    for e in elist:
        line = [(vlist[e[0]][0]*self.width, vlist[e[0]][1]*self.height),
                (vlist[e[1]][0]*self.width, vlist[e[1]][1]*self.height)]
        draw.line(line, **kwargs);
    newpix = np.array(ipil.getdata());
    newpix.shape=self.pixels.shape;
    return Image(pixels=newpix);

@ImageMethod
def GetWithPoints(self, plist, radius=None, outline=None, fill=None, width=None):
    if(radius is None):
        radius=self.width*0.01;

    if(outline is None):
        outline = (0,255,0);
    if(fill is None):
        fill = outline;
    if(width is None):
        width=1;
    ipil = PIM.fromarray(np.uint8(self.ipixels.copy()));
    draw = _ImageDraw.Draw(ipil)
    for v in plist:
        draw.ellipse([(v[0]-radius, v[1]-radius), (v[0]+radius, v[1]+radius)], outline=outline, fill=fill, width=width);
        # draw.point(v, **kwargs);
    newpix = np.array(ipil.getdata());
    newpix.shape=self.pixels.shape;
    return Image(pixels=newpix);