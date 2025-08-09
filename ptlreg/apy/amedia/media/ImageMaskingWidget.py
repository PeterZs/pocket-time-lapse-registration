from ptlreg.apy.amedia.media import *
import ipywidgets as widgets
import IPython.display

class ImageMaskingWidget():
    def __init__(self,im):
        self.source_image = im
        self.mask = np.zeros([im.shape[0], im.shape[1]]);
        self.drag_start = None;
        self.fig,self.ax = plt.subplots()
        self.bgpix = self.getBackgroundPixels();
        self.img = self.ax.imshow(self.bgpix);
        self.down_event = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.up_event = self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        disconnect_button = widgets.Button(description="Disconnect mpl")
        IPython.display.display(disconnect_button)
        disconnect_button.on_click(self.disconnect_mpl)
        self.release_events = [];

    def getBackgroundPixels(self):
        return self.source_image.fpixels.copy() * 0.75;

    def onclick(self, event):
        print(event)
        self.drag_start = [int(event.ydata),int(event.xdata)];
        self.fig
        self.img.set_data(self.bgpix);

    def onrelease(self, event=None):
        if(event is not None):
            self.release_events.append(event);
            drag_end = [int(event.ydata),int(event.xdata)];
            self.mask[self.drag_start[0]:drag_end[0],self.drag_start[1]:drag_end[1]]=1;
        self.bgpix[self.mask>0]=[0,1.0,0];
        plt.draw()
        self.img.set_data(self.bgpix);

    def disconnect_mpl(self,_):
        self.fig.canvas.mpl_disconnect(self.down_event);
        self.fig.canvas.mpl_disconnect(self.up_event);

    def GetMask(self):
        return self.mask;

    def GetMaskRGB(self):
        mask = self.mask;
        return np.dstack((mask, mask, mask))

    def GetMaskImage(self):
        return Image(pixels=self.GetMask());

    def GetMaskImageRGB(self):
        return Image(pixels=self.GetMaskRGB());

    def GetMaskedImage(self):
        imcopy = self.source_image.clone();
        imcopy.pixels = imcopy.pixels*self.mask;
        return imcopy;