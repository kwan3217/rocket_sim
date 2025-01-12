"""
Animate "Exponential Beats All"
"""

from picturebox import *
from kwanmath.interp import linterp,trap
from collections.abc import Iterable
import numpy as np
import os
import pathlib
import matplotlib

class Stage:
    w0 = 1920
    h0 = 1080

    def __init__(self,w=None,h=None,f0=0,f1=100,shadow=False,facecolor='#e0e0ff'):
        self.actors=[]
        self.w=Stage.w0 if w is None else w
        self.h=Stage.h0 if h is None else h
        self.f0=f0
        self.f1=f1
        self.shadow=shadow
        self.facecolor=facecolor
    def setup(self,pb:PictureBox):
        pass
    def teardown(self,pb:PictureBox):
        pass
    def perform(self):
        digits=len(str(self.f1))
        oupath=f"render/images/{os.path.basename(__file__)[:-3]}/{type(self).__name__}/"
        pathlib.Path(oupath).mkdir(parents=True,exist_ok=True)
        oufn_pat=oupath+f"{type(self).__name__}%0{digits}d.png"
        with PictureBox(self.w,self.h,title=type(self).__name__,facecolor=self.facecolor) as pb:
            self.setup(pb)
            perform(pb,self.actors,self.f0,self.f1,shadow=self.shadow,oufn_pat=oufn_pat)
            self.teardown(pb)

class cpl:
    """
    Continuous Piecewise Linear function
    """
    def __init__(self,ys):
        """

        :param ys: Table of y values. Each value is the value of the
                   function at the beginning of the corresponding phase.
                   Must have at least one more value than intended phases.
        """
        self.ys=ys
    def __call__(self,phase,t):
        if phase<0:
            phase=len(self.ys)+phase-1
        return linterp(0,self.ys[phase],1,self.ys[phase+1],t)


class Ticks(EnterActor):
    def _enter(self,pb:PictureBox,tt:float,alpha:float=1.0,shadow:bool=False,
             u0:float=None,u1:float=None,du:float=1.0,
             px0:float=None,dx0:float=None,px1:float=None,dx1:float=None,tx0:float=None,tx1:float=None,
             py0:float=None,dy0:float=None,py1:float=None,dy1:float=None,ty0:float=None,ty1:float=None,
             lx:float=None,ly:float=None,lfmt:str=None,
             **kwargs)->None:
        """
        Draw a series of evenly spaced ticks along a straight line

        :param pb: PictureBox to draw on
        :param tt: Time parameter in this phase
        :param alpha: Transparency
        :param shadow: True if on the shadow pass
        :param u0: lowest numbered tick
        :param u1: highest numbered tick
        :param du: spacing between ticks in data space
        :param px0: Pixel horizontal coordinate of one end of the line
        :param dx0: Data horizontal coordinate of one end of the line
        :param px1: Pixel horizontal coordinate of the other end of the line
        :param dx1: Data horizontal coordinate of the other end of the line
        :param tx0: Horizontal Offset of one end of each tick in pixels from its spot
        :param tx1: Horizontal Offset of the other end of each tick in pixels from its spot
        :param py0: Pixel vertical coordinate of one end of the line
        :param dy0: Data vertical coordinate of one end of the line
        :param py1: Pixel vertical coordinate of the other end of the line
        :param dy1: Data vertical coordinate of the other end of the line
        :param ty0: Vertical offset of one end of each tick in pixels from its spot
        :param ty1: Vertical offset of the other end of each tick in pixels from its spot
        :param lx: if non-None, print numerical labels at each tick, offset this many pixels horizontally
        :param ly: if non-None, print numerical labels at each tick, offset this many pixels vertically
        :param fmt: If non-None, format the tick number with this % string
        :param kwargs: Passed to pb.line
        """
        if shadow:
            return
        if alpha==0:
            return
        for u in np.arange(u0,u1+0.00001,du):
            if linterp(u0,0,u1,1,u)>tt:
                continue
            #Data coordinates of tick
            dx=linterp(u0,dx0,u1,dx1,u)
            dy=linterp(u0,dy0,u1,dy1,u)
            #Pixel coordinates of tick
            px=px0 if px0==px1 else linterp(dx0,px0,dx1,px1,dx)
            py=py0 if py0==py1 else linterp(dy0,py0,dy1,py1,dy)
            if tx0 is not None:
                #Draw the line
                pb.line(px+tx0,py+ty0,px+tx1,py+ty1,alpha=alpha,**kwargs)
            if lx is not None:
                #Print the label
                pb.text(px+lx,py+ly,lfmt%u,alpha=alpha,font=pathlib.Path('fonts/texgyreadventor-regular.otf'),**kwargs)

def LabeledAxis(ts:Iterable[float]=None,
                ticklen:float=30,size:float=22.5,color='k',alpha=1.0,
                px0:float=None,dx0:float=None,px1:float=None,dx1:float=None,dux:float=1.0,
                py0:float=None,dy0:float=None,py1:float=None,dy1:float=None,duy:float=1.0,
                lfmtx:str='%.0f',lfmty:str='%.0f'):
    denter = ts[1] - ts[0]
    return [
        Axis(x0=px0, x1=px1, y0=py1, y1=py0, color='k', ts=ts, alpha=alpha),
        # X axis ticks
        Ticks(ts=[ts[0]+denter*(1/6+1/3),ts[0]+denter*(1/6+3/3)]+ts[2:],
              u0=dx0, u1=dx1, du=dux,
              px0=px0, dx0=dx0, px1=px1, dx1=dx1, tx0=0, tx1=0,
              py0=py0, dy0=dy0, py1=py0, dy1=dy0, ty0=0, ty1=ticklen,
              color=color,alpha=alpha),
        Ticks(ts=[ts[0]+denter*(1/6+1/3),ts[0]+denter*(1/6+3/3)]+ts[2:],
                  u0=dx0, u1=dx1, du=dux,
                  px0=px0, dx0=dx0, px1=px1, dx1=dx1,
                  py0=py0, dy0=dy0, py1=py0, dy1=dy0,
                  lx=-5, ly=ticklen, size=size, lfmt=lfmtx, ha='right', va='baseline',
                  color=color,alpha=alpha),
        # Y axis ticks
        Ticks(ts=[ts[0]+denter*(1/6+0/3),ts[0]+denter*(1/6+2/3)]+ts[2:],
                   u0=dy0, u1=dy1, du=duy,
                   px0=px0, dx0=dx0, px1=px0, dx1=dx0, tx0=-ticklen, tx1=0,
                   py0=py0, dy0=dy0, py1=py1, dy1=dy1, ty0=0, ty1=0,
                   color=color,alpha=alpha),
        Ticks(ts=[ts[0]+denter*(1/6+0/3),ts[0]+denter*(1/6+2/3)]+ts[2:],
                   u0=dy0, u1=dy1, du=duy,
                   px0=px0, dx0=dx0, px1=px0, dx1=dx0,
                   py0=py0, dy0=dy0, py1=py1, dy1=dy1,
                   lx=-5, ly=ticklen, size=size, lfmt=lfmty, ha='right', va='baseline',
                   color=color,alpha=alpha)
    ]


def main():
    matplotlib.use('agg')
    f=lambda x:x**2
    g=lambda x:np.exp(x)

    dx0=0
    dx1=5
    dy0=0
    dy1=np.ceil((f(dx1) if f(dx1)>g(dx1) else g(dx1))/25)*25
    px0=150
    px1=Stage.w0-150
    py0=Stage.h0-150
    py1=150
    if False:  #Fade in the competitors, f(x)=x**2 and g(x)=exp(x)
        class IntroduceFormulas(Stage):
            def __init__(self):
                super().__init__(f0=0,f1=100)
                self.actors.append(Text(ts=[0,30,100,100],x=450,y=450,size=45,s='$f(x)=x^2$',color='r',usetex=True))
                self.actors.append(Text(ts=[30,60,100,100],x=450,y=600,size=45,s='$g(x)=e^x$',color='b',usetex=True))
        IntroduceFormulas().perform()
    if False:  #Draw the formulas, and the curves.
        class IntroduceCurves(Stage):
            def __init__(self):
                super().__init__(f0=0,f1=100)
                self.actors.append(Text(ts=[0,0,100,100],x=450,y=450,size=45,s='$f(x)=x^2$',color='r',usetex=True))
                self.actors.append(Text(ts=[0,0,100,100],x=450,y=600,size=45,s='$g(x)=e^x$',color='b',usetex=True))
                self.actors=self.actors+LabeledAxis(ts=[0,30,100,100],
                                          px0=px0,dx0=dx0,px1=px1,dx1=dx1,dux=1,
                                          py0=py0,dy0=dy0,py1=py1,dy1=dy1,duy=25)
                self.actors.append(Function(ts=[20,70,100,100],px0=px0,px1=px1,dx0=dx0,dx1=dx1,py0=py0,py1=py1,dy0=dy0,dy1=dy1,f=lambda phase,tt:lambda x:f(x),color='r'))
                self.actors.append(Function(ts=[40,90,100,100],px0=px0,px1=px1,dx0=dx0,dx1=dx1,py0=py0,py1=py1,dy0=dy0,dy1=dy1,f=lambda phase,tt:lambda x:g(x),color='b'))
        IntroduceCurves().perform()
    if False: #Crank up the exponent on the x curve
        class CrankExponent(Stage):
            def __init__(self):
                super().__init__(f0=0,f1=100)
                self.actors.append(Text(ts=[0,0,100,100],x=450,y=450,size=45,s=lambda phase,t:'$f(x)=x^{'+"%.2f"%(2 if phase==0 else linterp(0,2,1,5,t) if phase==1 else 5)+"}$",color='r',usetex=True))
                self.actors.append(Text(ts=[0,0,100,100],x=450,y=600,size=45,s='$g(x)=e^x$',color='b',usetex=True))
                self.actors=self.actors+LabeledAxis(ts=[0,0,100,100],
                                          px0=px0,dx0=dx0,px1=px1,dx1=dx1,dux=1,
                                          py0=py0,dy0=dy0,py1=py1,dy1=dy1,duy=25)
                self.actors.append(Function(ts=[0,0,100,100],px0=px0,px1=px1,dx0=dx0,dx1=dx1,py0=py0,py1=py1,dy0=dy0,dy1=dy1,f=lambda phase,tt:lambda x: x**linterp(0,2,1,5,tt),color='r'))
                self.actors.append(Function(ts=[0,0,100,100],px0=px0,px1=px1,dx0=dx0,dx1=dx1,py0=py0,py1=py1,dy0=dy0,dy1=dy1,f=lambda phase,tt:lambda x: g(x),color='b'))
        CrankExponent().perform()
    dy1a=dy1
    dy1b=400000
    dy1=lambda phase,tt:np.exp(linterp(0,np.log(dy1a+1),1,np.log(dy1b),tt))
    if False:
        class VertScale(Stage):
            def __init__(self):
                super().__init__(f0=0,f1=100)
                self.actors.append(Text(ts=[0,0,100,100],x=450,y=450,size=45,s='$f(x)=x^5$',color='r',usetex=True))
                self.actors.append(Text(ts=[0,0,100,100],x=450,y=600,size=45,s='$g(x)=e^x$',color='b',usetex=True))
                #Ticked X axis, doesn't need to be repeated for other axes
                self.actors=self.actors+LabeledAxis(ts=[0,0,100,100],
                                          px0=px0,dx0=dx0,px1=px1,dx1=dx1,dux=1,
                                          py0=py0,dy0=dy0,py1=py1,dy1=dy1,duy=9e9)
                #First Y axis, labeled every 25 units, doesn't label X axis, fades out once it reaches 300
                self.actors=self.actors+LabeledAxis(ts=[0,0,100,100],
                                          px0=px0,dx0=dx0,px1=px1,dx1=dx1,dux=dx1+1,
                                          py0=py0,dy0=dy0,py1=py1,dy1=dy1,duy=25,alpha=lambda phase,tt:1 if phase==0 else 0 if phase==-1 else linterp(0.07,1,0.17,0,tt,bound=True) )
                #First Y axis, labeled every 100 units, doesn't label X axis, fades out once it reaches 1000
                self.actors=self.actors+LabeledAxis(ts=[0,0,100,100],
                                          px0=px0,dx0=dx0,px1=px1,dx1=dx1,dux=dx1+1,
                                          py0=py0,dy0=dy0,py1=py1,dy1=dy1,duy=100,alpha=lambda phase,tt:1 if phase==0 else 0 if phase==-1 else linterp(0.17,1,0.27,0,tt,bound=True) )
                #Second Y axis, labeled every 1000, fades out once it reaches 10000
                self.actors=self.actors+LabeledAxis(ts=[0,0,100,100],
                                          px0=px0,dx0=dx0,px1=px1,dx1=dx1,dux=dx1+1,
                                          py0=py0,dy0=dy0,py1=py1,dy1=dy1,duy=1000,alpha=lambda phase,tt:0 if phase==0 else 0 if phase==-1 else linterp(0.46,1,0.56,0,tt,bound=True) )
                #Third Y axis, labeled every 10000, fades out once it reaches 50000
                self.actors=self.actors+LabeledAxis(ts=[0,0,100,100],
                                          px0=px0,dx0=dx0,px1=px1,dx1=dx1,dux=dx1+1,
                                          py0=py0,dy0=dy0,py1=py1,dy1=dy1,duy=10000,alpha=lambda phase,tt:0 if phase==0 else 0 if phase==-1 else linterp(0.75,1,0.85,0,tt,bound=True) )
                #Fourth Y axis, labeled every 100000, runs to 400000
                self.actors=self.actors+LabeledAxis(ts=[0,0,100,100],
                                          px0=px0,dx0=dx0,px1=px1,dx1=dx1,dux=dx1+1,
                                          py0=py0,dy0=dy0,py1=py1,dy1=dy1,duy=100000)
                self.actors.append(Function(ts=[0,0,100,100],px0=px0,px1=px1,dx0=dx0,dx1=dx1,py0=py0,py1=py1,dy0=dy0,dy1=dy1,f=lambda phase,tt:lambda x: x**5,color='r'))
                self.actors.append(Function(ts=[0,0,100,100],px0=px0,px1=px1,dx0=dx0,dx1=dx1,py0=py0,py1=py1,dy0=dy0,dy1=dy1,f=lambda phase,tt:lambda x: g(x),color='b'))
        VertScale().perform()
    dy1=dy1b
    dx1a=dx1
    dx1b=13
    dx1=lambda phase, tt: linterp(0, dx1a, 1, dx1b, tt)
    if False:
        class HorizScale(Stage):
            def __init__(self):
                super().__init__(f0=0,f1=100)
                self.actors.append(Text(ts=[0,0,100,100],x=450,y=450,size=45,s='$f(x)=x^5$',color='r',usetex=True))
                self.actors.append(Text(ts=[0,0,100,100],x=450,y=600,size=45,s='$g(x)=e^x$',color='b',usetex=True))
                self.actors=self.actors+LabeledAxis(ts=[0,0,100,100],
                                          px0=px0,dx0=dx0,px1=px1,dx1=dx1,dux=1,
                                          py0=py0,dy0=dy0,py1=py1,dy1=dy1,duy=100000)
                self.actors.append(Function(ts=[0,0,100,100],px0=px0,px1=px1,dx0=dx0,dx1=dx1,py0=py0,py1=py1,dy0=dy0,dy1=dy1,f=lambda phase,tt:lambda x: x**5,color='r'))
                self.actors.append(Function(ts=[0,0,100,100],px0=px0,px1=px1,dx0=dx0,dx1=dx1,py0=py0,py1=py1,dy0=dy0,dy1=dy1,f=lambda phase,tt:lambda x: g(x),color='b'))
        HorizScale().perform()
    dx1=dx1b
    if False:
        class FadeCurves(Stage):
            def __init__(self):
                super().__init__(f0=0,f1=30)
                self.actors.append(Text(ts=[0,0,0,30],x=450,y=450,size=45,s='$f(x)=x^5$',color='r',usetex=True))
                self.actors.append(Text(ts=[0,0,0,30],x=450,y=600,size=45,s='$g(x)=e^x$',color='b',usetex=True))
                self.actors=self.actors+LabeledAxis(ts=[0,0,0,30],
                                          px0=px0,dx0=dx0,px1=px1,dx1=dx1,dux=1,
                                          py0=py0,dy0=dy0,py1=py1,dy1=dy1,duy=100000)
                self.actors.append(Function(ts=[0,0,0,30],px0=px0,px1=px1,dx0=dx0,dx1=dx1,py0=py0,py1=py1,dy0=dy0,dy1=dy1,f=lambda phase,tt:lambda x: x**5,color='r'))
                self.actors.append(Function(ts=[0,0,0,30],px0=px0,px1=px1,dx0=dx0,dx1=dx1,py0=py0,py1=py1,dy0=dy0,dy1=dy1,f=lambda phase,tt:lambda x: g(x),color='b'))
        FadeCurves().perform()
    if True:
        class DrawTable(Stage):
            def __init__(self):
                super().__init__(f0=0,f1=100)
                fp=matplotlib.font_manager.FontProperties(fname=pathlib.Path('fonts/texgyreadventor-regular.otf'),size=15)
                self.actors.append(TableGrid  (ts=[0,30,100,130],x0=50*1.5,x1=570*1.5,yt=80*1.5,y0=102*1.5,yb=self.h-60*1.5,xs=[105*1.5,205*1.5,415*1.5],color='k'))
                self.actors.append(TableColumn(ts=[0,30,100,130],header="Time"     ,data=np.arange(37),x=100*1.5,y0=100*1.5,dy=15*1.5,horizontalalignment='right',color='k',font=fp))
                self.actors.append(TableColumn(ts=[15,45,100,130],header="$f(x)=x^5$" ,data=np.arange(37)**5,x=200*1.5,y0=100*1.5,dy=15*1.5,horizontalalignment='right',color='r',font=fp))
                self.actors.append(TableColumn(ts=[30,60,100,130],header="$g(x)=e^x$",data=np.round(10*np.exp(np.arange(37)))/10,x=410*1.5,y0=100*1.5,dy=15*1.5,horizontalalignment='right',color='b',font=fp))
                self.actors.append(TableColumn(ts=[45,75,100,130],header="$f(x)/g(x)=x^5/e^x$",data=np.floor(1000*(np.arange(37)**5/np.exp(np.arange(37))))/1000,x=560*1.5,y0=100*1.5,dy=15*1.5,horizontalalignment='right',color='#8000ff',font=fp))
            def setup(self,pb:PictureBox):
                pb.translate(400,0)
            def teardown(self,pb:PictureBox):
                pb.resetM()
        DrawTable().perform()


if __name__ == "__main__":
    main()