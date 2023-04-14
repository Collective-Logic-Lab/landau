# toolbox.py
#
# Bryan Daniels
# 2023/4/13
#
# Tools from my personal `toolbox` package that are useful here.
#

import pylab
import scipy
import pickle
import io
import sys
import copy

# defaultFigure

def setDefaultParams(usetex=False):
    # 4.23.2012 for PNAS (sizeMultiple=2)
    params = {'axes.labelsize': 16,
        'font.size': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.titlesize': 14,
        'text.usetex': usetex,
    }
    pylab.rcParams.update(params)

def makePretty(leg=None,ax=None,cbar=None,cbarNbins=6,frameLW=0.5):
    if ax is None: ax = pylab.gca()
    # set frame linewidth
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(frameLW)
    # set tick length
    ax.tick_params('both',width=frameLW,length=2)
    if leg is not None:
        # set legend frame linewidth
        leg.get_frame().set_linewidth(frameLW)
    if cbar is not None:
        # same for colorbar
        ax2 = cbar.ax
        tick_locator = ticker.MaxNLocator(nbins=cbarNbins)
        cbar.locator = tick_locator
        cbar.update_ticks()
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(frameLW)
        cbar.outline.set_linewidth(frameLW)
        ax2.tick_params(which='both',width=frameLW,length=2)

# 4.8.2011
def aboveDiagFlat(mat,keepDiag=False,offDiagMult=None):
    """
    Return a flattened list of all elements of the
    matrix above the diagonal.
    
    Use offDiagMult = 2 for symmetric J matrix.
    """
    m = copy.copy(mat)
    if offDiagMult is not None:
        m *= offDiagMult*(1.-scipy.tri(len(m)))+scipy.diag(scipy.ones(len(m)))
    if keepDiag: begin=0
    else: begin=1
    return scipy.concatenate([ scipy.diagonal(m,i)                          \
                              for i in range(begin,len(m)) ])


# simplePickle

def save(obj,filename):
    fout = io.open(filename,'wb')
    # Note: we currently save using backward-compatible protocol 2
    pickle.dump(obj,fout,2)
    fout.close()

def load(filename):
    fin = io.open(filename,'rb')
    try:
        obj = pickle.load(fin)
    except UnicodeDecodeError:
        # try using backward-compatible encoding in case
        # the file was saved using python 2
        fin.close()
        fin = io.open(filename,'rb')
        obj = pickle.load(fin,encoding='bytes')
    fin.close()
    return obj

