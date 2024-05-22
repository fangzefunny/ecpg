import numpy as np 
import seaborn as sns 
import matplotlib.colors
import matplotlib as mpl 
import matplotlib.pyplot as plt 

class viz:
    '''Define the default visualize configure
    '''
    # basic
    white   = np.array([255, 255, 255]) / 255
    new_blue= np.array([ 98, 138, 174]) / 255
    new_red = np.array([195, 102, 101]) / 255
    dBlue   = np.array([ 56,  56, 107]) / 255
    Blue    = np.array([ 46, 107, 149]) / 255
    lBlue   = np.array([241, 247, 248]) / 255
    lBlue2  = np.array([166, 201, 222]) / 255
    Green   = np.array([  8, 154, 133]) / 255
    lGreen  = np.array([242, 251, 238]) / 255
    dRed    = np.array([108,  14,  17]) / 255
    Red     = np.array([199, 111, 132]) / 255
    lRed    = np.array([253, 237, 237]) / 255
    lRed2   = np.array([254, 177, 175]) / 255
    dYellow = np.array([129, 119,  14]) / 255
    Yellow  = np.array([220, 175, 106]) / 255
    lYellow2= np.array([166, 201, 222]) / 255
    lYellow = np.array([252, 246, 238]) / 255
    Purple  = np.array([108,  92, 231]) / 255
    ocGreen = np.array([ 90, 196, 164]) / 255
    oGrey   = np.array([176, 166, 183]) / 255
    Palette = [Blue, Yellow, Red, ocGreen, Purple]

    # palette for agents
    b1      = np.array([ 43, 126, 164]) / 255
    r1      = np.array([249, 199,  79]) / 255
    r2      = np.array([228, 149,  92]) / 255
    r3      = np.array([206,  98, 105]) / 255
    m2      = np.array([188, 162, 149]) / 255
    g       = np.array([.7, .7, .7])
    Pal_agent = [b1, g, Red, r2, m2] 

    # palette for block types
    dGreen  = np.array([ 15,  93,  81]) / 255
    fsGreen = np.array([ 79, 157, 105]) / 255
    Ercu    = np.array([190, 176, 137]) / 255
    Pal_type = [dGreen, fsGreen, Ercu]

    # palette for block types 2
    
    gg1     = np.array([112, 169, 161]) / 255
    gg2     = np.array([158, 193, 163]) / 255
    gg3     = np.array([207, 224, 195]) / 255
    Pal_type2 = [gg1, gg2, gg3]

    # Morandi
    m0      = np.array([ 16, 186, 184]) / 255
    m1      = np.array([236, 123, 119]) / 255
    m2      = np.array([188, 162, 149]) / 255
    Pal_fea = [m2, m1, m0]

    # lambda gradient
    lmbda0  = np.array([249, 219, 189]) / 255
    lmbda1  = np.array([255, 165, 171]) / 255
    lmbda2  = np.array([218,  98, 125]) / 255
    lmbda3  = np.array([165,  56,  96]) / 255
    lmbda4  = np.array([ 69,   9,  32]) / 255
    lmbda_gradient = [lmbda0, lmbda1, lmbda2, lmbda3, lmbda4]


    # for insights
    BluesMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizBlues',   [lBlue, Blue])
    RedsMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizReds',    [lRed, dRed])
    YellowsMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizYellows', [lYellow, Yellow])
    GreensMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizGreens',  [lGreen, Green])
    cool_warm = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'cool_warm',   [new_blue, white, new_red])

    @staticmethod
    def get_style(): 
        '''The style of our figures'''
        sns.set_context("talk")
        mpl.rcParams['pdf.fonttype']       = 42
        mpl.rcParams['axes.spines.right']  = False
        mpl.rcParams['axes.spines.top']    = False
        mpl.rcParams['savefig.format']     = 'pdf'
        mpl.rcParams['savefig.dpi']        = 300
        mpl.rcParams['figure.facecolor']   = 'w'

    @staticmethod
    def violin(ax, data, x, y, order, palette, orient='v',
        hue=None, hue_order=None,
        scatter_size=7, scatter_alpha=1,
        mean_marker_size=6, err_capsize=.11, 
        add_errs=True, errorbar='se', errorcolor=[.3]*3,
        errorlw=2):
        g_var = y if orient=='h' else x
        v_var = x if orient=='h' else y
        v=sns.violinplot(data=data, 
                            x=x, y=y, order=order, 
                            hue=g_var if hue is None else hue, 
                            hue_order=order if hue is None else hue_order,
                            orient=orient, palette=palette, 
                            legend=False, alpha=.1, inner=None, density_norm='width',
                            ax=ax)
        plt.setp(v.collections, alpha=.5, edgecolor='none')
        sns.stripplot(data=data, 
                            x=x, y=y, order=order, 
                            hue=g_var if hue is None else hue, 
                            hue_order=order if hue is None else hue_order, 
                            orient=orient, palette=palette, 
                            size=scatter_size,
                            edgecolor='auto', jitter=True, alpha=scatter_alpha,
                            dodge=False if hue is None else True,
                            legend=False, zorder=2,
                            ax=ax)
        if add_errs:
            groupby = [g_var, hue] if hue is not None else [g_var]
            sns.barplot(data=data, 
                        x=x, y=y, order=order, 
                        orient=orient, 
                        hue=g_var if hue is None else hue, 
                        hue_order=order if hue is None else hue_order,
                        errorbar=errorbar, linewidth=1, legend=False,
                        edgecolor=(0,0,0,0), facecolor=(0,0,0,0),
                        capsize=err_capsize, err_kws={'color': errorcolor, 'linewidth': errorlw},
                        ax=ax)
            sns.stripplot(data=data.groupby(by=groupby)[v_var].mean().reset_index(), 
                            x=x, y=y, order=order, 
                            hue=hue, hue_order=hue_order, 
                            palette=[errorcolor]*len(hue_order) if hue is not None else None,
                            dodge=False if hue is None else True,
                            legend=False,
                            marker='o', size=mean_marker_size, color=errorcolor, ax=ax)
    
    @staticmethod
    def violin_with_tar(ax, data, color, x, y, order, orient='v',
        hue=None, hue_order=None,
        scatter_size=4, scatter_alpha=1,
        mean_marker_size=6, err_capsize=.14, 
        errorbar='se', errorlw=3,
        errorcolor=[.5]*3):
        g_var = y if orient=='h' else x
        v_var = x if orient=='h' else y
        palette = [[.9]*3, color]
        sns.violinplot(data=data, 
                x=x, y=y, order=order, 
                hue=g_var if hue is None else hue, 
                hue_order=order if hue is None else hue_order,
                orient=orient, palette=palette, 
                legend=False, alpha=.5, 
                inner=None, density_norm='width',
                edgecolor='none', gap=.15,
                split=True,
                ax=ax)
        sns.stripplot(data=data.query(f'{hue}=="{hue_order[1]}"'), 
                x=x, y=y, order=order, 
                hue=g_var if hue is None else hue, 
                hue_order=order if hue is None else hue_order, 
                orient=orient, palette=palette, 
                size=scatter_size,
                edgecolor='auto', jitter=True, alpha=scatter_alpha,
                dodge=True,
                legend=False, zorder=2,
                ax=ax)
        point_palette = [color, errorcolor]
        sns.pointplot(data=data, 
                x=x, y=y, order=order, 
                orient=orient, 
                hue=hue, hue_order=hue_order,
                legend=False,
                palette=point_palette,
                ls='none', dodge=.4,
                errorbar=errorbar,
                markersize=mean_marker_size,
                capsize=err_capsize, err_kws={'linewidth': errorlw},
                ax=ax)
    