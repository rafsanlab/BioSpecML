import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import seaborn as sns
import os
from ..processings.cheimg_projections import projection_area, projection_std


def plt_annotate_dict(ax, dict:dict, idx:list, params:list=None):
    """
    Take a dict { n|tuple : 'strings..'} and draw a line in n-position of plt's ax.
    n can also be a tuple to indicate range.
    
    Arguments:
    - ax = plt's ax
    - idx = dataframes's index that will mapped to dict key
    - params = [filter:bool, y:int, rot:int, va:str, ha:str, colo:str, alpha:float]
        - filter : set True if only include keys within the data
        - y : position of y of the annotation
        - rot : rotation of the annotation
        - va : vertical alignment of the annotation
        - ha : horizontal alignment of the annotation
        - color : set colour for both line and anootation
        - alpha : set alpha for both both line and anootation
        
    """
    # Set default values if params is None
    params = params or [True, 0, -45, 1, 10, 'top', 'left', 'black', 0.5]

    # Unpack params
    filter, y, rot, linewidth, fontsize, va, ha, color, alpha = params

    # Define common parameters for plotting
    common_params = {'color': color, 'alpha': alpha}

    for key, val in dict.items():
        if isinstance(key, tuple):  # Check if key is a tuple
            for num in key:
                if (not filter or num in idx) or (filter and min(idx) <= num <= max(idx)):
                    ax.axvline(x=num, linestyle=':', linewidth=linewidth, **common_params)
                    plt.text(num, y, val, rotation=rot, va=va, ha=ha, fontsize=fontsize, **common_params)
        else:
            if (not filter or key in idx) or (filter and min(idx) <= key <= max(idx)):
                ax.axvline(x=key, linestyle='-', linewidth=linewidth, **common_params)
                plt.text(key, y, val, rotation=rot, va=va, ha=ha, fontsize=fontsize, **common_params)


def plot_df(df, check_data:bool=True, plot_mode:str='line', drop_cols:list=None, groupby=None,
            drop_rows:list=None, plot_cols:list=None, plt_args:dict=None, set_grid:bool|dict=True,
            shade_df=None, shade_alpha:float=0.1,
            linewidth:float=1.5, width:float=0.7, figsize:tuple=(7, 4), ylim:tuple=None,
            cmap:str=None, color:str|list=None, hide_spines:list=['top', 'right'],
            stacked:bool=False, fname:str=None,
            legend_off:bool=False, legend_outside:bool=False, legend_loc:str='best', legend_col:int=1, legend_fontsize:str|int='small',
            spines_width:float=1.5,  x_axis:str='', xlabel:str|dict=None, ylabel:str|dict=None, title=None,
            show_plot=True, annotation_dict:dict=None, annotation_args:list=None,
            yscale:float=None, xtick_rotate:float=None, line_styles:list=None,
            save_dpi:int=300, show_dpi:int=80,
            xticks:list=None, xticks_range:list=None, yticks:list=None, yticks_range:list=None,
            # process:list=None, label_name:str=None,ylist:list=None,
            # xticks_range:range=None, xlist:list=None,
            ):

    """
    This is the swiss knife function to plot df's data!
    Expects index to be x-axis, and columns as features.

    Arguments:
    - plot_mode (str) : pd.DataFrame.plot king argument
        - i.e: 'line', 'bar', 'barh', 'hist', 'box', 'kde', 'density', 'area', 'pie', 'scatter', 'hexbin'
    - x_axis (str) : name of the column to be x-axis
    - ylim (tup) = min and max of y-axis, i.e; (0,1)
    - shade_df (pd.DataFrame) : data in df to be plotted as shaded region, must have the same column name with df
    - set_grid = {'color'='black', 'linewidth'=0.2}
    - xlabel, ylabel (str|dict) : labels for axes,
        - can also pass plt's **args via dict
            - i.e: xlabel = {'xlabel': 'Variables', 'fontsize': 10, 'fontweight': 'bold'},
    - color (str|list) = accept a single colour: 'red', or a list of colours: ['red', 'blue']
    - annotation_dict (dict) : annotate x-axis with lines (works when plot_mode='line')
    - annotation_args (list) = [filter, y, rot, linewidth, fontsize, va, ha, color, alpha]
        - filter (bool) : set True if only include wavenumbers in the plot
        - y (int) : position of y of the annotation
        - rot (+/-int) : rotation of the annotation
        - linewidth (float) : width of the line
        - fontsize (int) : dict's value text font size
        - va (str) : vertical alignment of the annotation
        - ha (str) : horizontal alignment of the annotation
        - color (str) : set colour for both line and anootation
        - alpha (float) : set alpha for both both line and anootation

    """

    # ----- checking ------

    # create plt_args
    if plt_args == None:
        plt_args = {}

    # check total rows to plot
    if check_data:
        if df.shape[0] > 50000 or df.shape[1] > 50000:
            raise Exception('Too many rows or columns (>50,000) to plot, set *check_data = False to skip this warning.')
    
    # condition for if plot_cols is defined so that we can drop other columns
    if plot_cols != None:
        columns = df.columns.tolist()
        columns = [c for c in columns if c not in plot_cols and c not in x_axis]        
        if drop_cols is not None:
            drop_cols.extend(columns)
        else:
            drop_cols = columns.copy()

    # condition to drop columns
    if drop_cols != None:
        for col in drop_cols:
            if col in df.columns.tolist():
                df = df.drop([col], axis=1)

    # condition to drop rows
    if drop_rows != None:
        for row in drop_rows:
            if row in df.index.tolist():
                df = df.drop([row], axis=0)
    

    # ----- plotting ------

    # set plotting colors
    if color==None and cmap==None:
        plt_args.update({'cmap': 'Spectral'}) # default color using cmap
        # plt_args = {'cmap':'Spectral'} # default color using cmap
    elif color!=None:
        if isinstance(color, str): # set colour directly using color
            plt_args.update({'color':color})
            # plt_args = {'color':color}
        elif isinstance(color, list): # set colour using list of color 
            # custom cmap based on number of df columns
            n = len(df.columns)
            custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', color, N=256).resampled(n)
            plt_args.update({'cmap':custom_cmap})
            # plt_args = {'cmap':custom_cmap}
    elif cmap!=None:
        plt_args.update({'cmap':cmap}) # set colour using cmap
        # plt_args = {'cmap':cmap} # set colour using cmap
    
    # make subgroups for plotting
    if groupby!=None:
        df = df.groupby(groupby)

    # plot line or others
    if plot_mode == 'line':
        if x_axis != '':
            ax = df.plot(kind=plot_mode, stacked=stacked, figsize=figsize, linewidth=linewidth, **plt_args, x=x_axis)
        else:
            ax = df.plot(kind=plot_mode, stacked=stacked, figsize=figsize, linewidth=linewidth, **plt_args)
    else:
        if x_axis != '':
            ax = df.plot(kind=plot_mode, stacked=stacked, figsize=figsize, **plt_args, x=x_axis)
        else:
            ax = df.plot(kind=plot_mode, stacked=stacked, figsize=figsize, **plt_args)

    # ---- apply shade region -----
    
    if shade_df is not None:
        colors = [line.get_color() for line in ax.get_lines()]
        for i, col in enumerate(df.columns):
            ax.fill_between(df.index, df[col]-shade_df[col], df[col]+shade_df[col], alpha=shade_alpha, color=colors[i])

    # ---- annotate plot -----

    if annotation_dict != None:
        idx_df = [int(float(i)) for i in df.index.tolist()]
        plt_annotate_dict(ax, annotation_dict, idx_df, annotation_args)

    # ----- styling each lines  ------

    if line_styles != None:

        # iterate line in line styles
        for i, line in enumerate(ax.get_lines()):
            line.set_linestyle(line_styles[i])

        # Update legend with custom line styles
        legend_handles = []
        for i, (colname, line) in enumerate(zip(df.columns, ax.get_lines())):
            line.set_linestyle(line_styles[i])
            legend_handles.append(mlines.Line2D([], [], color=line.get_color(), linestyle=line_styles[i], label=colname))

    # ----- other plt args ------

    plt.tight_layout()
    plt.gcf().set_dpi(show_dpi) 

    # legend arguments
    if legend_off:
        ax.legend().remove() 
    else:
        if legend_outside!=False:
            ax.legend(loc=legend_loc, ncol=legend_col, fontsize=legend_fontsize,  bbox_to_anchor=(1, 1))
        else:
            ax.legend(loc=legend_loc, ncol=legend_col, fontsize=legend_fontsize)

    # grids arguments
    if set_grid is not False:
        if isinstance(set_grid, dict):
            ax.grid(True, **set_grid)
        else:
            ax.grid(True, color='black', linewidth=0.2)  
    
    # setting spines
    for spine in hide_spines:
        plt.gca().spines[spine].set_visible(False)
    
    # setting spine width
    for spine in ax.spines.values():
        spine.set_linewidth(spines_width)

    # option to manually set xticks
    if xticks is not None:
        if xticks_range is None:
            xticks_range = range(min(xticks), max(xticks)+1)
        ax.set_xticks(xticks_range)
        ax.set_xticklabels(xticks)

    # option to manually set yticks
    if yticks is not None:
        if yticks_range is None:
            yticks_range = range(min(yticks), max(yticks)+1)
        ax.set_yticks(yticks_range)
        ax.set_ytickslabels(yticks)

    if ylim != None:
        plt.ylim(ylim[0], ylim[1]) 

    if yscale != None:
        plt.yscale(yscale)

    if xtick_rotate != None:
        plt.xticks(rotation=xtick_rotate)

    if xlabel != None:
        if isinstance(xlabel, str):
            ax.set_xlabel(xlabel, fontweight='bold')
        elif isinstance(xlabel, dict):
            ax.set_xlabel(**xlabel)

    if ylabel != None:
        if isinstance(ylabel, str):
            ax.set_ylabel(ylabel, fontweight='bold')
        elif isinstance(ylabel, dict):
            ax.set_ylabel(**ylabel)

    if title != None:
        if isinstance(title, str):
            plt.title(title, fontweight='extra bold')
        elif isinstance(title, dict):
            plt.title(**title)

    if fname != None:
        plt.savefig(fname, dpi=save_dpi, bbox_inches='tight')
    
    if show_plot != False:
        plt.show()

    plt.close()


def plot_images_from_folder(
    folder_path, rows:int=1, cols:int=1, img_format:str='.png', figsize:tuple=(10, 10),
    title_size:int=10, cmap:str='binary', cmap_reverse:bool=False, fname:str=None,
    save_dpi:int=200, show_plot:bool=True
    ):
    """
    Plot images from a folder in a single figure.

    Args:
    - folder_path (str): Path to the folder containing images.
    - rows (int): Number of rows in the subplot grid.
    - cols (int): Number of columns in the subplot grid.
    - figsize (tuple): Figure size (width, height) in inches.
    """

    # ----- get the images -----

    # image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files = [f for f in os.listdir(folder_path) if f.endswith((img_format))]
    image_files.sort()

    # ----- plot each images -----

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(image_files):
            img_path = os.path.join(folder_path, image_files[i])
            img = mpimg.imread(img_path)
            if cmap_reverse==False:
                ax.imshow(img, cmap=plt.colormaps.get_cmap(cmap))
            elif cmap_reverse==True:
                ax.imshow(img, cmap=plt.colormaps.get_cmap(cmap).reversed())
            ax.set_title(image_files[i], fontsize=title_size)
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()

    # ----- other params -----

    if show_plot==True:
        plt.show()

    if fname!=None:
        plt.savefig(fname, dpi=save_dpi, bbox_inches='tight')
    
    plt.close(fig)
    plt.clf() 


def rows_to_img(rows, label_col:str, w=28, h=28, cmap='viridis', drop_col=None, fname=None):
    """
    Plot images from multiple rows in a DataFrame.

    Parameters:
        rows (pandas.DataFrame): DataFrame containing rows to plot.
        label_col (str): The name of the column containing labels.
        w (int): Width of the image.
        h (int): Height of the image.
        cmap (str): Colormap to use for displaying the image.
        drop_col (str or list): Column(s) to drop from the row before plotting.

    Returns:
        None
    """
    num_rows = len(rows)
    plt.figure(figsize=(15, 5 * num_rows))  # Adjust the figure size as needed

    for i, (_, row) in enumerate(rows.iterrows(), 1):
        # Make a copy of the row to avoid modifying the original DataFrame
        row_copy = row.copy()

        # Extract label
        label = row_copy[label_col]

        # Drop label column and specified additional columns
        if drop_col is not None:
            if isinstance(drop_col, str):
                drop_col = [drop_col]
            row_copy = row_copy.drop([label_col] + drop_col)
        else:
            row_copy = row_copy.drop([label_col])

        # Convert row values to numpy array and reshape to image dimensions
        img = row_copy.to_numpy().reshape(w, h)

        # Plot the image
        plt.subplot(1, num_rows, i)
        plt.imshow(img, cmap=cmap)
        plt.title(f'Label: {label}')
        plt.axis('off')  # Turn off axis

    if fname!=None:
        plt.savefig(fname, bbox_inches='tight')
    plt.tight_layout()  # Adjust subplots to avoid overlap
    plt.show()



def plot_chemimg(p, wn, method='area', cmap='viridis', title=None, fname=None, show_plot=True, title_size=10, show_axes=True):
    """
    Plot a chemical image.
    
    Parameters:
    - p (array-like): The data to plot.
    - wn (array-like): Wavenumbers or other axis labels.
    - cmap (str): Colormap for the image.
    - title (str): Title of the plot.
    - fname (str): Filename to save the plot.
    - show_plot (bool): Whether to show the plot.
    - title_size (int): Font size of the title.
    - show_axes (bool): Whether to show axes.
    
    Returns:
    - None
    """
    if method == 'area':
        chemimg = projection_area(p, wn)
    elif method == 'std':
        chemimg = projection_std(p, wn)
    else:
        raise ValueError('<method> should be either "area" or "std".')
    plt.imshow(chemimg, cmap=cmap)
    if not show_axes:
        plt.axis('off')
    if title is not None:
        if isinstance(title, str):
            plt.title(title, fontweight='extra bold', fontsize=title_size)
        elif isinstance(title, dict):
            title.update({'fontsize':title_size})
            plt.title(**title)
    plt.tight_layout()
    if fname is not None:
        dir_ = os.path.dirname(fname)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        plt.savefig(fname, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_spectra(sp, wn, skip_zeros=False, convert_wn_to_string=False,
                 fname=None, show_plot=True, legend_off=False, legend_outside=False,
                 plt_type='mean', random_spectra=10, index_list=None, include_mean:bool=True,
                 title={'label':None},
                 xlabel:dict={'xlabel':'Wavenumber ($cm^{-1}$)', 'fontsize':10}, 
                 ylabel:dict={'ylabel':'Absorbance (a.u)', 'fontsize':10}, 
                 cmap='Spectral',
                 plot_df_args=None):
    """
    Plot the spectra data.

    Parameters:
    - sp (array-like): The spectra data.
    - wn (array-like): The wavenumbers.
    - skip_zeros (bool): Whether to skip rows with zero sums.
    - convert_wn_to_string (bool): Whether to convert wavenumbers to string.
    - plt_type (str): Type of plot ('mean', 'random', 'index').
    - random_spectra (int): Number of random spectra to plot if plt_type is 'random'.
    - index_list (list): List of indices to plot if plt_type is 'index'.
    - title (str): Title of the plot.
    - fname (str): Filename to save the plot.
    - show_plot (bool): Whether to show the plot.
    - plot_df_args: (dict) Additional plotting arguments.

    Returns:
    - None
    """
    if convert_wn_to_string:
        wn = [str(w) for w in wn]
    
    df = pd.DataFrame(sp.T, columns=wn)

    if skip_zeros:
        df = df[df.sum(axis=1) != 0]

    plot_df_args_temp = {
        'show_plot': show_plot,
        'legend_off':legend_off,
        'legend_outside':legend_outside,
        'spines_width':1,
        'cmap':cmap,
        }

    if plot_df_args is None:
        plot_df_args = plot_df_args_temp
    else:
        plot_df_args.update(plot_df_args_temp)

    df_mean = df.mean().to_frame().rename(columns={0: 'Mean spectra'})

    if plt_type == 'mean':
        df = df_mean
    
    elif plt_type == 'random':
        if random_spectra > 0 and random_spectra <= df.shape[0]:
            df = df.sample(random_spectra).T
        else:
            raise ValueError('<random_spectra> must be greater than 0 and less than or equal to the number of available spectra.')
    
    elif plt_type == 'index':
        if index_list is not None:
            index_list = [idx for idx in index_list if idx in df.index]
            if not index_list:
                raise ValueError('None of the provided indices are valid.')
            df = df.loc[index_list].T
        else:
            raise ValueError('index_list cannot be None when <plt_type> is "index".')
    else:
        raise ValueError('Choose <plt_type> between "mean", "random", or "index".')
        
    if plt_type != 'mean' and include_mean == True:
        df = pd.concat([df,df_mean], axis=1)
    
    if fname is not None:
        dir_ = os.path.dirname(fname)
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    plot_df(df, check_data=False,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            fname=fname,
            **plot_df_args)


def plot_chemimg_spectra(p, sp, wn, fname, cmap, plt_type, random_spectra, index_list,
                         legend_outside, saveplotdir, show_plot, save_plot,
                         skip_zeros=False, plot_chemimg_status=True, plot_spectra_status=True, prefix='',
                         plot_df_args=None,
                         **args):
    if plot_chemimg_status:
        plot_chemimg(p, wn, show_plot=show_plot, cmap=cmap,
                    show_axes=False,
                    title={'label':f'{prefix}:{fname}'},
                    fname=os.path.join(saveplotdir, f'{fname}/{prefix}_auc.png') if save_plot else None,
                    )
    if plot_spectra_status:
        plot_spectra(sp, wn, show_plot=show_plot, cmap=cmap,
                    skip_zeros=skip_zeros, index_list=index_list, plt_type=plt_type,
                    random_spectra=random_spectra, title={'label':f'{prefix}:{fname}'},
                    fname=os.path.join(saveplotdir, f'{fname}/{prefix}_spectra_random-mean.png') if save_plot else None,
                    legend_off=False, legend_outside=legend_outside, plot_df_args=plot_df_args)


def plot_3dwaterfall(df:pd.DataFrame, convert_int:bool=True, 
                     figsize:tuple=(9,8), cmap:str='Reds',
                     invert_cmap:bool=False, box_aspect:list=[1,2,1],
                     elev:int=25, azim:int=0, labelsdict:tuple=(None,None,None),
                     title:str=None, legend_args:dict={'bbox_to_anchor':(1.1, 0.75), 'loc':'best'},
                     fname=None, show_plot:bool=True):

    """ - column and index must be a number or numerical values.
        - column will be the x-axis (depth)
        - index will be the y-axis (length)
        - values wil be the z-axis (height)
        Args:
            - box_aspect(list): aspect ratio for x, y, z
            - elev(int) : elevation view of the 3d plot
            - azim(int) : azimuth view of the 3d plot
            - labelsdict(tuple) : labels for each x, y, z
    """
    num_samples = df.shape[1] # columns
    if convert_int:
        df.columns = [int(i) for i in df.columns.to_list()]
        df.index = [int(i) for i in df.index.to_list()]
    # X is the columns, Y is the index
    X, Y = np.meshgrid(df.columns.to_list(), df.index.to_list())
    Z = df.values

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    palette = sns.color_palette(cmap, num_samples)[::-1]
    if invert_cmap:
        palette = palette[::-1]

    for i in range(df.shape[1]):
        ax.plot(X[:, i],Y[:, i],Z[:, i], label=df.columns[i], color=palette[i])

    ax.set_box_aspect(box_aspect)
    ax.view_init(elev=elev, azim=azim)
    xlabel, ylabel, zlabel = labelsdict
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.legend(**legend_args)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')
    if show_plot:
        plt.show()