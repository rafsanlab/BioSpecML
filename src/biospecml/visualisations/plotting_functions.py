import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import os

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


def plot_df(df, check_data:bool=True, plot_mode:str='line', drop_cols:list=None, groupby:str|int|float=None,
            drop_rows:list=None, plot_cols:list=None, plt_args:dict={}, set_grid:bool|dict=True,
            linewidth:float=1.5, width:float=0.7, figsize:tuple=(7, 4), ylim:tuple=None,
            cmap:str=None, color:str|list=None, hide_spines:list=['top', 'right'],
            stacked:bool=False, fname:str=None,
            legend_off:bool=False, legend_outside:bool=False, legend_loc:str='best', legend_col:int=1,
            spines_width:float=1.5,  x_axis:str='', xlabel=None, ylabel=None, title=None,
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
            ax.legend(loc=legend_loc, bbox_to_anchor=(1, 1), ncol=legend_col)
        else:
            ax.legend(loc=legend_loc, ncol=legend_col)

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
        plt.title(title, fontweight='extra bold')

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
    