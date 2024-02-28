import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd


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


def plot_df(df, check_data:bool=True, plot_mode:str='line', drop_cols:list=None,
            drop_rows:list=None, plot_cols:list=None, set_grid:bool|dict=False,
            width:float=0.7, figsize:tuple=(7, 4), ylim:tuple=None,
            cmap:str=None, color:str=None, hide_spines:list=['top', 'right'],
            stacked:bool=False, fname:str=None, legend_outside:bool=False,
            spines_width:float=1.5,  x_axis=None, xlabel=None, ylabel=None, title=None,
            show_plot=True, annotation_dict:dict=None, annotation_args:list=None,
            yscale:float=None, xtick_rotate:float=None, line_styles:list=None,
            # process:list=None, label_name:str=None,ylist:list=None,
            # xticks_range:range=None, xticks:list=None, xlist:list=None,
            ):

    """
    This is the swiss knife function to plot df's data!
    Expects index to be x-axis, and columns as features.

    Arguments:
    - x_axis (str) : name of the column to be x-axis
    - ylim (tup) = min and max of y-axis, i.e; (0,1)
    - set_grid = {'color'='black', 'linewidth'=0.2}
    - xlabel, ylabel (str|dict) : labels for axes,
        - can also pass plt's **args via dict
            - i.e: xlabel = {'xlabel': 'Variables', 'fontsize': 10, 'fontweight': 'bold'},
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
        plt_args = {'cmap':'Spectral'}
    elif color!=None:
        plt_args = {'color':color}
    elif cmap!=None:
        plt_args = {'cmap':cmap}

    # plot line or others
    if plot_mode == 'line':
        if x_axis is not None:
            ax = df.plot(kind=plot_mode, stacked=stacked, figsize=figsize, **plt_args, x=x_axis)
        else:
            ax = df.plot(kind=plot_mode, stacked=stacked, figsize=figsize, **plt_args)
    else:
        if x_axis is not None:
            ax = df.plot(kind=plot_mode, stacked=stacked, width=width, figsize=figsize, **plt_args, x=x_axis)
        else:
            ax = df.plot(kind=plot_mode, stacked=stacked, width=width, figsize=figsize, **plt_args)

    # option to plot bar
    # if plot_mode == 'bar':
    #     if x_axis is not None:
    #         ax = df.plot(kind='bar', stacked=stacked, width=width, figsize=figsize, cmap=cmap, x=x_axis)
    #     else:
    #         ax = df.plot(kind='bar', stacked=stacked, width=width, figsize=figsize, cmap=cmap)

    # # option to plot horizontal bar
    # elif plot_mode == 'barh':
    #     if x_axis is not None:
    #         ax = df.plot(kind='barh', stacked=stacked, width=width, figsize=figsize, cmap=cmap, x=x_axis)
    #     else:
    #         ax = df.plot(kind='barh', stacked=stacked, width=width, figsize=figsize, cmap=cmap)

    #     # ax = df.plot.barh(stacked=stacked, width=width, figsize=figsize, cmap=cmap)

    # # option to plot line
    # elif plot_mode == 'line':
    #     if x_axis is not None:
    #         ax = df.plot(kind='line', stacked=stacked, figsize=figsize, cmap=cmap, x=x_axis)
    #     else:
    #         ax = df.plot(kind='line', stacked=stacked, figsize=figsize, cmap=cmap)#, y=df.columns.tolist(), x=df.index)

    # # pass pd.DataFrame.plot kind
    # else:
    #     ax = df.plot(kind=plot_mode, stacked=stacked, figsize=figsize, cmap=cmap)

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

    if legend_outside!=False:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    if set_grid is not None:
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

    # if xticks_range != None and xticks != None:
        # ax.set_xticks(xticks_range)
        # ax.set_xticklabels(xticks)
    # if xticks != None:
        # ax.set_xticks(df.index.tolist())
        # ax.set_xticklabels(df.index.tolist())
        # ax.set_xticks(range(len(xlist)))

    # if ylist != None:
    #     ax.set_yticks(range(len(ylist)))
    #     ax.set_yticklabels(ylist)

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
    
    plt.tight_layout()

    if fname != None:
        plt.savefig(fname)

    if show_plot != False:
        plt.show()

    plt.close()