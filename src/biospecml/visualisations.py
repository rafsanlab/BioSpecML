import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os
import random

def plot_spectra_mean(
    dataframe, shade=True, random_row_num=None, save_dpi=200, plot_dpi=80,
    figsize=(7, 4), title='Mean spectra', fname=None
    ):
    """
    Plot average spectra with SD shade from a dataframe (please exclude metadata
    columns) with additional random spectra if needed.

    Args:
        dataframe: pandas df.
        shade(bool): condition of SD for the mean spectra.
        random_row_num(int): number of random spectra to include in the plot.
        save_dpi(int): dpi for saved figure.
        plot_dpi(int): dpi for plotted figue.
        figsize(tuple): figure size.
        title(str): title name.
        filename(str): filename or save path with filename.
    Return:
        Matplotlib figure with saved file.
    """

    plt.figure(dpi=plot_dpi, figsize=figsize)

    """ condition to additionally plot random rows """
    if random_row_num != None:
        total_rows = len(dataframe)
        random_indices = random.sample(range(total_rows), random_row_num) # for row indices
        random_rows = dataframe.iloc[random_indices]  # the corresponding random rows from the df
        for idx, (_, row) in enumerate(random_rows.iterrows()):
            # this condition allows only one label of the random spectra shown
            if idx == 0:
                plt.plot(row.index, row.values, color='grey', linewidth=0.5, label='Random spectra')
            else:
                plt.plot(row.index, row.values, color='grey', linewidth=0.5)

    """ calculate and plot mean and condition for std dev """
    average_row = dataframe.mean()
    plt.plot(average_row.index, average_row.values, color='red', linewidth=1.0, label='Mean')
    if shade != False: # Plot the standard deviation as shaded area
        std_row = dataframe.std()
        plt.fill_between(
            average_row.index.astype(float),
            average_row.values - std_row.values,
            average_row.values + std_row.values,
            color='red', alpha=0.3
            )

    """ the rest of the plotting """
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Absorbance')
    plt.title(title)
    plt.legend()
    if fname != None:
        plt.savefig(fname=fname, dpi=save_dpi, bbox_inches='tight')
    plt.show()

def plot_rxc(nrows=1, ncols=2, dpi=120, figsize=(9,3), imgs=[], titles=[],
             title_size=10, cmap='binary', show=True, fname=None):
    """
    Plot X number of rows and columns subplot.

    Args:
        nrows: number of rows (1)
        ncols: number of columns
        dpi: dpi value of the figure
        figsize: figure size
        imgs: list of image array to be plotted
        titles: list of titles in string
        title_size: title font size
        cmap: plt's cmap
        show: to show plot or not
        fname: str of image name with img format

    Returns:
        plt.show()
    """  
    _, axs = plt.subplots(nrows, ncols, dpi=dpi, figsize=figsize)
    for i in range(ncols): axs[i].imshow(imgs[i], cmap=cmap)  
    if titles != []:
        for i in range(ncols): axs[i].title.set_text(titles[i])
        for i in range(ncols): axs[i].title.set_size(title_size)
    for i in range(ncols): axs[i].set_axis_off()
    plt.tight_layout()
    if fname != None:
        plt.savefig(fname, transparent=True, dpi=dpi)
    if show==False:
        plt.ioff()
        plt.close()
    elif show==True:
        plt.show(block=False);
        plt.close()
      

def plot_linear(data_dirs=[], delimiter='\t', header=0, replace_epoch=False, epoch_col='',
                figsize=(6, 6), col_names=[], colors=['royalblue'], line_styles=['-'], line_thickness = 1.5,
                title='', xlabel='', ylabel='', show_plot=True, fname:str=None, save_dpi=300, return_data=False):
    """
    Plot a linear graph by the column names.

    Args:
        - data_dirs:list =  list to the file paths.
        - delimiter:str = delimiter of the files.
        - header:int = header of the files.
        - replace_epoch:bool = option to replace epoch based on index if using multiple files.
        - epoch_col:str = name of the epoch column use for plotting and replace_epoch.
        - figsize:tup = figure size.
        - col_names:list = list of the column names to be plot.
        - colors:list = list of colours of each plotted columns.
        - line_styles:list = list of the line styles of each plotted columns.
        - line_thickness:float = thickness of all lines.
        - title:str = title of the plot.
        - xlabel:str, ylabel:str = x and y axis label.
        - show_plot:bool = option to show plot.
        - fname:str = option to save the plot, use path and format i.e: '/folder/plot.png'.
        - save_dpi:int = dpi of the saved figure.
        - return_data:bool = option to return the plotted dataframe.
    Returns:
        - plotted image
        - plotted data if return_data=True
    """
    #--- read and proces the files ---
    data = pd.DataFrame()
    for data_dir in data_dirs:
        data_i = pd.read_csv(data_dir, delimiter=delimiter, header=header)
        data = pd.concat([data, data_i], ignore_index=True)
    
    #--- option to replace epoch ---
    if replace_epoch!=False:
        new_epochs = data.index + 1
        data[epoch_col] = new_epochs

    #--- setting figure and plotting ---
    plt.figure(figsize=figsize, facecolor='white')
    ax = plt.gca()
    for col, color, line_style in zip(col_names, colors, line_styles):
        ax = data.plot(x=epoch_col, y=col, kind='line', ax=ax, color=color, linestyle=line_style, 
                    linewidth=line_thickness, label=col)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    #--- labels and metadata stuffs ---
    ax.grid(True, color='black', linewidth=0.2)
    plt.title(title, fontweight='extra bold')
    plt.xlabel(xlabel=xlabel, fontsize='medium', fontweight='extra bold')
    plt.ylabel(ylabel=ylabel, fontsize='medium', fontweight='extra bold')
    legend = plt.legend(frameon=False, fontsize='medium')
    plt.tight_layout()

    #--- and the rest.. ---
    if fname!=None:
        plt.savefig(fname=fname, dpi=save_dpi)
    if show_plot:
        plt.show()
    plt.close()
    if return_data!=False:
        return data
    

def plot_images_from_folder(
    folder_path, rows=1, cols=1, img_format='.png', figsize=(10, 10),
    title_size=10, cmap='binary', cmap_reverse=False, fname=None,
    save_dpi=200, show_plot=True):
    """
    Plot images from a folder in a single figure.

    Args:
        folder_path (str): Path to the folder containing images.
        rows (int): Number of rows in the subplot grid.
        cols (int): Number of columns in the subplot grid.
        figsize (tuple): Figure size (width, height) in inches.
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    # === get the images ===
    # image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files = [f for f in os.listdir(folder_path) if f.endswith((img_format))]
    image_files.sort()

    # === plot each images ===
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

    # === show condition ===
    if show_plot==True:
        plt.show()
        # save condition
        if fname!=None:
            plt.savefig(fname, dpi=save_dpi)
        plt.close(fig)
        plt.clf() 
    elif show_plot==False:
        # save condition
        if fname!=None:
            plt.savefig(fname, dpi=save_dpi)
        plt.close(fig)
        plt.clf() 
