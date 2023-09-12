from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def cluster_k(p, k=2, init=1, max_iter=300, scale=False):
    """
    Cluster p to k value (default=2) Returns an image mask based on clustering.
    init can be 'auto', 10='random' or 1='k-means++'. p can be either 3D numpy
    ftir image (output from read_mat) or 2D dataframe matrix without metadata.

    Args:
        p(np.array or pd.DataFrame): 3D array or 2D dataframe.
        k(int): cluster value.
        max_iter(int): KMeans() paramter for iteration value.
        scale(bool): option for Scaler in KMeans().
    Return:
        model(?): KMeans model.
        mask(np.array): image consisting of clusters number for 3D array only.

    """
    if isinstance(p, pd.DataFrame) == True:
        pk = p
    else:
        pk = np.transpose(p, (1, 2, 0))
        pk = pk.reshape(pk.shape[0] * pk.shape[1], pk.shape[2])
        pk = pk.astype(float)
    if scale != False:
        scaler = StandardScaler()
        pk = scaler.fit_transform(pk)
    model = KMeans(n_clusters=k, n_init=init, max_iter=max_iter)
    model = model.fit(pk)
    if isinstance(p, pd.DataFrame) == True:
        mask = None
    else:
        mask = model.labels_.reshape(p.shape[1], p.shape[2])
    return model, mask


def cluster_dbscan(dfX, eps, min_samples):
    """
    DBSCAN clustering.

    Args:
        dfX(pd.DataFrame): dataframe data matrix.
        eps(int): epsilon value for DBSCAN().
        min_samples(int): minimum points for DBSCAN().
    Return:
        model(?): DBSCAN fit() model.
        model.labels(np.array): clusters data.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(dfX)
    return model, model.labels_


def opt_elbow_method(dfX, clusters, scale=True, plot=False, fname=None, return_results=False):
    """
    An elbow method to find optimum k-cluster number for K-means.

    Args:
        dfX(pd.DataFrame or np.array): *arg from cluster_k().
        clusters(int): maximum cluster to test.
        scale(bool): cluster_k() scaler setting.
        plot(bool): option to plot the elbow.
        fname(str): filename or path to save the plot.
        return_results(bool): option to return descriptive data.
    Return:
        Matplotlib elbow plot if plot=True.
        inertia(np.array): from model.inertia_ if return_results=True.
    """
    inertia = []
    for cluster in range(1,clusters):
        model, mask = cluster_k(dfX, k=cluster, scale=scale)
        inertia.append(model.inertia_)
    if plot!=False:
        frame = pd.DataFrame({'Cluster':range(1,clusters), 'SSE':inertia})
        plt.figure(figsize=(4,4))
        plt.plot(frame['Cluster'], frame['SSE'], marker='.', linewidth=0.5)
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow method to determine k-cluster number')
        if fname!=None:
            plt.savefig(fname=fname)
        plt.show()
    if return_results!=False:
        return inertia


def opt_neighbor_method(dfX, n=2, plot=False, fname=None, return_results=False):
    """
    A neighbor method to find optimum epsilon for DBSCAN.

    Args:
        dfX(pd.DataFrame or np.array): *arg from cluster_k().
        n(int): n_neighbors.
        plot(bool): option to plot the elbow.
        fname(str): filename or path to save the plot.
        return_results(bool): option to return descriptive data.
    Return:
        Matplotlib elbow plot if plot=True.
        model(?): from model.fit() if return_results=True.
        distances(np.array): distance values if return_results=True.
    """
    # __finding neighbors__
    neighbor = NearestNeighbors(n_neighbors=n)
    model = neighbor.fit(dfX)
    distances, indices = model.kneighbors(dfX)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    # __plotting condition__
    if plot!=False:
        plt.figure(figsize=(4,4))
        plt.plot(distances,  marker='.', linewidth=0.5)
        plt.title('K-distance graph')
        plt.xlabel('Data Points sorted by distance')
        plt.ylabel('Epsilon')
        # __saving condition__
        if fname!=None:
            plt.savefig(fname=fname)
        plt.show()
    # __return condition__
    if return_results!=False:
        return model, distances


def plot_clusters_img(cluster_col, w, h, cmap='Spectral', fname=None):
    """
    Plot image from based on clustered values.
    Args:
        cluster_col(pd.DataFrame): a column from df containing the cluster.
        w, h (int): width and height of the image.
        cmap(str): Matplotlib cmap of choise.
        fname(str): option to save the image.
    Return:
        Matplotlib image.
    """
    img_cluster = cluster_col.values.reshape(w, h)
    plt.imshow(img_cluster, cmap=cmap)
    ticks = [x for x in np.unique(cluster_col.values) if not np.isnan(x)]
    plt.colorbar(ticks=ticks)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if fname!=None:
        plt.savefig(fname=fname)
    plt.show()

def plot_spectra_clusters(
    dataframe, cluster_column_name, cmap='Spectral', drop_columns=None, shade=False,
    save_dpi=200, plot_dpi=80, figsize=(7, 4), legend='in', legend_size=7, wavenumber=None, wavenumber_xticks=10,
    title='Mean spectra of all clusters', fname=None, show_plot=True
    ):
    """
    Plot mean spectra based on cluster.

    Args:
        dataframe(pd.DataFrame): df containing data.
        cluster_column_name(str): name of cluster column.
        cmap(str): Matplotlib cmap name.
        drop_columns(tuple): range of columns to be dropped.
        shade(bool): option for SD shade for each cluster.
        save_dpi(int): dpi for saved figure.
        plot_dpi(int): dpi for plotted figue.
        figsize(tuple): figure size.
        legend(str): to be 'in' plot or 'out'.
        legend_size(int): size of legend.
        title(str): title name.
        fname(str): filename or save path with filename.
    Return:
        Matplotlib figure.
    """
    # get unique cluster number from the column
    unique_clusters = dataframe[cluster_column_name].unique()
    unique_clusters.sort() # <-this sort will tally to the image legend

    # === plotting ===
    plt.figure(dpi=plot_dpi, figsize=figsize)
    cmap, i = plt.get_cmap(cmap), 0 # <-for plot line colours
    for cluster_number in unique_clusters:

        # === this condition exclude NaN value ===
        if isinstance(cluster_number, int or float):
            print('True')
            if np.isnan(cluster_number):
                cluster_number = int(cluster_number)
                continue

        filtered_rows = dataframe[dataframe[cluster_column_name] == cluster_number]

        # === this condition define range columns to be exclude ===
        if drop_columns != None:
            filtered_rows = filtered_rows.drop(
                filtered_rows.iloc[:, drop_columns[0]:drop_columns[1]],axis = 1
                )
        average_row = filtered_rows.mean()

        # === colour mapping each cluster ===
        # i dont get this, but it works
        denominator = len(unique_clusters) - 1
        if denominator != 0:
            line_color = cmap(i / denominator)
        else:
            line_color = cmap(i / 1)
        if wavenumber!=None:
            plt.plot(
                average_row.index, average_row.values, linewidth=1.0,
                label=f'{cluster_number}', color=line_color,
                )
            # x = np.arange(0, len(wavenumber), 200)
            # x_labels = wavenumber[::1]
            # wavenumber_xticks = 10
            x = np.linspace(0, len(wavenumber) - 1, wavenumber_xticks, dtype=int)
            x_labels = [wavenumber[i] for i in x]
            plt.xticks(ticks=x, labels=x_labels, rotation=None)  # Adjust rotation if needed
            # plt.subplots_adjust(wspace=0.5)
        else:
            plt.plot(
                average_row.index, average_row.values, linewidth=1.0,
                label=f'{cluster_number}', color=line_color
                )

        if shade != False:
            # === this condition plot SD as shaded area ===
            std_row = filtered_rows.std()
            plt.fill_between(
                average_row.index.astype(float),
                average_row.values - std_row.values,
                average_row.values + std_row.values,  alpha=0.2
                )

        i += 1 # <-for cmap colour

    # === legend in or out condition ===
    if legend != 'in':
        legend = plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.subplots_adjust(right=0.7) # <-make space for the legend
    else:
        plt.legend(prop={'size': legend_size})

    # === other parameters ===
    plt.set_cmap('jet')
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Absorbance')
    plt.title(title)
    if fname != None:
        plt.savefig(fname=fname, dpi=save_dpi, bbox_inches='tight')
    if show_plot==True:
        plt.show()
    else:
        plt.close()

'''

def plot_spectra_clusters(
    dataframe, cluster_column_name, cmap='Spectral', drop_columns=None, shade=False,
    save_dpi=200, plot_dpi=80, figsize=(7, 4), legend='in', legend_size=7,
    title='Mean spectra of all clusters', fname=None, show_plot=True
    ):
    """
    Plot mean spectra based on cluster.

    Args:
        dataframe(pd.DataFrame): df containing data.
        cluster_column_name(str): name of cluster column.
        cmap(str): Matplotlib cmap name.
        drop_columns(tuple): range of columns to be dropped.
        shade(bool): option for SD shade for each cluster.
        save_dpi(int): dpi for saved figure.
        plot_dpi(int): dpi for plotted figue.
        figsize(tuple): figure size.
        legend(str): to be 'in' plot or 'out'.
        legend_size(int): size of legend.
        title(str): title name.
        fname(str): filename or save path with filename.
    Return:
        Matplotlib figure.
    """
    # get unique cluster number from the column
    unique_clusters = dataframe[cluster_column_name].unique()
    unique_clusters.sort() # <-this sort will tally to the image legend

    # === plotting ===
    plt.figure(dpi=plot_dpi, figsize=figsize)
    cmap, i = plt.get_cmap(cmap), 0 # <-for plot line colours
    for cluster_number in unique_clusters:

        # === this condition exclude NaN value ===
        if isinstance(cluster_number, int or float):
            print('True')
            if np.isnan(cluster_number):
                cluster_number = int(cluster_number)
                continue
            
        filtered_rows = dataframe[dataframe[cluster_column_name] == cluster_number]

        # === this condition define range columns to be exclude ===
        if drop_columns != None:
            filtered_rows = filtered_rows.drop(
                filtered_rows.iloc[:, drop_columns[0]:drop_columns[1]],axis = 1
                )
        average_row = filtered_rows.mean()

        # === colour mapping each cluster ===
        # i dont get this, but it works
        denominator = len(unique_clusters) - 1
        if denominator != 0:
            line_color = cmap(i / denominator)
        else:
            line_color = cmap(i / 1)

        plt.plot(
            average_row.index, average_row.values, linewidth=1.0,
            label=f'{cluster_number}', color=line_color
            )

        if shade != False:
            # === this condition plot SD as shaded area ===
            std_row = filtered_rows.std()
            plt.fill_between(
                average_row.index.astype(float),
                average_row.values - std_row.values,
                average_row.values + std_row.values,  alpha=0.2
                )

        i += 1 # <-for cmap colour

    # === legend in or out condition ===
    if legend != 'in':
        legend = plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.subplots_adjust(right=0.7) # <-make space for the legend
    else:
        plt.legend(prop={'size': legend_size})

    # === other parameters ===
    plt.set_cmap('jet')
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Absorbance')
    plt.title(title)
    if fname != None:
        plt.savefig(fname=fname, dpi=save_dpi, bbox_inches='tight')
    if show_plot==True:
        plt.show()
    else:
        plt.close()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def plot_clusters_distr(df, cluster_col, data_col, mode='boxplot', cmap='Spectral',
    return_median=False,
    yscale='linear', save_dpi=200, plot_dpi=80, figsize=(6, 4), title=None, fname=None):
    """
    Create violin plot based on clusters (x-axis), with data column (y-axis).

    Args:
        df(pd.DataFrame): dataframe.
        mode(str): either 'boxplot' or 'violinplot'
        cluster_col(str): name of cluster column.
        data_col(str): name of data column.
        return_median
        yscale(str): plt.yscale() to allow y-axis scaling i.e: 'log', 'symlog'.
        cmap(str): name of cmap colour of choice.
        save_dpi(int): dpi for saved figure.
        plot_dpi(int): dpi for plotted figue.
        figsize(tuple): figure size.
        title(str): title name.
        fname(str): filename or save path with filename.
    Return:
        Matplotlib figure.
    """

    """ set colour based on cmap for each clusters """
    cmap = plt.colormaps[cmap]
    num_colors = len(df[cluster_col].unique()) # get how many clusters
    color_values = np.linspace(0.1, 0.9, num_colors) # set colorspace
    colors = [cmap(color_values[i]) for i in range(num_colors)] # colour range

    """ plotting """
    plt.figure(figsize=figsize, dpi=plot_dpi)
    if mode=='violinplot':
        sns.violinplot(
            data=df, x=cluster_col, y=data_col, inner='quartile',
            linewidth=0.8, palette=colors)
        if title==None:
            plt.title(f"Violin Plot of {data_col} by {cluster_col}")
        if return_median==True:
            medians = df.groupby(cluster_col)[data_col].median()
    elif mode=='boxplot':
        sns.boxplot(
            data=df, x=cluster_col, y=data_col, linewidth=0.8, palette=colors,
            flierprops={"marker": "x"}, fliersize=0.5)
        if title==None:
            plt.title(f"Box and whisker plot of {data_col} by {cluster_col}")
        if return_median==True:
            medians = df.groupby(cluster_col)[data_col].median()
    plt.yscale(yscale)
    plt.xlabel(cluster_col)
    plt.ylabel(data_col)
    plt.tight_layout()
    if fname!=None:
        plt.savefig(fname=fname, dpi=save_dpi, bbox_inches='tight')
    plt.show()
    if return_median==True:
        return medians
'''
