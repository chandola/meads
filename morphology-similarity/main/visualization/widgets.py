"""
Contains functions to build interactive Jupyter Notebook widgets.
"""

import json
import ipywidgets as widgets
import pandas as pd
import numpy as np
from .figures import get_distance_matrix_figure, get_image_figure

def binarize(image, thresh=0.5):
    return (image > thresh).astype(int)

def reshape_image(image):
    image = np.array(image)
    try:
        return image.reshape((100, 400))
    except ValueError as e:
        image = image.reshape((100, 800))
        return image[:400]

def get_distance_matrix_dataframe_widget(dist_matrix, df_x, df_y=None, zmax=None):
    """
    Creates a widget which displays a Plotly Heatmap representing the provided distance matrix.
    Clicking on a cell in the distance matrix displays the two underlying morphologies.

    Arguments:
        dist_matrix: A distance matrix, represented as a 2D Numpy array.
        df_x: A Pandas DataFrame corresponding to the columns (x-direction) of the distance matrix.
        df_y: A Pandas DataFrame corresponding to the rows (y-direction) of the distance matrix.
                Default to None, which assumes df_y = df_x.
        zmax: Set the maximum for the color-scale in the distance matrix (Defaults to None for no max).

    Returns:
        An interactive ipywidget (viewable in Jupyter)
    """
    if df_y is None:
        df_y = df_x

    # These are the hover labels
    labels_x = [
        json.dumps(dict(row[['BR', 'CHI', 'version', 'timestep']]))
        for idx, row in df_x.iterrows()
    ]
    labels_y = [
        json.dumps(dict(row[['BR', 'CHI', 'version', 'timestep']]))
        for idx, row in df_y.iterrows()
    ]
    dist_fig = get_distance_matrix_figure(
        dist_matrix,
        labels_x=labels_x,
        labels_y=labels_y
    )

    # The stand-in morphology visualizations.
    sample_fig_x = get_image_figure(margin=None)
    sample_fig_y = get_image_figure(margin=None)

    # Prints the morphology information.
    out_x = widgets.Output()
    out_y = widgets.Output()

    # This gets triggered when the Heatmap is clicked on.
    def show_samples(trace, points, selector):
        del trace, selector # unused
        # Get the params from the clicked cell
        x_params = pd.Series(json.loads(points.xs[0]))
        # Find it in the DataFrame
        sample_x = df_x[(df_x[['BR', 'CHI', 'version', 'timestep']] == x_params).all(1)].iloc[0]
        # Set the morphology viz to this image
        sample_fig_x.data[0].z = reshape_image(sample_x.image)
        # Clear the output, then print the morph. info to it
        out_x.clear_output()
        with out_x:
            print("Sample:")
            print(sample_x[['BR', 'CHI', 'version', 'timestep']])

        # Do the same as above, but for the y-datapoint.
        y_params = pd.Series(json.loads(points.ys[0]))
        sample_y = df_y[(df_y[['BR', 'CHI', 'version', 'timestep']] == y_params).all(1)].iloc[0]
        sample_fig_y.data[0].z = reshape_image(sample_y.image)
        out_y.clear_output()
        with out_y:
            print("Sample:")
            print(sample_y[['BR', 'CHI', 'version', 'timestep']])

    # Set the click callback on the distance Heatmap to the above function
    dist_fig.data[0].on_click(show_samples)
    if zmax is not None:
        dist_fig.data[0].update(zmin=0., zmax=zmax)

    # We use VBox and HBox to structure the visualizations
    return widgets.VBox([
        dist_fig,
        widgets.HBox([
            widgets.VBox([
                sample_fig_x,
                out_x
            ], layout={'max_width': '300px'}),
            widgets.VBox([
                sample_fig_y,
                out_y
            ], layout={'max_width': '300px'})
        ], layout=widgets.Layout(
            display='flex',
            justify_content='flex-start'
        ))
    ])


def get_distance_matrix_widget(dist_matrix, traj_x, traj_y=None, binarize_sample=True, zmax=None):
    """
    Creates a widget which displays a Plotly Heatmap representing the provided distance matrix.
    Clicking on a cell in the distance matrix displays the two underlying morphologies.

    Arguments:
        dist_matrix: A distance matrix, represented as a 2D Numpy array.
        traj_x: A Trajectory corresponding to the columns (x-direction) of the distance matrix.
        traj_y: A Trajectory corresponding to the rows (y-direction) of the distance matrix.
                Default to None, which assumes traj_y = traj_x.
        zmax: Set the maximum for the color-scale in the distance matrix (Defaults to None for no max).

    Returns:
        An interactive ipywidget (viewable in Jupyter)
    """
    if traj_y is None:
        traj_y = traj_x

    # These are the hover labels
    labels_x = [
        str(morph)
        for morph in traj_x.morphologies
    ]
    labels_y = [
        str(morph)
        for morph in traj_y.morphologies
    ]
    dist_fig = get_distance_matrix_figure(
        dist_matrix,
        labels_x=labels_x,
        labels_y=labels_y
    )

    # The stand-in morphology visualizations.
    sample_fig_x = get_image_figure(margin=None)
    sample_fig_y = get_image_figure(margin=None)

    # Prints the morphology information.
    out_x = widgets.Output()
    out_y = widgets.Output()

    # This gets triggered when the Heatmap is clicked on.
    def show_samples(trace, points, selector):
        del trace, selector # unused
        # Get the params from the clicked cell
        x_params = points.xs[0]
        # Find it in the trajectory
        for morph in traj_x.morphologies:
            if str(morph) == x_params:
                break
        # Set the morphology viz to this image
        image = morph.image
        if binarize_sample:
            image = binarize(image)
        sample_fig_x.data[0].z = image
        # Clear the output, then print the morph. info to it
        out_x.clear_output()
        with out_x:
            print("Sample:")
            print(morph)

        # Do the same as above, but for the y-datapoint.
        y_params = points.ys[0]
        for morph in traj_y.morphologies:
            if str(morph) == y_params:
                break
        image = morph.image
        if binarize_sample:
            image = binarize(image)
        sample_fig_y.data[0].z = image
        out_y.clear_output()
        with out_y:
            print("Sample:")
            print(morph)

    # Set the click callback on the distance Heatmap to the above function
    dist_fig.data[0].on_click(show_samples)
    if zmax is not None:
        dist_fig.data[0].update(zmin=0., zmax=zmax)

    # We use VBox and HBox to structure the visualizations
    return widgets.VBox([
        dist_fig,
        widgets.HBox([
            widgets.VBox([
                sample_fig_x,
                out_x
            ], layout={'max_width': '300px'}),
            widgets.VBox([
                sample_fig_y,
                out_y
            ], layout={'max_width': '300px'})
        ], layout=widgets.Layout(
            display='flex',
            justify_content='flex-start'
        ))
    ])
