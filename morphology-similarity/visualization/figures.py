"""
Contains functions to build Plotly figures.
"""

import plotly.graph_objs as go


def get_image_figure(data=None, height=75, width=300, margin=0):
    """
    Create a Plotly Heatmap to visualize our image data (akin to imshow)

    Arguments:
        data: An image, represented as a 2D NumPy array
        height: The desired height of the figure as an int in pixels
                (None will use the data's shape; default to 75)
        height: The desired width of the figure as an int in pixels
                (None will use the data's shape; default to 300)
        margin: The desired margin of the figure.  Supplying an int will
                set all margins to that value; otherwise takes a dict to
                be passed to Plotly.  Default to 0.

    Returns:
        A Plotly Heatmap figure for the data.
    """

    if height is None:
        height = data.shape[0]
    if width is None:
        width = data.shape[1]
    if height < 10:
        height = 10
    if width < 10:
        width = 10

    if margin is None:
        margin = dict(l=20, r=20, t=5, b=5)
    if isinstance(margin, int):
        margin = dict(l=margin, r=margin, t=margin, b=margin)

    # Generate Layout
    sample_fig_layout = go.Layout(
        height=height,
        width=width,
        showlegend=False,
        margin=margin,
        xaxis={
            'showgrid': False,
            'zeroline': False,
            'visible': False
        },
        yaxis={
            'showgrid': False,
            'zeroline': False,
            'visible': False
        },
    )

    # Generate Fig
    sample_fig = go.FigureWidget(
        data=go.Heatmap(
            z=data,
            hoverinfo='none',
            showscale=False
        ),
        layout=sample_fig_layout
    )
    return sample_fig


def get_distance_matrix_figure(
    dist_matrix,
    labels_x=None,
    labels_y=None,
    scaleanchor='x',
    showticklabels=False,
    title=None,
    xtitle=None,
    ytitle=None
):
    """
    Create a Plotly Heatmap to vizualize the provided distance matrix.

    Arguments:
        dist_matrix: An distance matrix, represented as a 2D NumPy array
        labels_x: The labels for the data in the x-direction
        labels_y: The labels for the data in the y-direction
        scaleanchor: Used to maintain the aspect ratio.  Defaults to 'x', making
                the y-axis scale to the x-axis.  None will ignore the aspect ratio.

    Returns:
        A Plotly Heatmap figure for the distance matrix.
    """
    dist_fig = go.FigureWidget(
        data=go.Heatmap(
            x=labels_x,
            y=labels_y,
            z=dist_matrix
        ),
        layout=go.Layout(
            title=title,
            coloraxis_showscale=False,
            height=600,
            width=600,
            xaxis=dict(
                title=xtitle,
                autorange=True,
                showgrid=False,
                ticks='inside',
                tickmode='array',
                type='category',
                showticklabels=showticklabels
            ),
            yaxis=dict(
                title=ytitle,
                scaleanchor=scaleanchor,
                autorange=True,
                showgrid=False,
                ticks='inside',
                tickmode='array',
                type='category',
                showticklabels=showticklabels
            )
        )
    )
    return dist_fig
