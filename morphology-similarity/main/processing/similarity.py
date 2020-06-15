"""
Contains the functions to compute the distance between images.
"""
import numpy as np


def compute_distance(image_1, image_2, macro, signature):
    """
    Compute the distance between the 2 images based on their morphology
    Args:
        image_1 (ndarray): 
            First image to be compared (Single channel/grayscale)
        image_2 (ndarray):
            Second image to be compared (Single channel/grayscale)
        macro (str):
            name of the function that computes the macro/structural
            similarity between the images
        signature (str):
            name of the function that computes distance between the
            individual components of the image

    Return:
        distance (float): 
            Distance computed between the 2 images

    TODO:
    - Write sanity functions
    """
    if macro == "BFS":
        # generate the region adjacency graph of both the images
        image_1_rag = processing.generate_region_adjacency_graph(
            image_1, 
            signature)
        image_2_rag = processing.generate_region_adjacency_graph(
            image_2, 
            signature)

        # genrate the vector representation of the graph
        image_1_vector = processing.generate_bfs_vector(image_1_rag)
        image_2_vector = processing.generate_bfs_vector(image_2_rag)

        # make the 2 vectors of equal length
        vector_pair = [image_1_vector, image_2_vector]
        padded_vectors = processing.generate_padded_vectors(vector_pair)

        # compute the euclidean distance of the 2 vectors
        distance = np.linalg.norm(
            padded_vectors[0]-padded_vectors[1])

        return distance
