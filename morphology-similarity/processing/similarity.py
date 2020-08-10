"""
Contains the functions to compute the distance between images.
"""
import numpy as np
import processing
import pickle
from pathlib import Path


def compute_distance(image_1, image_2, macro, signature, cache_keys):
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
        cache_keys (list):
            values used to identify the image and trajectory for caching

    Return:
        distance (float): 
            Distance computed between the 2 images

    TODO:
    - Write sanity functions
    """
    if macro == "BFS":
        
        # generate the region adjacency graph of both the images
        if cache_keys:
            image_1_rag = get_rag(image_1, signature, cache_keys[0].split(':'))
            image_2_rag = get_rag(image_2, signature, cache_keys[1].split(':'))
        else:
            image_1_rag = get_rag(image_1, signature, None, None)
            image_2_rag = get_rag(image_2, signature, None, None)

        # generate the vector representation of the graph
        image_1_vector = processing.generate_bfs_vector(image_1_rag)
        image_2_vector = processing.generate_bfs_vector(image_2_rag)

        # make the 2 vectors of equal length
        vector_pair = [image_1_vector, image_2_vector]
        padded_vectors = processing.generate_padded_vectors(vector_pair)

        # compute the euclidean distance of the 2 vectors
        distance = np.linalg.norm(padded_vectors[0]-padded_vectors[1], 1)

        return distance




def get_rag(image, signature, trajectory_name, image_index):
    """
    Generate the Region Adjacency graph corresponding to a morphology
    Also implements caching using pickle
    Args:
        image (ndarray):
            the morphology
        signature (string):
            the signature used to assign weight to the nodes of the graph
        trajectory_name (string):
            name of the trajectory the morphology belongs to, used for caching
        image_index (string):
            image number in the morphology used for caching

    Return:
        rag (networkx.graph):
            the region adjaccency graph
    """

    

    try:
        if trajectory_name:
            cache_location = Path(f"./.cache/RAG/{trajectory_name}/{signature}/")
            return pickle.load(open(cache_location/f'{image_index}', 'rb'))
        else:
            raise IOError

    except IOError:
        print("Handling exception")
        rag = processing.generate_region_adjacency_graph(
            image, 
            signature)

        if trajectory_name:
            # make sure the location exists
            cache_location.mkdir(parents=True, exist_ok=True)
            # cache the file
            pickle.dump(rag, open(cache_location/f'{image_index}', "wb"))

        return rag
