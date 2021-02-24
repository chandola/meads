"""
Contains the functions to compute the distance between images.
"""
import numpy as np
import processing
import pickle
from pathlib import Path
from skimage import io
from skimage.color import rgb2gray

CACHE_ENABLED = True

def get_distance_matrix(morphologies: list, macro: str, signature: str):
    """
    Generate a distance matrix for the given list of morphologies
    Args:
        morphologies (list): 
            list of morphology filenames
        macro (str):
            name of the function that computes the macro/structural
            similarity between the images
        signature (str):
            name of the function that computes distance between the
            individual components of the image
             
    Return:
        distance_matrix (ndarray): 
            list of Path objects corresponding to each morphology

    TODO:
        get the imread operation to not be executed 
        if cache is going to be used. 
    """
    morphology_count = len(morphologies)
    distance_matrix = np.empty((morphology_count, morphology_count))

    for m1_idx, morphology_file_1 in enumerate(morphologies):
        for m2_idx, morphology_file_2 in enumerate(morphologies):

            # read the image and convert it to greyscale
            morphology_1 = io.imread(morphology_file_1)
            morphology_2 = io.imread(morphology_file_2)

            gray_morphology_1 = rgb2gray(morphology_1)
            gray_morphology_2 = rgb2gray(morphology_2)


            # compute similarity
            distance = processing.compute_distance(gray_morphology_1, 
                                                   gray_morphology_2, 
                                                   macro, 
                                                   signature,
                                                   [get_cache_key(morphology_file_1), 
                                                    get_cache_key(morphology_file_2)])

            # populate the distance matrix
            distance_matrix[m1_idx, m2_idx] = distance

    return distance_matrix

def get_cache_key(morphology_file: Path):
    """
    Generate cache key for the given morphology. 
    cache_key = trajectory_name + morphology_index
    morphology_index is the filename
    trajectory_name is container directories name. 
    might not be true for a different morphology organizational structure. 
    Args:
        morphology_file (Path): 
            Path to the morphology file. 

    Return:
        cache_key (string): 
            Unique based on trajectory name and file name
    """
    file_parents = morphology_file.parents
    trajectory_name = file_parents[0].name
    morphology_index = morphology_file.stem

    return trajectory_name + "_" + morphology_index

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
            values used to identify the morphology and trajectory for caching

    Return:
        distance (float): 
            Distance computed between the 2 images

    TODO:
    - Write sanity functions
    """
    if macro == "BFS":
        
        # generate the region adjacency graph of both the images
        if cache_keys:
            image_1_rag = get_rag(image_1, signature, cache_keys[0])
            image_2_rag = get_rag(image_2, signature, cache_keys[1])
        else:
            image_1_rag = get_rag(image_1, signature)
            image_2_rag = get_rag(image_2, signature)

        # generate the vector representation of the graph
        image_1_vector = processing.generate_bfs_vector(image_1_rag)
        image_2_vector = processing.generate_bfs_vector(image_2_rag)

        # make the 2 vectors of equal length
        vector_pair = [image_1_vector, image_2_vector]
        padded_vectors = processing.generate_padded_vectors(vector_pair)

        # compute the euclidean distance of the 2 vectors
        distance = np.linalg.norm(padded_vectors[0]-padded_vectors[1], 1)

        return distance

def get_rag(image, signature, cache_key=None):
    """
    Generate the Region Adjacency graph corresponding to a morphology
    Also implements caching using pickle
    Args:
        image (ndarray):
            the morphology
        signature (string):
            the signature used to assign weight to the nodes of the graph
        cache_key (string):
            something that uniquely identifies the morphology, usually a 
            combination of trajectory name and morphology index

    Return:
        rag (networkx.graph):
            the region adjaccency graph
    """

    

    try:
        cache_location = Path(f"./.cache/RAG/{cache_key}_{signature}")
        if CACHE_ENABLED and cache_key:
            # cache key = cache key + signature
            print(f"FOUND CACHE AT: {cache_location}")
            return pickle.load(open(cache_location, 'rb'))
        else:
            raise IOError

    except IOError:
        print("Handling exception")
        rag = processing.generate_region_adjacency_graph(
            image, 
            signature)

        if cache_key:
            # make sure the location exists
            # cache_location.mkdir(parents=True, exist_ok=True)
            # cache the file
            pickle.dump(rag, open(cache_location, "wb"))
            print(f"CACHED AT: {cache_location}")

        return rag

def get_pxbypx_distance_matrix(morphologies: list):
    """
    Generate a distance matrix for the given list of morphologies based 
    on pixel by pixel distance
    Args:
        morphologies (list): 
            list of morphology filenames
             
    Return:
        distance_matrix (ndarray): 
            list of Path objects corresponding to each morphology

    TODO:
        get the imread operation to not be executed 
        if cache is going to be used. 
    """
    morphology_count = len(morphologies)
    distance_matrix = np.empty((morphology_count, morphology_count))

    for m1_idx, morphology_file_1 in enumerate(morphologies):
        for m2_idx, morphology_file_2 in enumerate(morphologies):

            # read the image and convert it to greyscale
            morphology_1 = io.imread(morphology_file_1)
            morphology_2 = io.imread(morphology_file_2)

            # grayscale the image
            gray_morphology_1 = rgb2gray(morphology_1)
            gray_morphology_2 = rgb2gray(morphology_2)

            # binarize the image
            binary_morphology_1 = (gray_morphology_1 > 0.5).astype(int)
            binary_morphology_2 = (gray_morphology_2 > 0.5).astype(int)


            # compute pixel by pixel similarity
            distance = np.sum(np.absolute(binary_morphology_1 - \
                                            binary_morphology_2))

            # populate the distance matrix
            distance_matrix[m1_idx, m2_idx] = distance

    return distance_matrix

def get_2dfft_distance_matrix(morphologies: list):
    """
    Generate a distance matrix for the given list of morphologies based 
    on 2d fft distance
    Args:
        morphologies (list): 
            list of morphology filenames
             
    Return:
        distance_matrix (ndarray): 
            list of Path objects corresponding to each morphology

    TODO:
        get the imread operation to not be executed 
        if cache is going to be used. 
    """
    morphology_count = len(morphologies)
    distance_matrix = np.empty((morphology_count, morphology_count))

    for m1_idx, morphology_file_1 in enumerate(morphologies):
        for m2_idx, morphology_file_2 in enumerate(morphologies):

            # read the image and convert it to greyscale
            morphology_1 = io.imread(morphology_file_1)
            morphology_2 = io.imread(morphology_file_2)

            # grayscale the image
            gray_morphology_1 = rgb2gray(morphology_1)
            gray_morphology_2 = rgb2gray(morphology_2)

            # binarize the image
            binary_morphology_1 = (gray_morphology_1 > 0.5).astype(int)
            binary_morphology_2 = (gray_morphology_2 > 0.5).astype(int)

            # generate the 2d fft of the image
            fft_morphology_1 = np.fft.fft2(binary_morphology_1)
            fft_morphology_2 = np.fft.fft2(binary_morphology_2)

            # compute pixel by pixel similarity
            distance = np.sum(np.absolute(fft_morphology_1 - \
                                            fft_morphology_2))

            # populate the distance matrix
            distance_matrix[m1_idx, m2_idx] = distance

    return distance_matrix