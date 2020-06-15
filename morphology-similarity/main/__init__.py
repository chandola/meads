"""
Classes for morphology management and processing.
"""
from .processing import *
from .visualization import *

import os
from multiprocessing import Process, Manager


class Morphology:
    """A morphological data point"""
    
    def __init__(self, br, chi, version, timestep):
        self.br = float(br)
        self.chi = float(chi)
        self.version = int(version)
        self.timestep = int(timestep)
        
        self.image = None
        self.components = None
        self.signatures = {}
        
        self.data_path = f'../data/trajectories/{self.br}/{self.chi}/{self.version}/{self.timestep:03}'
        os.makedirs(f'{self.data_path}/signatures', exist_ok=True)
        
        if os.path.isfile(f'{self.data_path}/image.npy'):
            self.load_image()
        if os.path.isfile(f'{self.data_path}/components.npy'):
            self.load_components()
            self.load_signatures()
        
    def load_image(self):
        self.image = np.load(f'{self.data_path}/image.npy')
    
    def cache_image(self, image):
        np.save(f'{self.data_path}/image', reshape_image(image))
        
    def load_components(self):
        self.components = np.load(f'{self.data_path}/components.npy')
        
    def cache_components(self):
        if self.image is None:
            self.load_image()
        components = extract_components(self.image, background=1)
        np.save(f'{self.data_path}/components', components)
        self.load_components()
        
    def load_signatures(self):
        sig_names = sorted(os.listdir(f'{self.data_path}/signatures'))
        for sig_name in sig_names:
            self.load_signature(sig_name.replace('.npy', ''))
    
    def load_signature(self, sig_name):
        self.signatures[sig_name] = np.load(f'{self.data_path}/signatures/{sig_name}.npy')
        
    def cache_signature(self, sig_func):
        if self.components is None:
            self.load_components()
        signature = apply_to_components(self.components, sig_func)
        np.save(f'{self.data_path}/signatures/{sig_func.__name__}', signature)
        self.load_signature(sig_func.__name__)
        
    def viz(self):
        return get_image_figure(self.image)
        
    def __repr__(self):
        return f'BR: {self.br}, CHI: {self.chi}, ver: {self.version}, t: {self.timestep}'

    
class Trajectory:
    """The collection of morphologies produced from the evolution of the morphology"""
    
    def __init__(self, br, chi, version):
        self.br = br
        self.chi = chi
        self.version = version
        self.data_path = f'../data/trajectories/{br}/{chi}/{version}'
        os.makedirs(self.data_path, exist_ok=True)
        
        self.timesteps = []
        self.morphologies = []
        self.load_morphologies()
        
    def __repr__(self):
        return f'BR: {self.br}, CHI: {self.chi}, ver: {self.version}'
        
    def load_morphologies(self):
        self.timesteps = sorted(os.listdir(self.data_path))
        self.morphologies = [
            Morphology(
                br=self.br,
                chi=self.chi,
                version=self.version,
                timestep=timestep
            ) for timestep in self.timesteps
        ]
    
    def cache_morphologies(self, images, timesteps, cache_components=True, replace=False):
        if timesteps is None:
            timesteps = range(len(images))
        images = list(images)
        timesteps = list(timesteps)
        for image, timestep in zip(images, timesteps):
            if replace and timestep in self.timesteps:
                continue
            morph = Morphology(
                br=self.br,
                chi=self.chi,
                version=self.version,
                timestep=timestep
            )
            morph.cache_image(image)
            if cache_components:
                morph.cache_components()
        self.load_morphologies()
        
    def cache_signatures(self, sig_func):
        for morph in self.morphologies:
            morph.cache_signature(sig_func)
        

def load_trajectory_from_dataframe(df):
    sample = df.iloc[0]
    traj = Trajectory(br=sample.BR, chi=sample.CHI, version=sample.version)
    traj.cache_morphologies(df.image, df.timestep)
    return traj

def get_dist_matrix(traj_x, traj_y, sig_func):
    dist_matrix = []

    for morph_x in traj_x.morphologies:
        row = []
        sig_x = apply_to_components(morph_x.components, sig_func)
        for morph_y in traj_y.morphologies:
            sig_y = apply_to_components(morph_y.components, sig_func)
            
            if len(sig_x) == 0:
                sig_x = [0.]
            if len(sig_y) == 0:
                sig_y = [0.]
            row.append(emd_samples(sig_x, sig_y))
        dist_matrix.append(row)

    return np.array(dist_matrix)