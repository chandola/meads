# Morphology Similarity

Code related to finding the similarity between morphologies.  This covers image processing, signature extraction, and distance measures.

## Setup

There's a lot of ways to set this up.  If you're using Anaconda, you're likely to have most of the packages installed already and might just want to install whatevers left-over along the way.

Here's my setup, but it can easily be replaced with something like an Anaconda environment:

1) Create a virtual Python 3.6 environment: `python -m venv venv`
2) Activate the environment: `source venv/bin/activate`
3) Install dependencies: `pip install -r requirements`
4) Add the Jupyter kernel: `python -m ipykernel install --name=morphology-similarity`

After this, you should be able to start your Jupyter instance and select `morphology-similarity` for your kernel to run these notebooks.  The notebooks use a small dataset for experiments.  This data, a single Pandas DataFrame contained in a pickle file, can be downloaded from Google Drive by running `sh start`.  This will save the file to `data/sample_morphologies.pickle`.
