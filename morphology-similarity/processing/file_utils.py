from pathlib import Path
from tqdm.notebook import tqdm
import pickle


ALLOWED_FILE_TYPES = ['png', 'jpg']

def load_morphologies(selected_trajectories: list, source_dir: Path):
	"""
    Return a list of filenames of every morphology for  the given trajectories. 
    Args:
        selected_trajectories (list): 
            list of trajectories (folder names) to be read
        source_dir (Path):
            source 
    Return:
        all_morphology_file_names (list): 
            list of Path objects corresponding to each morphology
    """
	all_morphology_file_names = []
	for trajectory in tqdm(selected_trajectories):
	    trajectory_dir = source_dir / trajectory
	    assert trajectory_dir.exists(), f"trajectory \"{trajectory}\" not found in the source directory {source_dir}"
	    
	    # get all morphologies in the trajectory_dir and filter out any other files
	    directory_morphologies = filter(lambda x: x.suffix[1:] in ALLOWED_FILE_TYPES, trajectory_dir.iterdir())
	    
	    # sort the morphology file_names based on their index number
	    sorted_morphology_file_names = sorted([x for x in directory_morphologies], key=lambda x: int(x.name.split(".")[0]))
	    
	    # append to master list
	    all_morphology_file_names+=(sorted_morphology_file_names)

	return all_morphology_file_names


def load_Last_morphology(selected_trajectories: list, source_dir: Path):
	"""
    Return a list of filenames of the last morphology for the given trajectories. 
    Args:
        selected_trajectories (list): 
            list of trajectories (folder names) to be read
        source_dir (Path):
            source 
    Return:
        all_morphology_file_names (list): 
            list of Path objects corresponding to each morphology
    """
	last_morphology_file_names = []
	for trajectory in tqdm(selected_trajectories):
	    trajectory_dir = source_dir / trajectory
	    assert trajectory_dir.exists(), f"trajectory \"{trajectory}\" not found in the source directory {source_dir}"
	    
	    # get all morphologies in the trajectory_dir and filter out any other files
	    directory_morphologies = filter(lambda x: x.suffix[1:] in ALLOWED_FILE_TYPES, trajectory_dir.iterdir())
	    
	    # sort the morphology file_names based on their index number
	    sorted_morphology_file_names = sorted([x for x in directory_morphologies], key=lambda x: int(x.name.split(".")[0]))
	    
	    # append to master list
	    last_morphology_file_names.append(sorted_morphology_file_names[-1])

	    # print file name to make sure same last index is the same 
	    # for all the morphologies
	    print(f"LOADED: trajectory->{trajectory}; index->{sorted_morphology_file_names[-1]}")

	return last_morphology_file_names


def pickle_variable(variable, filename: str, location: Path = None):
	"""
	Store a variable as a pickle to be loaded later
	Args:
		variable:
			the variable to be pickled. Any file type that is
			supported by the pickle library
		filename (str):
			name of the file to whcih the variable will be pickled to
		location (Path):
			if the location should not be the default location (optional)
	Return:
		None
	"""
	DEFAULT_PICKLE_LOCATION = Path("/home/namit/codes/meads/morphology-similarity/playground/.cache/pickled_variables")

	pickle_file = (location or DEFAULT_PICKLE_LOCATION) / filename

	with open(pickle_file, 'wb') as f:
		pickle.dump(variable, f)



def unpickle_variable(filename: str, location: Path = None):
	"""
	Fetch a variable that has been pickled
	Args:
		filename (str):
			name of the file to which the variable has been pickled to
		location (Path):
			location where the pickle file shall be found. 
			defaults to the default location. 
	Return:
		pickled_variable

	TODO:
		Add if file exists check
	"""
	DEFAULT_PICKLE_LOCATION = Path("/home/namit/codes/meads/morphology-similarity/playground/.cache/pickled_variables")

	pickle_file = (location or DEFAULT_PICKLE_LOCATION) / filename

	with open(pickle_file, 'rb') as f:
		return pickle.load(f)

