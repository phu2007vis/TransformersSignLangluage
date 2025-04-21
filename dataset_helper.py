
from pathlib import Path
import os
import pathlib
from typing import List, Optional, Tuple
from iopath.common.file_io import g_pathmgr

def get_labelmap(dataset_root_path):
	if not isinstance(dataset_root_path,Path):
		dataset_root_path = Path(dataset_root_path)

	number_classes = max(find_number_classes(os.path.join(dataset_root_path,phase,'rgb')) for phase in ['train','val','test'])
	label2id = {label: int(label) for  label in range(1,number_classes+1)}
	id2label = {i: label for label, i in label2id.items()}
	
	print(f"Unique classes: {list(label2id.keys())}.")
	return label2id,id2label



class LabeledVideoPaths:
	"""
	LabeledVideoPaths contains pairs of video path and integer index label.
	"""

	@classmethod
	def from_path(cls, data_path: str) :
		"""
		Factory function that creates a LabeledVideoPaths object depending on the path
		type.
		- If it is a directory path it uses the LabeledVideoPaths.from_directory function.
		- If it's a file it uses the LabeledVideoPaths.from_csv file.
		Args:
			file_path (str): The path to the file to be read.
		"""

		if g_pathmgr.isdir(data_path):
			return LabeledVideoPaths.from_directory(data_path)
		else:
			raise FileNotFoundError(f"{data_path} not found.")


	@classmethod
	def from_directory(cls, dir_path: str) :
		"""
		Factory function that creates a LabeledVideoPaths object by parsing the structure
		of the given directory's subdirectories into the classification labels. It
		expects the directory format to be the following:
			 dir_path/<class_name>/<video_name>.mp4

		Classes are indexed from 0 to the number of classes, alphabetically.

		E.g.
			dir_path/rgb/xxx.ext
			dir_path/rgb/xxy.ext
			dir_path/rgb/xxz.ext

		  

		Would produce two classes labeled 0 and 1 with 3 videos paths associated with each.

		Args:
			dir_path (str): Root directory to the video class directories .
		"""
		assert g_pathmgr.exists(dir_path), f"{dir_path} not found."

		# Find all classes based on directory names. These classes are then sorted and indexed
		# from 0 to the number of classes.
	
		video_paths_and_label = make_dataset(
			dir_path, class_to_idx = None, extensions=("mp4", "avi")
		)
		assert (
			len(video_paths_and_label) > 0
		), f"Failed to load dataset from {dir_path}."
		return cls(video_paths_and_label)

	def __init__(
		self, paths_and_labels: List[Tuple[str, Optional[int]]], path_prefix=""
	) -> None:
		"""
		Args:
			paths_and_labels [(str, int)]: a list of tuples containing the video
				path and integer label.
		"""
		self._paths_and_labels = paths_and_labels
		self._path_prefix = path_prefix

	def path_prefix(self, prefix):
		self._path_prefix = prefix

	path_prefix = property(None, path_prefix)

	def __getitem__(self, index: int) -> Tuple[str, int]:
		"""
		Args:
			index (int): the path and label index.

		Returns:
			The path and label tuple for the given index.
		"""
		path, label = self._paths_and_labels[index]
		return (os.path.join(self._path_prefix, path), {"label": label})

	def __len__(self) -> int:
		"""
		Returns:
			The number of video paths and label pairs.
		"""
		return len(self._paths_and_labels)


from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
	"""Checks if a file is an allowed extension.

	Args:
		filename (string): path to a file
		extensions (tuple of strings): extensions to consider (lowercase)

	Returns:
		bool: True if the filename ends with one of given extensions
	"""
	return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_image_file(filename: str) -> bool:
	"""Checks if a file is an allowed image extension.

	Args:
		filename (string): path to a file

	Returns:
		bool: True if the filename ends with a known image extension
	"""
	return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_number_classes(directory: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
	"""Finds the class folders in a dataset.

	See :class:`DatasetFolder` for details.
	"""
	
	max_class = 0
	for entry in os.scandir(directory):
		if entry.is_file():
				try:
					class_id = int(os.path.splitext(entry.name)[0].split('P')[0].split('A')[1])
				except:
					print(entry, "Not follow the formate _A_P_.avi or .mp4 skip")
					continue
				max_class =  max_class if class_id is None else max(max_class, class_id)
	
	return max_class

from glob import glob

def make_dataset(
	directory: Union[str, Path],
	class_to_idx: Optional[Dict[str, int]] = None,
	extensions: Optional[Union[str, Tuple[str, ...]]] = None,
	is_valid_file: Optional[Callable[[str], bool]] = None,
	allow_empty: bool = False,
) -> List[Tuple[str, int]]:
	"""Generates a list of samples of a form (path_to_sample, class).

	See :class:`DatasetFolder` for details.

	Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
	by default.
	"""
	directory = os.path.expanduser(directory)
	number_classes = find_number_classes(directory)
	
	is_valid_file = cast(Callable[[str], bool], is_valid_file)

	instances = []
	available_classes = set()
	class_to_idx = {i: i for i in range(1, number_classes + 1)}
 
	for class_index in class_to_idx:
	
		rgb_target_path = glob(os.path.join(directory, f"*A{class_index}P*.avi"))+glob(os.path.join(directory, f"*A{class_index}P*.mp4"))
		# print(f"Class {class_index} has {len(rgb_target_path)} files!")
		for path  in sorted(rgb_target_path):
				
			item = path, class_index
			instances.append(item)

			if class_index not in available_classes:
				available_classes.add(class_index)
			
	
	#debuging siuuu
	empty_classes = set(class_to_idx.keys()) - available_classes
	if empty_classes and not allow_empty:
		msg = f"Found no valid file for the classes {', '.join(sorted(str(empty_classes)))}. "
		if extensions is not None:
			msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
		raise FileNotFoundError(msg)

	return instances
