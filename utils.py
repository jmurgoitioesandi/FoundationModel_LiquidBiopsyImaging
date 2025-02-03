import tarfile
import os
import shutil


def decompress_directory(tar_gz_path, dest_dir):
    """
    Decompress a .tar.gz file to the specified directory.

    Parameters:
    tar_gz_path (str): The path to the .tar.gz file.
    dest_dir (str): The directory where the contents should be extracted.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    with tarfile.open(tar_gz_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)


def remove_directory(dir_path):
    """
    Remove the original directory.

    Parameters:
    dir_path (str): The path to the directory to be remove.
    """

    shutil.rmtree(dir_path)
