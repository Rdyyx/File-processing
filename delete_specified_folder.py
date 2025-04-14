import os
import re
import shutil
import glob
import os


def rename_files(directory, pattern, new_name):
    files = glob.glob(os.path.join(directory, pattern))
    for old_file in files:
        new_file = os.path.join(directory, new_name)
        os.rename(old_file, new_file)

def delete_specified_folder(parent_file):
    """
    Delete the folder containing the name "instances" from the folder
    Delete the folder that contains the word "segmentation" from the folder

    Args:
        parent_file (str): Parent folder name

    Returns:
        None
    """
    count1 = 0
    count2 = 0

    if os.path.exists(parent_file):
        for case_name in os.listdir(parent_file): # case_00000
            for file_name in os.listdir(parent_file+'/'+case_name):

                pat1 = r'instances'
                if re.search(pat1, file_name) is not None:
                    file_path = os.path.join(parent_file+'/'+case_name, file_name)
                    shutil.rmtree(file_path) # Delete folder
                    print(f'old_file{file_path}')
                    count1 += 1

                pat2 = r'segmentation'
                if re.search(pat2, file_name) is not None:
                    file_path = os.path.join(parent_file + '/' + case_name, file_name)
                    os.remove(file_path) # Delete file
                    print(f'old_file{file_path}')
                    count2 += 1

    print(f'count1: {count1}')
    print(f'count2: {count2}')




# Sample call

parent_file = "data/kit23_new"
delete_specified_folder(parent_file)

