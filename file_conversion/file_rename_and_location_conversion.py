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

def move_file_to_parentfile(parent_file):
    """
    Transfer the file containing the word "tumsor" in the folder to its parent folder and rename it

    Args:
        parent_file (str): Parent folder name

    Returns:
        None
    """
    count1 = 0
    count2 = 0

    if os.path.exists(parent_file):
        for case_name in os.listdir(parent_file): # case_00000
            for file_name in os.listdir(parent_file+'/'+case_name+'/'+'instances'): # tumor

                pat1 = r'tumor_instance-1_annotation-1'
                if re.search(pat1, file_name) is not None:
                    file_path = os.path.join(parent_file+'/'+case_name+'/'+'instances', file_name)
                    shutil.move(file_path,parent_file+'/'+case_name) # Move to the parent directory
                    rename_files(parent_file+'/'+case_name,file_name,'tumor.nii.gz')
                    new_file_path = parent_file+'/'+case_name+'/'+'tumor.nii.gz'
                    print(f'old_file{file_path}\nnew_file{new_file_path}')
                    count1 += 1

                pat2 = r'cyst_instance-1_annotation-1'
                if re.search(pat2, file_name) is not None:
                    file_path = os.path.join(parent_file + '/' + case_name + '/' + 'instances', file_name)
                    shutil.move(file_path, parent_file + '/' + case_name)  # Move to the parent directory
                    rename_files(parent_file + '/' + case_name, file_name, 'cyst.nii.gz')
                    new_file_path = parent_file + '/' + case_name + '/' + 'cyst.nii.gz'
                    print(f'old_file{file_path}\nnew_file{new_file_path}')
                    count2 += 1
    print(f'count1: {count1}')
    print(f'count2: {count2}')




# Sample call

parent_file = "data/kit23_new/"
move_file_to_parentfile(parent_file)

