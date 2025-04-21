# File-processing
## Interpretation of folders in code
- [file_conversion](file_conversion)
- [files_split](files_split)
- [Dataset](Dataset)
## Introduction
### file_conversion
In `file_conversion`, there are two main parts:<br>
* the first part [delete_specified_folder.py](delete_specified_folder.py) moves the **vaguely matched** file in `data/kit23/cass_00001/instance` to the `data/kit23/cass_00001/` directory and **changes the name**<br>
* The second code [file_rename_and_location_conversion.py](file_rename_and_location_conversion.py)**deletes** the **instances folder** and **segmentation files** in the `data/kit23/cass_00001/` directory
### files_split
In `files_split`, subdirectories **case_00000** to **case_00500** on the `data/kit23` folder are assigned to two new folders, **train** and **test**, according to **8:2**
