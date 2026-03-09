import os
import pydicom
import pandas as pd
import glob

# Get the list of directories
dirs = glob.glob('C:/Users/einspaen/AppData/Local/xnat-dataset/additional-test/*/fill/')

subject_ids = []
date = []

# Loop through each directory
for dir in dirs:
    # Get the list of DICOM files in the directory
    dicom_files = glob.glob(os.path.join(dir, '*.dcm'))

    # Open the first DICOM file
    if dicom_files:
        dicom_file = pydicom.dcmread(dicom_files[0])

        # Get the subject id
        subject_id = dicom_file.PatientID
        subject_ids.append(subject_id)

        # Get the date
        date.append(dicom_file.StudyDate)

# Create a DataFrame
df = pd.DataFrame({'subject_id': subject_ids, 'date': date})

# Write to an Excel file
df.to_excel('test_wA.xlsx', index=False)