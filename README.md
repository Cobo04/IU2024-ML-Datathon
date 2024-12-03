# IU2024-ML-Datathon
Repository for all code relating to the machine learning aspect of the 2024 IU Datathon and Machine Learning Competition on Antisemitism. All code created by Christine Chen, Mikey Mauch, Cohen Schulz, and Matthew Widjaja.

## Important Information
- All code is found in `analysis_main.py`
- A specific version of `numpy` may be required depending on your system specifications. If you receive this error, please run the following command in terminal
 > `pip install --force-reinstall -v "numpy==1.25.1`

## Installation Instructions
- Ensure that all necessary dependencies have been installed. These can be found under the `Necessary Dependencies` header in `analysis_main.py`
- At the bottom of the file, there is a `path` variable under the `Main` header that signifies the path to the dataset. Ensure that this is correctly set up before continuing
- At the bottom of the file, there is a `labels` variable under the `Main` header that signifies the labels per column header of the csv file, following the format: `[text, sentiment]`. If the titles of the headers in the CSV file are different, then this will need to be changed. Otherwise, do not change the list

## Running Instructions
- After following all setup procedures correctly, simply run the `analysis_main.py` and the f1 score along with the confusion matrix will be output.
