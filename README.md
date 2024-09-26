# Project 6 - Exploratory Data Analysis of the Titanic Dataset

This project uses a combination of Python and Markdown to create an initial data story of the sinking of the Titanic. Class, Sex, and Age will all be explored in relation to survival.

The project includes a project virtual environment with popular libraries for data analytics including pandas, matplotlib, and seaborn, and uses a common process for exploratory data analysis projects.

## Create and Activate Project Virtual Environment

py -m venv .venv  
.venv\Scripts\Activate

## Add and Commit Changes to Github

git add .  
git commit -m  
git push -u origin main

## Add external dependencies

### Install packages separately from terminal

py -m pip install jupyterlab
py -m pip install numpy
py -m pip install pandas
py -m pip install matplotlib
py -m pip install seaborn
py -m pip install scipy

## Logging

import logging

--Configure logging to write to a file, appending new logs to the existing file
logging.basicConfig(filename='log.txt', level=logging.DEBUG, filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Program started") # add this at the beginning of the main method
logging.info("Program ended") # add this at the end of the main method

## Create files

1.  .gitignore
2.  README.md
3.  requirements.txt

### Install and Set up Jupyter in VS Code

1.  Install the Jupyter Extension: If not already installed, add the Jupyter extension to VS Code. This extension provides rich support for working with Jupyter notebooks.
2.  Open the Project Folder: Open your root project repository folder in VS Code. (Usually in your Documents folder.)
3.  Select the Python Interpreter: From the command palette (Ctrl+Shift+P), select "Python: Select Interpreter" and choose the interpreter from your virtual environment.
    Then create, open, and start a new notebook in your root project repository folder:
4.  Create the Notebook: In the VS Code Explorer, create a new file i.e., yourname_eda.ipynb. Ensure it has a .ipynb extension.
5.  Verify your new notebook is open for editing. If needed, view the project files in VS Code Explorer and double-click the notebook file to open it for editing.
6.  Add a Markdown cell at the top of your notebook with the introduction (include the title, author, date and the purpose of the project).

### Import Dependencies after introduction

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import requests
import jupyterlab
import pyarrow

### Dataset Description

The titanic dataset is one that is used often for exploratory data analysis. The Titanic dataset in Seaborn is a collection of passenger and crew information from the RMS Titanic, which sank in 1912.

https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv

### Steps for EDA

Perform exploratory data analysis (EDA) using pandas and other tools as needed.

1.  Data Acquisition
2.  Initial Data Inspection
3.  Initial Descriptive Statistics
4.  Initial Data Distribution for Numerical Columns
5.  Initial Data Distribution for Categorical Columns
6.  Initial Data Transformation and Feature Engineering
7.  Initial Visualizations
8.  Initial Storytelling and Presentation
