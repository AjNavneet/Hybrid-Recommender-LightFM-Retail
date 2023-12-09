# Hybrid Recommender System using LightFM

### Business Objective

There are two main methods for making these suggestions: content-based and collaborative filtering. Collaborative filtering finds similarities between users to make recommendations, while content-based filtering personalizes content for each user based on their previous actions and feedback. 

However, these methods struggle when there's not enough data. To address this, we'll explore a Hybrid Recommendation System, which combines both approaches.

---

### Data Description

The dataset used in this project contains transactional data for a UK-based online retail company that sells unique gifts for various occasions.

---

### Aim

Our goal is to build a Hybrid Recommendation system using different loss functions with the LightFM library.

---

### Tech Stack

- Language: `Python`
- Libraries: `pandas`, `numpy`, `scipy`, `lightfm`

---

## Approach

1. **Import required libraries**
2. **Read and merge the data**
3. **Prepare the data**
4. **Split the data into training and testing sets**
5. **Build models**
   - Model with WARP loss function
   - Model with logistic loss function
   - Model with BPR loss function
6. **Combine data for the final model**
7. **Generate recommendations**

---

## Modular Code

1. **input**: Contains the data we'll use for analysis, such as `data.xlsx`.
2. **src**: This folder holds all the code for our project, organized in a modular manner. It includes:
   - **ML_pipeline**
   - **engine.py**

   The `ML_pipeline` folder contains functions organized in different Python files, which are called from the `engine.py` file. There's also a `config.ini` file in the input folder, storing variables used in `engine.py`.

3. **output**: Contains our final models saved in pickle format.
4. **lib**: This is a reference folder that includes the original IPython notebook and reference pdfs for explanation.
5. **requirements.txt**: Lists all the required libraries with their respective versions. Install these libraries using the command `pip install -r requirements.txt`.
6. Instructions for running the code are in the `readme.md` file.

---

## Getting Started

### Install all the requirements

- pip install -r requirements.txt

#### Run the engine.py file to execute the code

---


### Note

In case you face issues while installing the  'lightfm' package; Try the following two methods:

1. In your VS code, perform the following executions on your terminal window

	- Upgrade your pip with: python -m pip install --upgrade pip

	- Upgrade your wheel with: pip install --upgrade wheel

	- Upgrade your setuptools with: pip install --upgrade setuptools

	- close the terminal

	- Try installing the pacakage again.


2. In case you face; error: Microsoft Visual C++
Download and install  
	- https://visualstudio.microsoft.com/visual-cpp-build-tools/

---

