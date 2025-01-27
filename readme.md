# Hybrid Recommender System using LightFM

## Business Objective

Recommendation systems are critical for improving user engagement by suggesting relevant products or services. Two popular approaches include:

- **Collaborative Filtering**: Identifies similarities between users to recommend items.
- **Content-Based Filtering**: Personalizes recommendations for users based on their past actions and preferences.

However, these methods face challenges with sparse or insufficient data. To address these limitations, we develop a **Hybrid Recommendation System** that combines both approaches, leveraging the **LightFM** library.

---

## Data Description

The dataset contains transactional data from a UK-based online retail company specializing in unique gifts for various occasions. Key features include:

- **User IDs**: Identifiers for customers.
- **Item IDs**: Identifiers for products.
- **Transactions**: User-item interactions.
- **Ratings/Feedback**: Implicit or explicit feedback data.

---

## Aim

To build a Hybrid Recommendation System using different loss functions provided by the **LightFM** library, including:

- **WARP (Weighted Approximate-Rank Pairwise)**
- **Logistic Loss**
- **BPR (Bayesian Personalized Ranking)**

---

## Tech Stack

- **Programming Language**: [Python](https://www.python.org/)
- **Libraries**:
  - [`pandas`](https://pandas.pydata.org/) for data manipulation.
  - [`numpy`](https://numpy.org/) for numerical operations.
  - [`scipy`](https://scipy.org/) for scientific computing.
  - [`lightfm`](https://making.lyst.com/lightfm/) for building hybrid recommendation systems.

---

## Approach

### 1. Import Required Libraries
- Load essential Python libraries for data manipulation and model building.

### 2. Read and Merge Data
- Import the dataset and combine user and item features into a single dataset for analysis.

### 3. Prepare Data
- Clean and preprocess the data for input into the LightFM model.

### 4. Split Data
- Divide the dataset into training and testing sets to evaluate model performance.

### 5. Build Models
- Train LightFM models using:
  - **WARP Loss Function**: Optimized for ranking.
  - **Logistic Loss Function**: Suitable for predicting probabilities.
  - **BPR Loss Function**: Designed for implicit feedback data.

### 6. Combine Data
- Integrate content-based and collaborative filtering data to train the hybrid model.

### 7. Generate Recommendations
- Use the trained hybrid model to recommend items for users based on their interactions and preferences.

---

## Project Structure

```plaintext
.
├── input/                                # Contains input data files (e.g., `data.xlsx`).
├── src/                                  # Source code folder.
│   ├── engine.py                         # Main script to execute the pipeline.
│   ├── ML_pipeline/                      # Modular Python functions for preprocessing and modeling.
│       ├── data_preparation.py           # Functions for data preprocessing.
│       ├── model_training.py             # Functions for training LightFM models.
│       ├── recommendation.py             # Functions to generate recommendations.
├── output/                               # Stores saved models and results.
├── lib/                                  # Reference materials and Jupyter notebooks.
├── requirements.txt                      # Lists dependencies and versions.
└── README.md                             # Project documentation.
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_folder>
```

### 2. Install Dependencies

Install all required Python libraries using:

```bash
pip install -r requirements.txt
```

### 3. Run the Project

Execute the pipeline by running the `engine.py` file:

```bash
python src/engine.py
```

### 4. Explore Results

- Check the `output/` folder for saved models and generated recommendations.
- Review reference notebooks in the `lib/` folder for detailed explanations.

---

## Troubleshooting

If you encounter issues installing the `lightfm` package, try the following steps:

### Method 1: Upgrade Essential Tools

Run the following commands in your terminal:

```bash
python -m pip install --upgrade pip
pip install --upgrade wheel
pip install --upgrade setuptools
```

Close the terminal and retry installing `lightfm`.

### Method 2: Install Microsoft Visual C++ Build Tools

Download and install the required tools from:
[Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

Retry installing `lightfm` after the installation.

---

## Results

- **Hybrid Model Performance**:
  - Combined the strengths of collaborative and content-based filtering.
  - Achieved personalized and accurate recommendations.
- **Generated Recommendations**:
  - Delivered tailored item suggestions based on user preferences and item features.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch:

```bash
git checkout -b feature-name
```

3. Commit your changes:

```bash
git commit -m "Add feature"
```

4. Push your branch:

```bash
git push origin feature-name
```

5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For any questions or suggestions, please reach out to:

- **Name**: Abhinav Navneet
- **Email**: mailme.AbhinavN@gmail.com
- **GitHub**: [AjNavneet](https://github.com/AjNavneet)

---

## Acknowledgments

Special thanks to:

- [LightFM](https://making.lyst.com/lightfm/) for providing a robust library for hybrid recommendations.
- The Python open-source community for excellent tools and resources.

---


