# Customer Segmentation using K-Means Clustering

An unsupervised machine learning project that groups retail customers according to their annual income and spending score. The resulting segments can support targeted marketing, customer retention, and purchasing-pattern analysis.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-K--Means-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-28A745?style=for-the-badge)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Objective](#project-objective)
- [Technologies Used](#technologies-used)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Results](#results)
- [Customer Segments](#customer-segments)
- [Learning Outcomes](#learning-outcomes)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [Internship Details](#internship-details)

---

## 🎯 Overview

This project is part of the **SkillCraft Technology Machine Learning Internship** and uses K-Means clustering to identify groups of customers with similar income and spending behavior.

**Workflow:**
- Load and inspect the Mall Customers dataset
- Select annual income and spending score as clustering features
- Use the Elbow Method to estimate the number of clusters
- Train a K-Means model with five clusters
- Visualize clusters and their centroids
- Export the clustered customer data to CSV

---

## 🎯 Project Objective

Segment customers into meaningful groups using **Annual Income (k$)** and **Spending Score (1-100)**, then interpret those groups for business decision-making.

---

## 🛠 Technologies Used

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- K-Means clustering

---

## 🔬 Methodology

1. Load the customer data from `data/Mall_Customers.csv`.
2. Select the income and spending-score columns.
3. Calculate WCSS for values of `k` from 1 to 10.
4. Apply the Elbow Method to select the cluster count.
5. Fit K-Means with `random_state=42` and `n_init=10`.
6. Label clusters using their centroid income and spending levels.
7. Save a combined Elbow Method and segmentation plot.
8. Export the results with cluster labels to a CSV file.

---

## 📊 Dataset

The project uses the **Mall Customers** dataset. The model uses these two features:

| Feature | Description |
|---------|-------------|
| `Annual Income (k$)` | Customer annual income in thousands of dollars |
| `Spending Score (1-100)` | Customer spending score |

The dataset contains 200 customer records. If the local CSV is unavailable, the script creates a reproducible sample dataset so it can still be executed.

---

## 📁 Project Structure

```text
SCT_ML_2/
├── README.md
├── kmeans_clustering.py
├── data/
│   └── Mall_Customers.csv
├── customer_segments_plot.png
└── segmented_customers.csv
```

---

## 📦 Requirements

Python 3.9 or later is recommended. Install the dependencies with:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## ⚙️ Setup

### 1. Open the project folder

```bash
cd "SCT_ML 2"
```

### 2. Create and activate a virtual environment (optional)

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## ▶️ How to Run

```bash
python kmeans_clustering.py
```

The script:
- prints dataset and cluster information
- saves `customer_segments_plot.png`
- saves clustered data to `segmented_customers.csv`

---

## 📈 Results

The model divides the customers into **five clusters** and produces a plot containing both the Elbow Method curve and the customer segments. Each point represents a customer, while the marked centroids represent the center of each cluster.

---

## 👥 Customer Segments

- **Target:** High income, high spending customers
- **Careful:** High income, low spending customers
- **Standard:** Moderate income and moderate spending customers
- **Careless:** Low income, high spending customers
- **Sensible:** Low income, low spending customers

---

## 🎓 Learning Outcomes

- Understanding unsupervised learning and K-Means clustering
- Selecting the number of clusters with WCSS and the Elbow Method
- Applying feature selection to a distance-based algorithm
- Interpreting cluster centroids
- Visualizing customer groups and exporting predictions

---

## 🚀 Future Improvements

- Scale features before clustering and compare the results
- Evaluate the solution using silhouette score
- Add demographic features such as age and gender
- Build an interactive dashboard for segment exploration

---

## 👨‍💻 Author

**Devarakonda Vignesh Varsha**

---

## 📌 Internship Details

| Detail | Information |
|--------|-------------|
| **Internship** | SkillCraft Technology |
| **Domain** | Machine Learning |
| **Task** | Task 02 - Customer Segmentation using K-Means Clustering |
| **Clusters** | 5 |

---

*Built with ❤️ during SkillCraft Technology ML Internship*
