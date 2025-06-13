# Customer-Segmentation-Using-K-Means

## Overview

This project applies unsupervised machine learning to segment customers into meaningful groups. It leverages the K-Means clustering algorithm based on behavioral data such as age, income, visit frequency, and spending score.

## Objective

To identify distinct customer segments for personalized marketing strategies using clustering techniques.

## Dataset

The dataset includes the following features for 200 synthetic customers:

* Customer ID
* Age
* Annual Income (k\$)
* Visit Frequency
* Spending Score

## Steps Performed

1. **Data Generation**: Synthetic customer data is created using NumPy.
2. **Data Preprocessing**: Features are scaled using `StandardScaler`.
3. **Clustering**: K-Means is used to cluster customers.
4. **Elbow Method**: Optimal number of clusters is determined using the Elbow plot.
5. **Visualization**:

   * 2D visualization using PCA
   * 3D visualization using PCA

## Output Files

* `customer_data.csv`: The dataset with assigned cluster labels
* `elbow_plot.png`: Visual to identify optimal clusters
* `customer_clusters_2D.png`: 2D plot of customer segments
* `customer_clusters_3D.png`: 3D plot of customer segments

## Requirements

* Python 3
* pandas, numpy, matplotlib, seaborn, scikit-learn

## How to Run

1. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. Run the script:

```bash
python customer_segmentation.py
```

## Conclusion

This segmentation helps businesses personalize engagement, enhance marketing ROI, and understand customer diversity.
