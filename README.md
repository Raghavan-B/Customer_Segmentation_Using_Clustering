# Customer Segmentation Using K-Means Clustering

This repository contains the code and resources for a customer segmentation project using the K-Means clustering algorithm. The project aims to provide insights into customer behavior based on their spending patterns and annual income. Furthermore using google gemini pro I've added AI based marketing strategies tailored to the cluster of the customer.
## Project Overview

Customer segmentation is a crucial aspect of marketing, allowing businesses to target specific groups of customers more effectively. This project focuses on segmenting customers into different groups based on their annual income and spending score. The segmentation helps in understanding customer behavior, tailoring marketing strategies, and improving customer satisfaction.

## Dataset

The dataset used for this project is sourced from Kaggle and contains the following features:

- **Customer ID**: Unique identifier for each customer
- **Gender**: Gender of the customer
- **Age**: Age of the customer
- **Annual Income**: Annual income of the customer in thousand dollars
- **Spending Score**: Score assigned by the mall based on customer behavior and spending nature
Dataset Link: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

## Project Structure

The project is structured as follows:

- **data/**: Contains the dataset used for the project.
- **notebooks/**: Jupyter notebooks for data analysis, model training, and evaluation.
- **segmentation.py**: Contains the Streamlit app code for customer segmentation and marketing strategy recommendations.
- **models/**: Saved models and pre-trained weights.
- **README.md**: Project overview and documentation.
  

## Installation
NOTE:  To access the AI marketing strategies recommendation which is a key part of this project, you need to have a [**Google Gemini**](https://ai.google.dev/aistudio) API Key. It should be stored in a **.env** file with a variable name **GOOGLE_API_KEY**

To get started with the project, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/Raghavan-B/PRODIGY_ML_02.git
    cd PRODIGY_ML_02.git
    ```

2. **Create a virtual environment**:

    ```bash
    conda create --name segmentation_env python=3.10
    conda activate segmentation_env
    ```

3. **Install the required packages**:

    ```bash
    pip install -r requirements.txt
    ```

## Data Analysis and Preprocessing

The project begins with data analysis to understand the distribution of customers based on age, gender, annual income, and spending score. Basic data preprocessing steps include handling missing values, encoding categorical variables, and scaling numerical features.

## K-Means Clustering

The core of the project involves applying the K-Means clustering algorithm to segment customers. The steps include:

1. **Feature Selection**: Choosing relevant features (Annual Income and Spending Score) for clustering.
2. **Silhotte Score Method**: Determining the optimal number of clusters using the Silhotte Scores.
3. **Model Training**: Training the K-Means model with the 5 clusters.
4. **Cluster Analysis**: Analyzing the characteristics of each cluster.

## Evaluation

The model's performance is evaluated various scores listed below, which measures how similar an object is to its own cluster compared to other clusters.

These are the results obtained:

1. WCSS Score:  65.56840815571681
2. Silhoutte Score:  0.5546571631111091
3. Davies Boulding Score:  0.5722356162263352
4. Calinski harabasz score:  248.64932001536357

### Customer Types
Based on the specified clusters found after performing K-Means clustering, here are the customer types we can infer:

**Cluster 1 (Annual Income: 40-80, Spending Score: 40-60)**

Customer Type: Mid-Income, Mid-Spenders
These customers have a moderate income and balanced spending behavior. They are likely to be cautious but willing to spend on value-for-money products.

**Cluster 2 (Annual Income: 60-140, Spending Score: 0-40)**

Customer Type: High-Income, Low Spenders
These are high-income individuals who spend conservatively. They might be saving for long-term goals or prefer investing over spending on discretionary items.

**Cluster 3 (Annual Income: 0-40, Spending Score: 0-40)**

Customer Type: Low-Income, Low Spenders
These customers have low income and spending capacity. They are budget-conscious and look for essential or low-cost items.

**Cluster 4 (Annual Income: 20-40, Spending Score: 60-100)**

Customer Type: Low-Income, High Spenders
These individuals, despite having lower incomes, tend to spend a lot. They might prioritize lifestyle or aspirational purchases over savings.

**Cluster 5 (Annual Income: 70-140, Spending Score: 60-100)**

Customer Type: High-Income, High Spenders
These are affluent customers who spend generously. They are likely to be attracted to luxury goods, premium services, and exclusive offers.

## Streamlit App

The project includes a Streamlit app that allows users to input customer details and get the corresponding customer segment. Additionally, the app provides marketing strategy recommendations based on the customer segment using the Google Gemini Pro model.

To run the app:

```bash
streamlit run segmentation.py
```

## Usage

The app collects customer details such as age, gender, annual income, and spending score. It then uses the trained K-Means model to assign the customer to one of the predefined clusters and provides tailored marketing strategies based on the cluster.

## Future Work

Future enhancements to the project could include:

- Incorporating additional features for clustering, such as purchase history and customer preferences.
- Exploring other clustering algorithms and comparing their performance.
- Integrating real-time data for dynamic customer segmentation.

## Acknowledgements

- The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com).
---

