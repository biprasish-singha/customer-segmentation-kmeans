# Customer Segmentation System

An end-to-end customer segmentation project using **unsupervised machine learning**. The project explores customer purchasing behaviour, spending patterns, loyalty, satisfaction, and digital engagement to identify groups of customers with similar characteristics.

The project includes:
- A Jupyter Notebook for data understanding, cleaning, exploratory data analysis (EDA), feature engineering, model experimentation, and interpretation.
- A saved Scikit-learn preprocessing and clustering pipeline.
- A Streamlit application that accepts customer details and assigns the customer to one of the learned segments.

> **Dataset note:** The dataset used in this project is synthetic. It was created for learning and portfolio demonstration; it does not represent real customers or establish real-world customer behaviour.

## Business problem

Customers have different purchasing habits and engagement patterns. Treating every customer in the same way can make marketing less effective.

The objective of this project is to discover meaningful customer groups from customer data and translate those groups into potential marketing actions.

Questions explored include:
- Are there identifiable groups of high-spending customers?
- Which customers show high discount usage?
- Which customers purchase frequently?
- Which customers may be becoming inactive?
- How do loyalty, satisfaction, and digital engagement differ across customer groups?

## Repository structure

The repository is expected to have the following structure:

```text
customer-segmentation/
│
├── app.py
├── customer_segmentation.csv
├── customer_segmentation.ipynb
├── customer_segmentation_model.pkl
├── requirements.txt
└── README.md
```

### File descriptions

| File | Purpose |
|---|---|
| `customer_segmentation.csv` | Synthetic customer dataset used for analysis and model development |
| `customer_segmentation.ipynb` | Notebook containing data preparation, EDA, feature engineering, clustering experiments, and model export |
| `customer_segmentation_model.pkl` | Serialized model bundle used by the Streamlit application |
| `app.py` | Streamlit interface for entering customer details and predicting a segment |
| `requirements.txt` | Python dependencies required to run the notebook and application |

## Dataset overview

The dataset contains customer-level information, including:

- **Demographics:** age, annual income, and region.
- **Customer relationship:** customer tenure.
- **Purchasing behaviour:** orders in the last 12 months, average order value, total spend, and days since last purchase.
- **Shopping behaviour:** discount order share and return rate.
- **Digital engagement:** web session share, app session share, and email open rate.
- **Customer experience:** satisfaction score and support tickets.
- **Loyalty:** loyalty points.
- **Preferences:** preferred product category, channel, shipping method, and marketing opt-in.

The notebook investigates missing values, duplicate records, inconsistent category labels, invalid values, skewed distributions, and outliers before modelling.

## Methodology

The notebook follows this workflow:

1. **Data loading and understanding** – inspect the dataset structure, data types, and summary statistics.
2. **Data cleaning** – investigate duplicates, missing values, inconsistent categories, currency-formatted values, and invalid observations.
3. **Exploratory data analysis** – examine numerical and categorical distributions and relationships between features.
4. **Feature engineering and transformation** – create log-transformed versions of selected skewed numerical features using `log1p`.
5. **Preprocessing** – standardize numerical features and one-hot encode categorical features.
6. **Clustering experiments** – evaluate K-Means and compare alternative approaches, including PCA-based clustering and Gaussian Mixture Models.
7. **Evaluation and interpretation** – use clustering metrics and inspect customer profiles to understand the resulting groups.
8. **Model export** – save the fitted preprocessing/clustering pipeline and the metadata required by the application.

### Final segmentation approach

The selected deployment approach is **PCA + K-Means with five clusters**, as represented in the notebook's model bundle. The PCA and clustering steps are packaged with preprocessing in a Scikit-learn `Pipeline`.

The cluster labels are translated into descriptive segment names:

| Cluster | Segment name | High-level interpretation |
|---:|---|---|
| 0 | Low-Value Emerging | Relatively newer customers with lower spending and purchasing activity |
| 1 | Discount-Driven & Digital | Customers showing high discount usage and digital engagement |
| 2 | Mainstream Active | Customers with comparatively balanced purchasing and engagement behaviour |
| 3 | At-Risk / Disengaged | Customers showing longer purchase inactivity and weaker engagement |
| 4 | High-Value Loyal | Customers with comparatively high spending, purchase frequency, tenure, and loyalty |

These labels are interpretations of clusters in this synthetic dataset. “At-Risk” is not a confirmed churn prediction, and “High-Value” refers to observed customer value measures rather than proven profitability.

## Streamlit application

The Streamlit application allows a user to:

1. Enter numerical customer details.
2. Select customer preferences from dropdowns.
3. Submit the information to the saved pipeline.
4. View the predicted cluster number and descriptive customer segment.
5. Expand a section to inspect the model input.

The application uses the saved preprocessing and clustering pipeline so that new inputs are transformed consistently with the training workflow.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository-name>.git
cd <your-repository-name>
```

Replace the URL placeholders with your GitHub username and repository name.

### 2. Create and activate a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

Make sure `customer_segmentation_model.pkl` is present in the project root, then run:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

### 5. Recreate the model (optional)

To reproduce the analysis and generate the model artifact:

1. Open `customer_segmentation.ipynb` in Jupyter Notebook or JupyterLab.
2. Ensure `customer_segmentation.csv` is in the same working directory.
3. Run the notebook cells in order.
4. Run the model-export cell to generate `customer_segmentation_model.pkl`.

The Streamlit application loads this file from the current working directory.

## Model evaluation

The notebook evaluates clustering rather than supervised prediction. There is no predefined target label in the dataset.

Metrics and diagnostics explored in the notebook include:
- **Elbow method / inertia:** helps assess how within-cluster sum of squares changes as the number of clusters changes.
- **Silhouette score:** evaluates how close observations are to their own cluster compared with other clusters.
- **Adjusted Rand Index (ARI):** compares clustering assignments from repeated runs.
- **Cluster profiling:** compares customer characteristics across the generated groups.

Clustering metrics should be considered alongside business interpretability. A cluster label is not a ground-truth customer category.

## Limitations

- The dataset is synthetic and may not reflect real customer populations or real commercial relationships.
- Clustering quality depends on feature selection, preprocessing, scaling, and the chosen number of clusters.
- Customer segments describe patterns in the supplied data; they do not prove why customers behave in a particular way.
- The “At-Risk / Disengaged” segment is not a trained churn prediction model.
- Spending is not the same as profitability; margin, acquisition cost, and service cost are not included.
- New customers are assigned to the nearest learned K-Means cluster through the fitted pipeline; this does not guarantee that the customer is a good fit for any cluster.

## Future improvements

- Validate the approach on real, appropriately governed customer data.
- Test alternative feature sets and feature weighting.
- Compare clustering solutions using sampling-based stability analysis.
- Investigate density-based and hierarchical clustering methods.
- Add a dashboard showing segment sizes and customer profiles.
- Develop a separate supervised churn model if reliable historical churn labels become available.
- Add automated input validation and model/version tracking.

## Technologies used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- SciPy
- Scikit-learn
- Joblib
- Jupyter Notebook
- Streamlit

## Disclaimer

This project is for educational and portfolio purposes. The generated segments and suggested marketing actions are exploratory and should not be used to make consequential decisions about real customers without validation.

## Author

**Biprasish Singha**

- GitHub: [biprasish-singha](https://github.com/biprasish-singha)
- LinkedIn: [biprasish-singha-925782426](https://www.linkedin.com/in/biprasish-singha-925782426/)
