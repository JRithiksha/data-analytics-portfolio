Electronic Retail Sales Analysis — ML Project

End-to-end machine learning pipeline on an electronics retail sales dataset — covering data cleaning, exploratory analysis, sales prediction, and customer segmentation.

🎯 Project Overview

This project analyzes electronic retail sales transactions to:

Predict transaction sales using a Linear Regression model.
Segment customers into Regular, VIP, and Potential groups using K-Means clustering, based on spending behavior.

🔧 Tech Stack
Language: Python 3.10
Data handling: pandas, numpy
Visualization: matplotlib, seaborn
Modeling: scikit-learn (LinearRegression, KMeans, StandardScaler, LabelEncoder)
Persistence: pickle

📊 Workflow
1. EDA & Preprocessing
Loaded raw sales data and inspected shape, types, and missing values
Handled missing values with forward-fill
Extracted Month and Day from the transaction date
Created the Sales target column (Price × Quantity) where not already present
Visualized sales trends over time, top-selling products, and a correlation heatmap
Encoded categorical fields and exported a cleaned dataset
2. Sales Prediction (Regression)
Features: unit price, quantity, month, product type, gender
Target: total price
One-hot encoded categorical features and scaled numeric features
Trained a Linear Regression model on an 80/20 train-test split
Evaluation: MAE and R² Score
3. Customer Segmentation (Clustering)
Aggregated transactions per customer into total spending, total quantity, and purchase frequency
Scaled features and applied K-Means (k=3)
Labeled clusters as Regular, VIP, and Potential based on spending/frequency patterns
Visualized segments with a scatter plot

🔮 Future Improvements
Compare Linear Regression against tree-based models (Random Forest, XGBoost)
Validate cluster count with Elbow/Silhouette analysis
Add cross-validation for more robust regression evaluation
Wrap trained models in a simple Flask/FastAPI service for live predictions







