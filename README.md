# Regression with Artificial Neural Networks Taxi Ride Price 

This project applies Artificial Neural Networks (ANNs) to solve real-world regression problems. The models are designed to predict:

Taxi Ride Price (based on ride details)

The assignment focuses on data exploration, preprocessing, ANN model building, optimization, and evaluation to understand how neural networks can be applied to regression tasks.

# 📂 Datasets

The datasets used in this project are available on Kaggle:

Taxi Price Prediction

# 🚀 Project Workflow
1. Data Understanding

Loaded datasets and explored first few rows

Identified input features and target variables (Salary, Price, Age)

Checked data types (numerical vs categorical)

Handled missing values and duplicates

2. Exploratory Data Analysis (EDA)

Distribution plots of target variables

Feature vs target relationships

Category-wise comparisons (e.g., Gender, Job Role, Passenger Count, Sex of Crab)

Correlation heatmaps for numerical features

3. Data Preprocessing

Encoded categorical variables

Scaled numerical features for ANN input

Split data into training and testing sets

4. Model Building (Baseline ANN)

Constructed a baseline ANN with 1 hidden layer

Trained the model on training data

Evaluated using MSE, MAE, and R² score

5. Model Optimization

Experimented with:

More hidden layers & neurons

Dropout for regularization

Different learning rates, batch sizes, and epochs

Applied EarlyStopping to prevent overfitting

6. Model Evaluation

Compared training vs testing performance

Plotted loss curves (training vs validation)

Visualized Actual vs Predicted values

Interpreted most influential features

# 📊 Results & Insights

ANN models successfully Taxi Price with improved accuracy after optimization.

Scaling and categorical encoding were crucial for stable training.

EarlyStopping and Dropout helped in controlling overfitting.

Feature importance analysis revealed the strongest predictors for each dataset.

# 📝 Deliverables

Jupyter Notebook: Includes EDA, preprocessing, ANN models, optimization, and evaluation

Short write-up: Summarizes EDA insights, model performance comparison, and challenges solved

# 🔑 Key Learnings

Preprocessing mixed datasets (numerical + categorical)

Building and tuning ANN regression models

Evaluating models using MSE, MAE, and R²

Gained deeper understanding of ANN applications in real-world regression

# 🛠️ Tech Stack

Python

TensorFlow / Keras (for ANN models)

Pandas, NumPy (data handling)

Matplotlib, Seaborn (visualization)

Scikit-learn (metrics & preprocessing)

# 📄 Author

👤 Sowjanya U

Data Science & Machine Learning Enthusiast

📧 Contact: usowjanyakiran@gmail.com

🌐 GitHub: https://github.com/SowjanyaKiran/
