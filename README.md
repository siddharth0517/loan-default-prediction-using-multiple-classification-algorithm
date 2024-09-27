
# Loan Default Prediction Project

This project aims to predict loan defaults using various machine learning models. The goal is to build and compare models that can classify whether a borrower will default on their loan, based on historical data.

## Project Structure

1. **Importing Libraries**: 
   The project uses libraries such as `numpy`, `pandas`, `matplotlib`, and `seaborn` for data manipulation and visualization, and `scikit-learn` for building machine learning models.

2. **Dataset and Exploratory Data Analysis (EDA)**:
   - The dataset is imported and cleaned.
   - Initial analysis and visualizations are conducted to understand key features and correlations.

3. **Data Preprocessing**:
   - **Encoding Categorical Data**: Categorical variables are converted into numerical formats using appropriate encoding techniques.
   - **Feature Scaling**: Ensuring that the features are on the same scale for better model performance.
   - **Train-Test Split**: The dataset is split into training and testing sets for model evaluation.

4. **Modeling**:
   Various models are trained and evaluated on the dataset:
   - **Logistic Regression**
   - **Decision Tree**
   - **Random Forest**
   - **K-Nearest Neighbors (KNN)**
   - **Support Vector Machine (SVM)**
   - **Naive Bayes**

5. **Model Evaluation**:
   Each model is evaluated based on performance metrics such as accuracy, precision, recall, and F1-score.

6. **Deployment**:
   The project includes a simple deployment using Streamlit, a popular framework for creating web apps for machine learning projects.

## Usage

1. Clone this repository.
2. Install required dependencies: `pip install -r requirements.txt`.
3. Run the Jupyter notebook to train the models and evaluate them.
4. Deploy the model using Streamlit by running `streamlit run app.py`.

## Conclusion

This project demonstrates a comparison of several machine learning models for predicting loan defaults. The results provide insights into the strengths and weaknesses of different approaches, and the final deployment allows users to interact with the model predictions in a user-friendly manner.
