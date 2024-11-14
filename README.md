# Predicting Customer Churn in Banking: A Neural Network Approach with Enhanced Class Imbalance Techniques

## Project Overview
This project aims to predict customer churn in the banking sector using a Feedforward Neural Network. The model leverages enhanced class imbalance techniques to improve prediction accuracy, ensuring better customer retention strategies by identifying customers likely to leave the bank. The insights gained from this model can help banks develop targeted interventions to reduce churn.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Feedforward Neural Network** for churn prediction.
- **Class imbalance handling** using techniques such as class weighting to improve prediction accuracy on minority classes.
- **Exploratory Data Analysis (EDA)** to understand key patterns and insights about customer churn.
- **Evaluation metrics**: F1 Score, Precision, Recall, and Accuracy.
- **Visualization** of model performance, including confusion matrix and accuracy trends over epochs.

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn

## Dataset
The dataset consists of customer information from a bank, including features like age, credit score, balance, number of products, and whether or not they churned (left the bank). 
The dataset is available from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) or another source, depending on your dataset source.

### Data Fields
- **CustomerId**: Unique identifier for each customer
- **CreditScore**: Customer's credit score
- **Geography**: Customer's country
- **Gender**: Gender of the customer
- **Age**: Customer's age
- **Tenure**: Number of years with the bank
- **Balance**: Account balance
- **NumOfProducts**: Number of products held by the customer
- **HasCrCard**: Whether the customer has a credit card
- **IsActiveMember**: Whether the customer is an active member
- **Exited**: Target variable (1 if the customer left the bank, 0 otherwise)

## Installation and Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repository-name.git
   cd your-repository-name
   ```

2. **Install Dependencies**:
   Make sure you have Python installed, then install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

   If there's no `requirements.txt`, you can manually install key libraries:
   ```bash
   pip install tensorflow numpy pandas matplotlib seaborn
   ```

3. **Prepare the Data**:
   - Download the dataset and place it in the `data` folder.
   - Run the preprocessing steps as described in the notebook.

## Usage
1. **Train the Model**: Open the `Churn_Prediction_Model.ipynb` notebook and follow the steps to preprocess the data, build the model, and train it.
2. **Evaluate the Model**: After training, evaluate the model performance using the provided evaluation metrics.
3. **Visualize Results**: The notebook includes sections to visualize model performance (accuracy, precision, recall, and confusion matrix).

## Results
- **Baseline Model** achieved approximately 65% accuracy.
- **Enhanced Model** with class imbalance handling achieved approximately 67% accuracy.
- The model was able to identify key patterns in customer churn, though further improvements may be required for production-level accuracy.

See the "Results" section in the notebook for a detailed analysis of model performance.

## Future Improvements
- **Hyperparameter Tuning**: Experiment with different architectures, learning rates, and other hyperparameters.
- **Additional Features**: Incorporate additional data on customer behavior to improve prediction accuracy.
- **Advanced Techniques**: Explore ensemble methods or transfer learning for better performance.

## Contributing
Contributions are welcome! If you'd like to improve the model, add new features, or enhance documentation, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
