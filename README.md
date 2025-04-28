📈 Linear Regression from Scratch with Python

In this project, we manually implemented a Linear Regression model using Python and compared it against Scikit-learn’s LinearRegression model.
🛠 Files Structure

    LinearRegression.py

        A custom LinearRegression class built from scratch.

        fit() method trains the model.

        predict() method makes predictions.

    LogisticModel.py

        Generates random data.

        Trains the custom model.

        Compares the custom model with Scikit-learn's model.

        Calculates and prints error metrics.

    main.py

        (Currently empty or supports running the LogisticModel.py.)

📚 Project Flow

    Data Generation

        We create 100 random samples between 0 and 2:

    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    y is generated based on a linear relation with added noise.

Training the Model

    Using fit(X_train, y_train), the custom model is trained.

    If the data is not a numpy.ndarray, it is converted.

    For each feature (column):

        Calculates SxxSxx​ (sum of squared differences for X).

        Calculates SxySxy​ (sum of cross products between X and y).

        Updates the model's intercept and coefficients accordingly.

Making Predictions

    The custom model predicts values using:

        y_head = np.dot(self.coeff_, X_test.T) + self.intercept_

    Comparing with Scikit-learn

        The Scikit-learn LinearRegression model is trained on the same data.

        We calculate the Mean Squared Error (MSE) for both models.

        Plotting the predicted vs real values using matplotlib.

📊 Outputs and Metrics

    Custom Model:

        Intercept value

        Coefficients

        Deviation from the real sample

        Custom calculated Mean Squared Error (MSE)

    Scikit-learn Model:

        Automatically fitted intercept and coefficients

        Scikit-learn calculated MSE

✅ Results show that our custom model performs similarly to Scikit-learn's implementation!
📈 Example Plot

After training, the following plot is displayed:

    Red line: Model's predicted linear fit.

    Dots: The actual randomly generated data points.

⚙️ Requirements

    Python 3.x

    Libraries:

        numpy

        pandas

        matplotlib

        scikit-learn

Install the requirements using:

pip install numpy pandas matplotlib scikit-learn

🚀 How to Run

python LogisticModel.py

Would you also like me to generate a little fancier version with badges (like Python version, license, etc.)? 🚀
(If you want, I can add that too!)
