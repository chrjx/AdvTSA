import numpy as np

class PersistenceModel:
    def predict(self, input_data):
        """
        Persistence forecast:
        p̂_{t+h} = p_t
        """
        if input_data is None or len(input_data) == 0:
            raise ValueError("Input data cannot be empty.")
        return float(input_data[-1])
    
class SimpleARIMA:
    def __init__(self, order=(1, 0, 0)):
        self.order = order
        self.model = None

    def fit(self, input_data):
        from statsmodels.tsa.arima.model import ARIMA
        if input_data is None or len(input_data) == 0:
            raise ValueError("Input data cannot be empty.")
        self.model = ARIMA(input_data, order=self.order).fit()

    def predict(self, steps=1):
        """
        Forecast future values using the fitted ARIMA model.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        forecast = self.model.forecast(steps=steps)
        return forecast.tolist()

class AdaptiveAR:
    def __init__(self, order=1, forgetting_factor=0.99, delta=1000.0):
        """
        Adaptive AR model using Recursive Least Squares (RLS)

        Parameters:
        order (int): AR order
        forgetting_factor (float): lambda, controls adaptivity (0.95–0.999)
        delta (float): initial covariance scaling
        """
        self.order = order
        self.lambda_ = forgetting_factor
        self.theta = np.zeros(order)              # AR parameters
        self.P = delta * np.eye(order)             # covariance matrix
        self.initialized = False

    def update(self, y_t, y_past):
        """
        Update model with one new observation.

        y_t: current value
        y_past: array of past values [y_{t-1}, ..., y_{t-p}]
        """
        phi = np.array(y_past).reshape(-1, 1)

        if len(phi) != self.order:
            raise ValueError("Incorrect number of past observations")

        # Prediction error
        y_hat = float(self.theta @ phi.flatten())
        error = y_t - y_hat

        # RLS gain
        K = self.P @ phi / (self.lambda_ + phi.T @ self.P @ phi)

        # Parameter update
        self.theta = self.theta + (K.flatten() * error)

        # Covariance update
        self.P = (self.P - K @ phi.T @ self.P) / self.lambda_

        self.initialized = True
        return y_hat, error

    def predict(self, y_past):
        """
        One-step-ahead prediction
        """
        if not self.initialized:
            raise ValueError("Model has not been updated yet.")
        return float(self.theta @ np.array(y_past))

model = AdaptiveAR(order=1, forgetting_factor=0.99)

predictions = []
errors = []

p = 10 + np.sin(np.linspace(0, 50, 100)) + np.random.normal(0, 0.5, 100)
for t in range(1, len(p)):
    y_past = [p[t-1]]
    y_hat, err = model.update(p[t], y_past)
    predictions.append(y_hat)
    errors.append(err)

theta_trace = []

for t in range(1, len(p)):
    y_past = [p[t-1]]
    y_hat, err = model.update(p[t], y_past)
    theta_trace.append(model.theta.copy())


