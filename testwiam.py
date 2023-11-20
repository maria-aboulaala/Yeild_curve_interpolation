import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize, curve_fit
from data_fetch import DataFetcher
import pandas as pd

class InterestRateModel:
    def _init_(self, selected_rate, start_date, end_date):
        self.data_fetcher = DataFetcher()
        self.data = self.data_fetcher.get_interest_rate_data(selected_rate, start_date, end_date)
        self.known_time = self.data.index
        self.known_rates = self.data['Close']
        self.interpolation_time = pd.date_range(start=start_date, end=end_date, freq='D')

    def interpolate_or_model(self, model_type, known_time, known_rates, interpolation_time, params=None):
        if model_type == 'Linear':
            return self.interpolate_linear(known_time, known_rates, interpolation_time)
        elif model_type == 'Polynomial':
            return self.interpolate_polynomial(known_time, known_rates, interpolation_time, params=params)
        elif model_type == 'Cubic':
            return self.interpolate_cubic(known_time, known_rates, interpolation_time)
        elif model_type == 'Exponential':
            return self.interpolate_exponential(known_time, known_rates, interpolation_time)
        elif model_type == 'Spline Linear':
            return self.spline_linear(known_time, known_rates, interpolation_time)
        elif model_type == 'Spline Cubic':
            return self.spline_cubic(known_time, known_rates, interpolation_time)
        elif model_type == 'Nelson-Siegel':
            return self.model_nelson_siegel(known_time, known_rates, params=params)
        elif model_type == 'Vasicek':
            return self.model_vasicek(known_time, known_rates, params=params)
        else:
            raise ValueError("Invalid model type.")

    # def interpolate_linear(self, known_time, known_rates, interpolation_time):
    #     f = interp1d(known_time, known_rates, kind='linear', fill_value='extrapolate')
    #     curves_yield = f(interpolation_time)
    #     return curves_yield

    # def interpolate_polynomial(self, known_time, known_rates, interpolation_time, degree=2):
    #     coefficients = np.polyfit(known_time, known_rates, degree)
    #     curves_yield = np.polyval(coefficients, interpolation_time)
    #     return curves_yield

    def interpolate_cubic(self, known_time, known_rates, interpolation_time):
        f = interp1d(known_time, known_rates, kind='cubic', fill_value='extrapolate')
        curves_yield = f(interpolation_time)
        return curves_yield

    def interpolate_exponential(self, known_time, known_rates, interpolation_time):
        p = np.polyfit(known_time, np.log(known_rates), 1)
        curves_yield = np.exp(np.polyval(p, interpolation_time))
        return curves_yield
    
    def spline_linear(self, known_time, known_rates, interpolation_time):
        f = interp1d(known_time, known_rates, kind='linear', fill_value='extrapolate')
        curves_yield = f(interpolation_time)
        return curves_yield

    def spline_cubic(self, known_time, known_rates, interpolation_time):
        f = CubicSpline(known_time, known_rates)
        curves_yield = f(interpolation_time)
        return curves_yield

    def model_nelson_siegel(self, known_time, known_rates, params=None):
        if params is None:
            params = self.fit_nelson_siegel_parameters(known_time, known_rates)
        ytm = params[0] + params[1] * (1 - np.exp(-known_time / params[2])) / (known_time / params[2]) + params[1] * np.exp(-known_time / params[2])
        return ytm

    def model_vasicek(self, known_time, known_rates, params=None):
        if params is None:
            params = self.fit_vasicek_parameters(known_time, known_rates)
        random_noise = np.random.normal(0, 1, len(known_time))
        ytm = params[0] + params[1] * (params[0] - known_time) * (1 - np.exp(-params[2] * known_time)) / params[2] + params[2] / params[1] * (
                    (1 - np.exp(-params[2] * known_time)) * random_noise - known_time * (1 - np.exp(-params[2] * known_time)))
        return ytm

    def fit_vasicek_parameters(self, known_time, known_rates):
        # Objective function to minimize (sum of squared errors)
        def objective_function(params, known_time, known_rates):
            return np.sum((known_rates - self.model_vasicek(known_time, *params))**2)

        # Initial guess for parameters and constraints
        initial_params = [0.01, 0.01, 0.01]
        constraints = ({'type': 'positive', 'fun': lambda x: x[2]})
        
        # Minimize the objective function using scipy.optimize.minimize
        optimized_params = minimize(objective_function, initial_params, args=(known_time, known_rates), constraints=constraints).x

        return optimized_params

    def fit_nelson_siegel_parameters(self, known_time, known_rates):
        # Objective function to minimize (sum of squared errors)
        def objective_function(params, known_time, known_rates):
            return np.sum((known_rates - self.model_nelson_siegel(known_time, *params))**2)

        # Initial guess for parameters
        initial_params = [0.01, 0.01, 1.0]
        
        # Minimize the objective function using scipy.optimize.curve_fit
        optimized_params, _ = curve_fit(objective_function, known_time, known_rates, p0=initial_params)

        return optimized_params
