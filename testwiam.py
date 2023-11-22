import streamlit as st
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

file_path ='C:/Users/AHMED/Desktop/projet asri 2/output.xlsx'

# Read the Excel file into a DataFrame
df = pd.read_excel(file_path)

# Extract 'maturities' and 'yields' columns as lists
maturities = df['maturities'].tolist()
yields = df['yields'].tolist()

# Print the lists (optional)
print("Maturities:", maturities)
print("Yields:", yields)



def linear_interpolation(maturities, yields, desired_maturity):
    f = interpolate.interp1d(maturities, yields, kind='linear')
    return f(desired_maturity)

def cubic_interpolation(maturities, yields, desired_maturity):
    f = interpolate.interp1d(maturities, yields, kind='cubic')
    return f(desired_maturity)

def plot_linear_curve(maturities, yields, desired_maturity):
    plt.figure(figsize=(8, 6))
    
    # Plotting the original yield curve data
    plt.plot(maturities, yields, 'o', label='Original Data Points')

    # Generating more points for smoother curve using linspace for plotting
    maturities_smooth = np.linspace(min(maturities), max(maturities), 1000)
    yield_linear = linear_interpolation(maturities, yields, desired_maturity)
    
    # Plotting the linear interpolated curve
    plt.plot(maturities_smooth, linear_interpolation(maturities, yields, maturities_smooth), label='Linear Interpolation', linestyle='--')

    # Plotting the cubic interpolated curve

    # Plotting the interpolated points
    plt.plot(desired_maturity, yield_linear, 'rx', markersize=10, label=f'Interpolated Point (Linear): {yield_linear:.2f}%')

    # Adding labels and title
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Yield (%)')
    plt.title('Yield Curve Interpolation')

    # Show legend
    plt.legend()


    
    
    
    
def plot_cubic_curve(maturities, yields, desired_maturity):
    plt.figure(figsize=(8, 6))
    
    # Plotting the original yield curve data
    plt.plot(maturities, yields, 'o', label='Original Data Points')

    # Generating more points for smoother curve using linspace for plotting
    maturities_smooth = np.linspace(min(maturities), max(maturities), 1000)
    yield_cubic = cubic_interpolation(maturities, yields, desired_maturity)
    
    # Plotting the linear interpolated curve

    # Plotting the cubic interpolated curve
    plt.plot(maturities_smooth, cubic_interpolation(maturities, yields, maturities_smooth), label='Cubic Interpolation', linestyle='--')

    # Plotting the interpolated points
    plt.plot(desired_maturity, yield_cubic, 'bx', markersize=10, label=f'Interpolated Point (Cubic): {yield_cubic:.2f}%')

    # Adding labels and title
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Yield (%)')
    plt.title('Yield Curve Interpolation')
    
    # Show legend
    plt.legend()






# Nelson-Siegel model function
def nelson_siegel(t, beta0, beta1, beta2, tau):
    return beta0 + beta1 * (1 - np.exp(-t / tau)) + beta2 * ((t / tau) * (1 - np.exp(-t / tau)))

# Objective function to minimize the sum of squared errors
def objective_function_ns(params, *args):
    t, y = args
    return np.sum((y - nelson_siegel(t, *params)) ** 2)


def plot_ns_curve(maturities,yields):

    # Sample data (maturities and corresponding yields)
    maturities = np.array(maturities)  # Maturities in years
    yields = np.array(yields)  # Corresponding yields

    # Initial guess for the parameters
    initial_params = [2.0, -1.0, -1.0, 1.0]  # beta0, beta1, beta2, tau

    # Fitting the model
    result = minimize(objective_function_ns, initial_params, args=(maturities, yields))
    if result.success:
        fitted_params = result.x
        st.write("Fitted parameters:  \n beta0 = ", fitted_params[0],  "  \nbeta1 = ", fitted_params[1], "  \nbeta2 = ", fitted_params[2], "  \ntau = ",fitted_params[3])
    else:
        st.write("Optimization did not converge.")

    # Using the fitted parameters to plot the curve
    curve_maturities = np.linspace(0, 30, 100)  # Generate finer maturities for smoother curve
    fitted_curve = nelson_siegel(curve_maturities, *fitted_params)

    plt.figure(figsize=(8, 6))
    plt.plot(curve_maturities, fitted_curve, label='Fitted Curve')
    plt.scatter(maturities, yields, color='red', label='Original Data')
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Yield')
    plt.legend()
    plt.title('Nelson-Siegel Fitted Curve')
    plt.grid(True)




# Nelson-Siegel-Svensson model function
def nelson_siegel_svensson(t, beta0, beta1, beta2, beta3, tau1, tau2):
    return beta0 + beta1 * (1 - np.exp(-t / tau1)) + beta2 * ((t / tau1) * (1 - np.exp(-t / tau1))) + beta3 * (((1-np.exp(-t / tau2))/(t / tau2))-np.exp(-t / tau2))

# Objective function to minimize the sum of squared errors
def objective_function_nss(params, *args):
    t, y = args
    return np.sum((y - nelson_siegel_svensson(t, *params)) ** 2)

def plot_nss_curve(maturities,yields):
    maturities = [1/12,2/12,3/12,4/12,6/12,1,2,3,5,7,10,20,30]  # in years
    # Sample data (maturities and corresponding yields)
    maturities = np.array(maturities)  # Maturities in years
    yields = np.array(yields)  # Corresponding yields

    # Initial guess for the parameters
    initial_params = [2.0, -1.0, -1.0, 1.0, 1.0, 1.0]  # beta0, beta1, beta2, beta3, tau1, tau2

    # Fitting the model
    result = minimize(objective_function_nss, initial_params, args=(maturities, yields))
    if result.success:
        fitted_params = result.x
        st.write("Fitted parameters:  \n beta0 = ", fitted_params[0],  "  \nbeta1 = ", fitted_params[1], "  \nbeta2 = ", fitted_params[2], "  \nbeta3 = ",fitted_params[3],"  \ntau1 = ",fitted_params[4],"  \ntau2 = ",fitted_params[5] )
    else:
        st.write("Optimization did not converge.")

    # Using the fitted parameters to plot the curve

    curve_maturities = np.linspace(0, 30, 100)  # Generate finer maturities for smoother curve
    fitted_curve = nelson_siegel_svensson(curve_maturities, *fitted_params)

    plt.figure(figsize=(8, 6))
    plt.plot(curve_maturities, fitted_curve, label='Fitted Curve')
    plt.scatter(maturities, yields, color='red', label='Original Data')
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Yield')
    plt.legend()
    plt.title('Nelson-Siegel Svensson Fitted Curve')
    plt.grid(True)






def plot_vasicek():
    class Rate():
        def _init_(self, N=1000, dt=0.05, sigma=0.001, alpha=10, beta=0.04, r0=0.04):
            """
            Parameters
            ----------
            N : Number of steps
            dt : Time steps
            sigma : Scale
            alpha : Speed of adjustment
            beta : Long-term equilibrium
            r0: Current interest rate
            """
            self.N = N
            self.dt = dt
            self.sigma = sigma
            self.alpha = alpha
            self.beta = beta
            self.r0 = r0

        def wiener(self):
            """
            Wiener process
            Returns
            -------
            Array of realisations of Wiener process
            """
            #Result vector
            out = np.zeros(self.N)

            #Initial value
            out[0] = np.sqrt(self.dt) * self.sigma * np.random.normal(0, 1)

            for j in range(1, self.N):
                out[j] = out[j - 1] + np.sqrt(self.dt) * self.sigma * np.random.normal(0, 1)
                
            return out

        
        def vasicek(self):
            """
            Returns
            -------
            Array of simulated interest rates 
            """        
            #Result vector
            out = np.zeros(self.N)
            
            #Initial value
            out[0] = self.r0
            
            #Wiener process
            w = Rate()
            w = w.wiener()

            for j in range(1, self.N):
                out[j] = out[j - 1] + self.alpha*(self.beta - out[j - 1])*self.dt + w[j]
                
            return out
    oo = Rate()
    vs = oo.vasicek()
    plt.plot(vs)

    
















st.title("Yield curve")

# Choix de la méthode : 
modele = st.radio("Choisir une méthode", ("Linear", "Cubic", "Nelson Siegel", "Nelson Siegel Svensson", "Vasicek"))

if modele == "Linear":
    desired_maturity  = st.number_input("desired maturity", value=6)
    plot_linear_curve(maturities, yields, desired_maturity)
elif modele == "Cubic":
    desired_maturity  = st.number_input("desired maturity", value=6)
    plot_cubic_curve(maturities, yields, desired_maturity)
elif modele == "Nelson Siegel":
    def nelson_siegel_formula():
        return r"$R(\tau) = \beta_0 + \beta_1 \cdot \frac{1 - e^{-\lambda \cdot \tau}}{\lambda \cdot \tau} + \beta_2 \cdot \left(\frac{1 - e^{-\lambda \cdot \tau}}{\lambda \cdot \tau} - e^{-\lambda \cdot \tau}\right)$"
    st.title("Méthode de Nelson-Siegel")
    st.write(nelson_siegel_formula())
    plot_ns_curve(maturities,yields)
elif modele == "Nelson Siegel Svensson":
    def nelson_siegel_svensson_formula():
        return r"$r(\tau) = \beta_0 + \beta_1 \left(\frac{1 - e^{-\lambda \tau}}{\lambda \tau}\right) + \beta_2 \left(\frac{1 - e^{-\lambda \tau}}{\lambda \tau} - e^{-\lambda \tau}\right) + \beta_3 \left(\frac{1 - e^{-\lambda_2 \tau}}{\lambda_2 \tau} - e^{-\lambda_2 \tau}\right)$"
    st.title("Méthode de Nelson-Siegel")
    st.write(nelson_siegel_svensson_formula())
    plot_nss_curve(maturities,yields)
elif modele == "Vasicek":
    plot_vasicek()

st.pyplot()

st.set_option('deprecation.showPyplotGlobalUse', False)