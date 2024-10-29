import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def plot_learning_curve(y_, lim, label=["curve"], figure_file="", title='Running average of previous 100 scores'):
    """Plots a learning curve with smoothing and saves the figure to a file."""
    
    color = ['r', 'g', 'b', 'y']  # Define colors for the curves
    x_ = np.arange(lim, dtype=np.float32)  # Create x-axis values based on the limit
    x = np.linspace(x_.min(), x_.max(), 500)  # Generate 500 points for smooth plotting
    
    plt.figure()  # Create a new figure
    for i in range(len(y_)):
        # Create a spline interpolation for smoothness
        X_Y_Spline = make_interp_spline(x_, y_[i])
        y = X_Y_Spline(x)
        
        # Plot the smoothed curve
        plt.plot(x, y, label=label[i], color=color[i])
    
    plt.ylabel(title)  # Set the y-axis label
    plt.xlabel("Episode")  # Set the x-axis label
    plt.title(title, color="m", weight='bold')  # Set the plot title
    plt.legend()  # Display the legend
    plt.savefig(figure_file)  # Save the figure to a file
