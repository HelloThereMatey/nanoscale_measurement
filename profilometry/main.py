import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import tkinter as tk
#from PyQt6 import QtWidgets  # PyQt6 is the BEST gui framework for python.
import plotly.express as px

#These parameters set my prefence for the appearance of a matplotlib chart. 
plt.rcParams["backend"] = 'TkAgg'  ##This ensures matplotlib uses the tk backend. 
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["figure.dpi"] = 135
plt.rcParams["savefig.format"] = "svg"
plt.rcParams['animation.frame_format'] = "svg"
plt.rcParams['font.family'] = 'serif'

##### This gets thee current working directory (where this .py file is located) and the parent directory.
import sys
import os
wd = os.path.dirname(__file__); parent = os.path.dirname(wd)
fdel = os.path.sep

########## STANDALONE UTILITY FUNCTIONS ##########
try:
    from PyQt6 import QtWidgets

    def qt_load_file_dialog(caption: str = "Choose a file", initial_dir: str = wd, 
                            filetypes: str = "All Files (*);;CSV Files (*.csv)"):
        app = QtWidgets.QApplication.instance()  # Check if an instance already exists
        if not app:  # If not, create a new instance
            app = QtWidgets.QApplication(sys.argv)

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(parent=None, caption = caption, directory = initial_dir,
                                                              filter = filetypes)
        return file_path

    load_dialog = qt_load_file_dialog

except ImportError:
    from tkinter import filedialog

    def tk_load_file_dialog(caption: str = "Choose a file", filetypes=(('CSV files', '*.csv'), 
                            ('All files', '*.*')),initial_dir: str = wd):
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        file_path = filedialog.askopenfilename(title=caption, filetypes=filetypes, initialdir=initial_dir)
        return file_path

    load_dialog = tk_load_file_dialog

## If you can install PyQt6, you can use thismain
#     file_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, dialog_title, initial_dir, file_types, options=QtWidgets.QFileDialog.Option.DontUseNativeDialog)

#     return file_path

# In case you can't install PyQt6, you can use the tkinter file dialog instead. Installtion of PyQt6 requires pip. 
def basic_load_dialog(initialdir: str = wd, title: str ='Choose your file...', 
                    filetypes: tuple = (('Image files', '*.png *.bmp *.jpg *.jpeg *.pdf *.svg *.tiff *.tif'),
                                                  ('All files', '*.*'))):
    window = tk.Tk()
    window.withdraw()
    file_path = filedialog.askopenfilename(filetypes=filetypes, initialdir=initialdir, parent=window, title=title)
    window.withdraw()  
    return file_path

def ewma_fb(df_column: pd.Series, span):
    ''' Apply forwards, backwards exponential weighted moving average (EWMA) to df_column. '''
    # Forwards EWMA.
    fwd = df_column.ewm(span=span).mean()
    # Backwards EWMA.
    bwd = df_column[::-1].ewm(span=span).mean()[::-1]
    # Add and take the mean of the forwards and backwards EWMA.
    fb_ewma = (fwd + bwd) / 2
    return fb_ewma

def remove_outliers(despiked: pd.Series, fbewma: pd.Series, delta_percent: float) -> pd.Series:
    ''' Remove data from despiked that is > delta_percent% away from fbewma. '''
    # Calculate the percentage-based delta
    print("Remove outliers method, ", "Delta %: ", delta_percent)
    data = despiked.copy() - despiked.min() + 0.00001
    ema = fbewma.copy() - fbewma.min() + 0.00001
    # Calculate the condition for outliers
    cond_delta = abs(((data - ema) / ema) * 100)
    
    # Replace outliers with NaN
    despiked_cleaned = despiked.where(cond_delta <= delta_percent, other=pd.NA)
    
    return despiked_cleaned, cond_delta

def find_closest_index(series: pd.Series, target_index_value: float) -> tuple:
    # Subtract the target value from the series and take the absolute value
    diff = abs((series.index - target_index_value))
    # Find the index of the minimum value in the resulting series
    closest_iloc = diff.argmin()
    # Find the iloc position of the closest index
    closest_index_value = series.index[closest_iloc]
    return closest_index_value, closest_iloc

def fit_line(data:pd.Series, x1: tuple , x2: tuple):  # x1 and x2 should be tuples returned by the get_closest_index function.
    x = data.index[x1[1]:x2[1]]
    y = data[x]
    vals = np.polyfit(x, y, 1)
    fit = np.polyval(vals, data.index)
    return vals, fit

def find_first_num(data: pd.DataFrame):
    for i in range(len(data)):
        try:
            float(data[data.columns[0]][i])
            return i
        except ValueError: 
            pass
    return None

def cssplinefill(input_data: pd.Series):
    """Interpolate missing values in a pandas Series using cubic spline interpolation."""
    data = input_data.copy()
    # Identify the indices of the non-NaN values
    non_nan_indices = data.dropna().index
    non_nan_values = data.dropna().values

    # Fit a cubic spline to the non-NaN values
    cs = CubicSpline(non_nan_indices, non_nan_values)

    # Interpolate the NaN values
    nan_indices = data.index[data.isna()]
    interpolated_values = cs(nan_indices)

    # Replace the NaN values in the original series with the interpolated values
    data[nan_indices] = interpolated_values
    return data

def interpolate_gaps(series: pd.Series) -> pd.Series:
    # Create a copy of the series to avoid modifying the original
    interpolated = series.copy()
    
    # Find the indices of NaN values
    nan_indices = np.where(interpolated.isna())[0]
    
    # Group consecutive NaN indices
    nan_groups = np.split(nan_indices, np.where(np.diff(nan_indices) != 1)[0] + 1)
    
    for group in nan_groups:
        if len(group) > 0:
            # Find the values and indices before and after the NaN group
            start_idx = group[0] - 1 if group[0] > 0 else None
            end_idx = group[-1] + 1 if group[-1] < len(interpolated) - 1 else None
            
            if start_idx is not None and end_idx is not None:
                # Perform cubic spline interpolation
                x = np.array([start_idx, end_idx])
                y = np.array([interpolated.iloc[start_idx], interpolated.iloc[end_idx]])
                
                # We need at least 4 points for cubic interpolation
                if len(x) < 4:
                    # Add more points if available
                    left_points = max(0, start_idx - 2)
                    right_points = min(len(interpolated) - 1, end_idx + 2)
                    x = np.arange(left_points, right_points + 1)
                    y = interpolated.iloc[x].values
                    x = x[~np.isnan(y)]
                    y = y[~np.isnan(y)]
                
                if len(x) >= 4:
                    f = interp1d(x, y, kind='linear', fill_value='extrapolate')
                    interpolated.iloc[group] = f(group)
                else:
                    # Fallback to linear interpolation if not enough points
                    f = interp1d(x, y, kind='linear', fill_value='extrapolate')
                    interpolated.iloc[group] = f(group)
            elif start_idx is not None:
                # If there's no end value, forward fill
                interpolated.iloc[group] = interpolated.iloc[start_idx]
            elif end_idx is not None:
                # If there's no start value, backward fill
                interpolated.iloc[group] = interpolated.iloc[end_idx]
    
    return interpolated

########## CLASS DEFINITIONS ##########
class trace_profile(object):   
    """ This class is designed to handle the loading, processing, and plotting of profilometry data.
    Parameters:
    - data: pd.Series, optional. The data to be plotted.
    - header: pd.Series, optional. The header data containing metadata from Vision64 software.
    You must provide BOTH data and header if you want to skip the load dialog. """ 

    def __init__(self, data: pd.Series = pd.Series(), header: pd.Series = pd.Series(), filePath: str = ""):
        """ Initialize the trace_profile object. 
        Parameters:
        - data: pd.Series, optional. The data to be plotted.
        - header: pd.Series, optional. The header data containing metadata from Vision64 software.
        You must provide BOTH data and header if you want to skip the load dialog. 
        - filePath: str, optional. The full path to the csv file containing the profilometry data. 
        Use filePath to skip the file dialog and load a specific file. """
        print("\n\n","proflometry module by James Bishop, use for analysis of profilometry trace data from DekTak profilomoter. Email: james.bishop@uts.edu.au",\
              "if you have further questions.","\n\n")

        if data.empty and header.empty:
            if filePath:
                self.header , self.data = self.load_csv_trace_data(filePath)
            else:
                self.header , self.data = self.load_csv_trace_data()
        else:
            self.data = data
            self.header = header
        self.raw_data = None
        self.leveled_data = None
        self.fig = None
        self.ax = None
        self.vertical_lines = []  # List to keep track of vertical lines
        self.fresh_texts = []  # List to keep track of text objects
        self.latest_height_measurement = None
        self.x_measurements = []
        self.x1 = None
        self.x2 = None
        self.all_data = None

    def profilometry_plot(self, data: pd.Series = None, title: str = "Profilometry data", **fig_kwargs):
        """ Produces a basic matplotlib plot of your 1D profilometry data.
        **fig_kwargs allows the user to pass in any keyword arguments that are accepted by the plt.subplots() function. 
        See matplotlib.pyplot.subplots documentation for more information. """

        fig, ax = plt.subplots(1, 1, **fig_kwargs)  # Create a figure and axis object
        if data is not None:
            self.data = data
        ax: plt.Axes
        ax.plot(self.data, label = "Raw trace data")
        ax.axhline(y=0, color='r', linestyle='--', lw = 0.75)
        ax.grid(visible=True, which='both', axis='both', ls = ":", lw = 0.75)
        # Increase the number of ticks on x and y axes
        ax.minorticks_on()
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("Height (nm)")
        ax.set_title(title, loc = "left")
        ax.margins(0.015, 0.03)
        self.fig = fig
        self.ax = ax
        self.y_max = self.data.max()
        # Connect the double-click event to the handler
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)

        # Initialize the text box
        self.text_box = self.fig.text(0.72, 0.03, "", ha='center', fontsize=10, color='red')
        self.fig.canvas.mpl_connect('close_event', self.on_close)

        if self.leveled_data is not None:
            self.ax.plot(self.data.index, self.substrate_fit, label="Substrate Fit", color='black', ls = "--", lw = 0.75)
            self.ax.plot(self.data.index, self.leveled_data, label="Leveled Data", color = "green")
            instructions = self.ax.text(0.4, 1.03, "Double-click to measure height of the leveled data trace at that x value.\n\
            Right-click to remove the latest height measurement. \nClose chart to return measuement value.", fontsize=9, color='blue', transform=self.ax.transAxes)
        else:
            instructions = self.ax.text(0.3, 1.01, "LEVEL TRACE DATA: Double-click to add a vertical line to choose x1 and then x2.\nThese points specify the x-range for the linear regression of substrate.\n\
        Right-click to delete in case of wrong choice. Last two lines placed will be used. \nClose chart to run leveling.", fontsize=9, color='blue', transform=self.ax.transAxes)

        self.ax.legend()
        plt.show()

    def remove_spikes(self, threshold: float, index_postion_group_distance: int = 10, leveled: bool = False):
        """
        Remove spikes from the data using a threshold on the first derivative.
        * **Parameters**:
            - threshold: float. The threshold value for the first derivative to identify spikes. Units rate of change of same units as the data.
            - index_postion_group_distance: int. The maximum distance between two spikes to be considered as a single spike. 
            Units are of posiion in index (i, .iloc), not distance or whatever units the index happens to be in. Index of series 
            is reset to a numeric index for the operation of spike grouping.
        """

        if self.data is None or self.data.empty:
            raise ValueError("No data available to process.")
        
        if leveled:
            self.original_data = self.leveled_data.copy()
        else:
            self.original_data = self.data.copy()
        # Calculate the first derivative
        self.derivative = self.original_data.diff()
        self.mask = pd.Series(0, index=self.original_data.index)
        self.mask[self.derivative.abs() > threshold] = 1
        
        grouped_mask = self.mask.copy().reset_index(drop=True)
        non_zero_positions = grouped_mask[grouped_mask != 0].index
        for i in range(len(non_zero_positions) - 1):
            if non_zero_positions[i + 1] - non_zero_positions[i] <= index_postion_group_distance:
                grouped_mask.iloc[non_zero_positions[i]:non_zero_positions[i + 1] + 1] = 1
        self.mask = pd.Series(grouped_mask.to_list(), index = self.original_data.index)
        
        self.all_data = pd.DataFrame({"Raw Data": self.original_data, "Derivative": self.derivative,
                                      "Mask": self.mask, "Despiked": self.original_data.copy()})
        self.all_data.loc[self.mask == 1, "Despiked"] = np.nan
        #Overwrite the main data series with the despiked data.
        self.data = self.all_data["Despiked"]

    def on_mouse_click(self, event):
        """Event handler for mouse click events."""
        if event.dblclick:
            x = event.xdata
            print("Adding vertical line for measurement at x = ", x)
            
            if x is not None:
                line = self.ax.axvline(x=x, color='black', linestyle='--', lw=0.75)
                self.vertical_lines.append(line)
                self.fig.canvas.draw()
                if self.x1 is not None and self.x2 is not None:
                    self.latest_height_measurement = self.leveled_data.iloc[find_closest_index(self.leveled_data, x)[1]]
                    textb = self.ax.text(x + 0.01, self.latest_height_measurement + 0.5 * self.latest_height_measurement, f"{self.latest_height_measurement:.2f} nm", fontsize=8, color='black')
                    self.fresh_texts.append(textb)
                    self.update_text_box()
                else:
                    self.x_measurements.append(x)
                    if len(self.x_measurements) > 2:
                        self.x_measurements.pop(0)
                        
        elif event.button == 3:  # Right-click
            if self.vertical_lines:
                line = self.vertical_lines.pop()
                line.remove()
                self.latest_height_measurement = None
                self.update_text_box()
            if self.fresh_texts:
                tb = self.fresh_texts.pop()
                tb.remove()
            if self.x_measurements:
                self.x_measurements.pop()
            self.fig.canvas.draw()

    def on_close(self, event):
        """Event handler for closing the chart."""
        if len(self.x_measurements) >= 2:
            self.x1, self.x2 = self.x_measurements[-2], self.x_measurements[-1]
        else:
            self.x1, self.x2 = None, None
        print(f"Will fit substratself.all_datae baseline for leveling between, x1: {self.x1}, x2: {self.x2}")
        
    def update_text_box(self):
        """Update the text box with the latest height measurement."""
        if self.latest_height_measurement is not None:
            self.text_box.set_text(f"Latest Height Measurement: {self.latest_height_measurement:.2f} nm")
            self.fig.canvas.draw()

    def separate_header(self) -> tuple:
        """ This function separates the header metadata from the trace data in a Vision64 csv file."""
        if self.raw_data is None:
            raise ValueError("No data has been loaded yet.")
        
        data = self.raw_data.dropna().reset_index(drop=True)
        first_num = find_first_num(data)
        if first_num is None:
            raise ValueError("No numeric values found in the specified column.")
        
        names_index = first_num - 1

        header = data.iloc[:names_index].copy()
        header.rename(columns={header.columns[1]: "Value"}, inplace=True)
        data = data.iloc[first_num-1:-1].copy()
        
        data.columns = data.iloc[0]
        data = data[1:]
        return header, data
    
    def load_csv_trace_data(self, filepath: str = "") -> pd.Series:
        """ This function loads a csv file containing profilometry data and returns the header and data as separate pd.Series objects.
        Provide filepath as a string to load a specific file. filepath should be full path to the file on your machine.
        If no filepath is provided, a file dialog (from Qt6) will open to allow the user to choose a file..."""

        if filepath:
            pass
        else:
            filepath = load_dialog(caption="Choose a .csv file containing profilometry data exported from Vision64 software.")
            
        self.raw_data = pd.read_csv(filepath)
        header, data = self.separate_header()
        header.set_index("Meta Data", inplace=True, drop = True)
        data = data.astype(float)
        data.set_index(data.columns[0], inplace=True, drop = True)
        data = pd.Series(data[data.columns[0]], name = data.columns[0]).astype(float)
        header = pd.Series(header[header.columns[0]], name = header.columns[0]).astype(str)
        return header, data
    
    def level_data(self, x1: float, x2: float):
        """ This function levels the data between two x values by fitting a line to the data between those points and subtracting it from the data."""
        print("Fitting line to substrate between x1 and x2 using linear regression (np.polyfit(deg = 1)).\n\
          Is a LBF with equal weighting to all points between x1 & x2.")
        
        self.x1 = x1; self.x2 = x2
        x1 = find_closest_index(self.data, x1)
        x2 = find_closest_index(self.data, x2)
        self.vals, self.substrate_fit = fit_line(self.data, x1, x2)
        self.leveled_data = self.data - self.substrate_fit
        self.profilometry_plot(title="Leveled profilometry data")
        if self.all_data is None:
            self.all_data = pd.DataFrame({"Total Profile(nm)": self.data, "Substrate Fit": self.substrate_fit, "Leveled Data": self.leveled_data})
        else:
            self.all_data["Leveled Data"] = self.leveled_data
            self.all_data["Substrate Fit"] = self.substrate_fit

    def remove_outers(self, delta_pct: float = 1.5, ewma_per: int = 10):
        """ Remove data from the leveled data that is > delta_pct from the moving average of the data. 
        * **Parameters**:
            - delta_pct: float, optional. The percentage value to remove data from the leveled data that is > delta_pct from the moving average of the data.
            - ewma_per: int, optional. The period for the exponential weighted moving average (EWMA) to be used in the outlier removal process."""
        
        if self.data is None:
            raise ValueError("No leveled data available to process.")
        self.all_data["EMA_FB_"+str(ewma_per)] = ewma_fb(self.data, ewma_per)
        self.data = pd.Series(remove_outliers(self.data, self.all_data["EMA_FB_"+str(ewma_per)], delta_pct)[0])
        self.all_data["Outliers Removed"] = self.data

    def interpolate_trace(self):
        """Interpolate missing values in the data using cubic spline interpolation."""
        if self.data is None or self.data.empty:
            print("No data available for interpolation.")
            return

        # Identify the indices of the non-NaN values
        non_nan_indices = self.data.dropna().index
        non_nan_values = self.data.dropna().values

        # Fit a cubic spline to the non-NaN values
        cs = CubicSpline(non_nan_indices, non_nan_values)

        # Interpolate the NaN values
        nan_indices = self.data.index[self.data.isna()]
        interpolated_values = cs(nan_indices)

        # Replace the NaN values in the original series with the interpolated values
        self.data[nan_indices] = interpolated_values

############ Conveniemnce function to run a standard thickness measurement.  #########
def measure_thickness(filepath: str = "", data: pd.Series = pd.Series(), header: pd.Series = pd.Series(), x1: float = 0, x2: float = 1, 
                      title: str = "Profilometry data", despike_first: bool = True, despike_threshold: float = 1.5, spike_window: float = 20,
                      outlier_threshold: float = 10, ewma_period: int = 20):
    """ This function is a convenience function to run the several functions needed to do a film thickness measuremnt on profilometry data. 
    * **Parameters:** 
        - **data:** pd.Series, optional. The data to be plotted.
        - **header:** pd.Series, optional. The header data containing metadata from Vision64 software.
        *You must provide BOTH data and header if you want to skip the load dialog.*
        - **filepath:** str, optional. The full path to the csv file containing the profilometry data.
        - **profile_data:** trace_profile object. The trace_profile object containing the data and header and everything else. 
        - **despike_first:** bool, optional. If True, the data is despiked first using the remove_spikes method of the trace_profile object.
        - **despike_threshold:** float, optional. The threshold value in the first derivative of your data to identify spikes.
        - **spike_window:** int, optional. The maximum distance between two spikes to be considered as a single spike. Units are of posiion 
        in index (i, .iloc), not distance or whatever units the index happens to be in. Index of series is reset to a numeric index for 
        the operation of spike grouping.
        - **outlier_threshold:** float, optional. The percentage value to remove data from the leveled data that is > delta_pct from 
        the moving average of the data.
        - **ewma_period:** int, optional. The period for the exponential weighted moving average (EWMA) to be used in the outlier removal process.
    * **Returns:** 
        - **latest_height_measurement**: float. The thickness of a film between two x values.
        - **profile_data**: trace_profile object. The trace_profile object containing the data and header and everything else.
    The data is first leveled by fitting a line to the data between x1 and x2 (chosen byu user, corresponding to film susbstrate surface,
    and subtracting it from the data. The thickness is then measured by double-clicking on the plot to get the height at a specific x value."""
    
    profile_data = trace_profile(filePath=filepath)
    if despike_first:
        profile_data.remove_spikes(threshold = despike_threshold, index_postion_group_distance=spike_window)
        profile_data.remove_outers(delta_pct=outlier_threshold, ewma_per=ewma_period)
    interped = interpolate_gaps(profile_data.data)
    profile_data.all_data["Interpolated"] = interped
    profile_data.profilometry_plot(data = interped, title=title)
    print("X1 and X2 set: ", profile_data.x1, profile_data.x2)

    if profile_data.x1 and profile_data.x2:
        profile_data.level_data(profile_data.x1, profile_data.x2)

    if not despike_first:
        profile_data.remove_spikes(threshold = despike_threshold, index_postion_group_distance=spike_window, leveled=True)
        profile_data.remove_outers(delta_pct=outlier_threshold, ewma_per=ewma_period)
        interped = interpolate_gaps(profile_data.data)
        profile_data.all_data["Interpolated"] = interped

    return profile_data.latest_height_measurement, profile_data

def plotly_plot(pf: trace_profile):
    melted = pf.all_data[['Raw Data', 'Despiked', 'EMA_FB_20',
                          'Outliers Removed', 'Interpolated', 'Leveled Data', 'Substrate Fit']].melt(ignore_index=False).reset_index()
    
    final_plot = px.line(melted, x="Lateral(mm)", y="value", color="variable",  # Assuming 'variable' is the column that differentiates the traces
                         title="Profilometry Data from Vision64",
                         labels={
                             'Lateral(mm)': 'Trace distance (mm)', 
                             'value': 'Height (nm)', 
                             'variable': 'Series'  # Optional: label for the color legend
                             })
    final_plot.update_traces(selector={"name": 'Substrate Fit'}, line=dict(width=1.5, dash='dash', color='black'))  # Set the line width to 1
    final_plot.update_traces(selector={"name": 'Raw Data'}, line=dict(color='blue')) 
    final_plot.update_traces(selector={"name": 'Leveled Data'}, line=dict(color='green')) 

    # Add a dotted red line along y = 0
    final_plot.add_shape(type="line", x0=melted["Lateral(mm)"].min(),  # Start of the x-axis
                         y0=0, x1=melted["Lateral(mm)"].max(), y1=0,  # y = 0
                         line=dict(color="red",width=1.5,dash="dash"))

    # Update layout to reduce whitespace and add title
    final_plot.update_layout(
        title_x=0.5,  # Center the title
        margin=dict(l=20, r=20, t=40, b=20),  # Reduce margins
        width=1200,  # Set the width of the plot
        height=600  # Set the height of the plot
    )

    return final_plot

if __name__ == "__main__":
    filepath = '/home/purpy-furcat/Documents/Code/Lab_Related/Data/Traces/Salma_2.csv'
    thickness = measure_thickness(filepath=filepath)
    if thickness[0] is not None:
        print(f"The thickness of the film is {thickness[0]:.2f} nm.")

    pf = thickness[1]
    print(pf.all_data.columns)
    # pf = trace_profile(filePath=filepath)
    # pf.remove_spikes(1.5, 20, leveled=False)
    # fig, axes = plt.subplots(2, 1)
    # ax = axes[0]; ax2 = axes[1]
    # ax2.plot(pf.derivative, label = "First Derivative (dY/dX)", color = 'blue', alpha = 0.5, lw = 1)
    # ax.plot(pf.original_data, color = 'black', label = "Original Data", alpha = 0.5, lw = 1)
    # ax.plot(pf.data, color = 'red', label = "After spike removal")
    # ax.legend(); ax2.legend()
    # ema = ewma_fb(pf.data, 200)
    # outlied = remove_outliers(pf.data, ema, 5)
    # interp = interpolate_gaps(outlied[0])
   
    # fig6, ax7 = plt.subplots()
    # ax7.plot(outlied[0], label = "After Outlier Removal", color = 'dodgerblue', alpha = 0.5, lw = 2.5)
    # ax7.plot(ema, label = "EMA_20", color = 'green', alpha = 0.5, lw = 1)
    # ax7.plot(interp, label = "Interpolated", color = 'black', lw = 1)
    # ax7.legend()
    # # fig2, ax3 = plt.subplots(1, 1)
    # # ax3.plot(pf.data, label = "After Outlier Removal", color = 'red', alpha = 0.75, lw = 1.5)
    # # ax3.plot(pf.all_data["Total Profile(nm)"], label = "Original Data", color = 'black', alpha = 0.5, lw = 1)
    # # ax3.plot(pf.all_data["EMA_FB_20"], label = "EMA_20", color = 'green', alpha = 0.5, lw = 1)
    # # ax3.legend()
    # # pf.profilometry_plot(title="Profilometry Data")
    
    # interp = interpolate_gaps(pf.data)
    # #pf.all_data.to_excel(wd+fdel+"PF_Data.xlsx")
    # plt.show()

    ############ Plotly express plot of the data. #########
    pplot = plotly_plot(pf)
    pplot.show()