import numpy
from datetime import datetime
import pandas
import matplotlib.pyplot as plt
from scipy.signal import lfilter, butter
import os
plt.ion()

def simulate_protocol(file_name, folder, sampling_frequency = 'H', cycle_days = [], activity_period = [], 
                      signal_type = 'sine', noise = False, snr_db = 20, only_positive = True, remove_file = False):
    """
    Simulate a protocol

    :param file_name: The name of the file
    :type file_name: str
    :param folder: The folder where the file will be saved
    :type folder: str
    :param sampling_frequency: The sampling frequency of the protocol in pandas format (e.g. '30T' for 30 minutes),
        defaults to 'H'
    :type sampling_frequency: str
    :param activity_period: The period of the sin curve (in hours), defaults to 24
    :type activity_period: int
    :param signal_type: The type of the signal (sine, square, sawtooth), defaults to 'sine'.
    :type signal_type: str
    :param noise: If True, noise will be added to the signal, defaults to None
    :type noise: bool   
    :param snr_db: The signal to noise ratio in dB, defaults to 0.1
    :type snr_db: float
    :param remove_file: If True, the file will be removed if it already exists, defaults to False
    :type remove_file: bool
    :return: The simulated activity data
    :rtype: list
    """
    if not isinstance(file_name, str):                                                                                  # If file_name is not a string
        raise TypeError("file_name must be a string")                                                                   # Raise an error
    if not isinstance(folder, str):                                                                                     # If folder is not a string
        raise TypeError("folder must be a string")                                                                      # Raise an error
    if not isinstance(sampling_frequency, str):                                                                         # If sampling_frequency is not a string
        raise TypeError("sampling_frequency must be a string")                                                          # Raise an error
    if not isinstance(cycle_days, list):                                                                                # If cycle_days is not a list
        raise TypeError("cycle_days must be a list")                                                                    # Raise an error
    if not isinstance(activity_period, list):                                                                           # If activity_period is not a list
        raise TypeError("activity_period must be a list")                                                               # Raise an error
    if len(cycle_days) != len(activity_period) and len(cycle_days) != 0:                                                # If the length of cycle_days is different than the length of activity_period
        raise ValueError("The number of cycle days is different than the number of activity periods")                   # Raise an error                                                                  
    if not isinstance(signal_type, str):                                                                                # If signal_type is not a string
        raise TypeError("signal_type must be a string (sine, square, sawtooth)")                                        # Raise an error
    if not isinstance(noise, bool):                                                                                     # If noise is not a boolean
        raise TypeError("noise must be a boolean")                                                                      # Raise an error
    if not isinstance(snr_db, float) and not isinstance(snr_db, int):                                                   # If snr_db is not a float
        raise TypeError("snr_db must be a float or integer")                                                            # Raise an error

    if len(cycle_days) != len(activity_period):                                                                   
        raise ValueError("The number of cycle days is different than the number of activity periods")                   # Raise an error

    act_path = folder + "/" + file_name + ".asc"                                                                        # The path of the file where the simulated activity data will be saved
        
    ts = pandas.date_range('00:00:00', periods = 2, freq = sampling_frequency)                                          # Create to get the sampling interval
    period = ts[1] - ts[0]                                                                                              # Get the sampling interval
    sample_period = period.total_seconds()                                                                              # Get the sampling interval in seconds
    desired_freq = [(period*60*60)/sample_period for period in activity_period]                                         # Get the desired frequency of the sin curve in hours

    check_folder = os.path.isdir(folder)                                                                                # Check if the folder exists
    if not check_folder:                                                                                                # If the folder doesn't exist
        os.makedirs(folder)                                                                                             # Create the folder to save the file
        check_act_file = os.path.isfile(act_path)                                                                       # Check if the file exists in the folder
        if check_act_file:                                                                                              # If the file exists
            if remove_file:                                                                                             # If remove_file is True
                os.remove(act_path)                                                                                     # Remove the existing file
            else:                                                                                                       # If remove_file is False
                raise ValueError("File already exists")                                                                 # Raise an error
    else:                                                                                                               # If the folder exists
        check_act_file = os.path.isfile(act_path)                                                                       # Check if the file exists in the folder
        if check_act_file:                                                                                              # If the file exists
            if remove_file:                                                                                             # If remove_file is True
                os.remove(act_path)                                                                                     # Remove the existing file
            else:                                                                                                       # If remove_file is False
                raise ValueError("File already exists")                                                                 # Raise an error
    
    start_date = '2022-01-01 00:00:00'
    num_cycle_days = numpy.sum(cycle_days)
    end_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S') + pandas.Timedelta(days = num_cycle_days)
    
    data = pandas.date_range(start_date, end_date, freq = sampling_frequency, inclusive = 'left')                       # Create the time series
    len_data = len(data)                                                                                                # Get the length of the time series
    time = numpy.linspace(0, len_data, len_data)                                                                        # Create the time vector

    count_days_df = pandas.DataFrame({'data': data, 'value': range(len_data)})                                          # Create a dataframe with the date and the value
    count_per_day = count_days_df.groupby(count_days_df['data'].dt.date).size()

    days_per_period = [0] + numpy.cumsum(cycle_days).tolist()                                                           # Get the days per period
    samples_per_period = []
    for count, _ in enumerate(days_per_period):
        if count != 0:
            samples_per_period.append(count_per_day[days_per_period[count - 1]:days_per_period[count]].sum())

    start_day_period = 0                                                                                                # Initialize the start day period
    total_activity = []
    last_activity = 0
    last_direction = 0

    for freq, samples in zip(desired_freq, samples_per_period):                                                         # For each frequency and cycle day
        end_day_period = start_day_period + samples                                                                     # Get the end day period
        
        if last_activity >= 0:                                                                                          # If the last activity is positive
            if last_direction == 1:                                                                                     # If the last direction is positive
                shift = numpy.arcsin(last_activity)                                                                     # Get the shift of the sin curve (positive and raising)
            else:                                                                                                       # If the last direction is negative
                shift = numpy.pi - numpy.arcsin(last_activity)                                                          # Get the shift of the sin curve (positive and falling)
        else:                                                                                                           # If the last activity is negative
            if last_direction == 1:                                                                                     # If the last direction is positive
                shift = numpy.arcsin(last_activity)                                                                     # Get the shift of the sin curve (negative and raising)
            else:                                                                                                       # If the last direction is negative
                shift = - numpy.pi - numpy.arcsin(last_activity)                                                        # Get the shift of the sin curve (negative and falli
        len_period = len(time[start_day_period:end_day_period])                                                         # Get the length of the period in seconds
        curve = numpy.sin(numpy.multiply(2*numpy.pi/freq, range(len_period)) + shift) + 1                               # Create the sin curve and shift to get only positve values (to simulate the activity and temperature)
        last_activity = curve[-1] - 1                                                                                   # Get the last value of the sin curve
        if last_activity > (curve[-2] - 1):                                                                             # If the last value is greater than the previous one
            last_direction = 1                                                                                          # The last direction is positive
        else:                                                                                                           # If the last value is smaller than the previous one
            last_direction = 0                                                                                          # The last direction is negative

        if signal_type == 'square':
            activity_curve = numpy.where(curve >= 1, 2, 0)                                                              # Put delimeted edges to the sin curve           
        elif signal_type == 'sine':
            activity_curve = curve
        elif signal_type == 'sawtooth':
            activity_curve = numpy.where(curve >= 1, 0, curve)                                                          # Put delimeted edges to the sin curve
        else:
            raise ValueError("The signal type is not valid (sine, square, triangle, sawtooth, square_lowpass)")  

        if noise:                                                                                                       # If noise is True
            amplitude = 50
            activity = amplitude*activity_curve                                                                         # Multiply the activity vector (with/without random values) by the activity curve
            activity = add_noise(activity, snr_db)                                                                      # Add noise to the activity vector
        else:                                                                                                           # If noise is False
            amplitude = 50
            activity = amplitude*activity_curve                                                                         # Multiply the activity vector (with/without random values) by the activity curve
        
        if only_positive:                                                                                               # If only_positive is True
            activity = numpy.where(activity < 0, 0, activity)                                                           # Set the negative values to 0

        activity = numpy.round(activity, 0)                                                                             # Round the values of the activity vector
        total_activity = numpy.append(total_activity, activity)                                                         # Append the activity vector to the total activity vec
        start_day_period = end_day_period                                                                               # The start day period is the end day period of the previous cycle

    total_activity = total_activity.tolist()                                                                            # Convert the total activity vector to a list
    
    with open(act_path, 'w+') as file:
        for count, date in enumerate(data):
            file.write(date.strftime("%m/%d/%y") + ',' + date.strftime("%H:%M:%S") + ',' +
                       '{:.2f}'.format((total_activity[count])) + '\n')

    print('Saved successfully in ' + act_path)

    return total_activity

def add_noise(signal, snr_db):
    noise = numpy.random.normal(0, 1, signal.shape)                         # Generate random noise with mean 0 and standard deviation 1
    signal_power = numpy.mean(signal ** 2)                                  # Calculate the power of the original signal
    noise_power = numpy.mean(noise ** 2)                                    # Calculate the power of the noise
    snr_linear = 10 ** (snr_db / 10.0)                                      # Convert SNR from dB to linear scale
    desired_noise_power = signal_power / snr_linear                         # Calculate the desired noise power based on the desired SNR (in dB)
    scaled_noise = numpy.sqrt(desired_noise_power / noise_power) * noise    # Scale the noise to achieve the desired SNR
    noisy_signal = signal + scaled_noise                                    # Add the scaled noise to the original signal to get the final noisy signal

    return noisy_signal