import numpy
import pandas
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import warnings
from CosinorPy import cosinor
plt.ion()

def positive_rad(rad):
    """
    Convert a radian value to a positive value between 0 and 2pi

    :param rad: Radian value
    :type rad: float
    :return: Positive radian value
    :rtype: float
    """
    if rad < 0:
        return 2 * numpy.pi + rad
    else:
        return rad


def _acrophase_ci_in_zt(best_models):
    """
    Get the acrophase in ZT

    :param acrophase: Acrophase in hours
    :type acrophase: float
    :param period: Period in hours
    :type period: float
    :return: Acrophase in ZT
    :rtype: float
    """
    periods = numpy.array(best_models['period'])
    acrophases = numpy.array(best_models['acrophase'])
    acrophases_ci = list(best_models['CI(acrophase)'])

    acrophases_lower = []
    acrophases_upper = []
    for values in acrophases_ci:
        if isinstance(values, list):
            acrophases_lower.append(values[0])
            acrophases_upper.append(values[1])
        else:
            acrophases_lower.append(numpy.nan)
            acrophases_upper.append(numpy.nan)
    
    lower_diff = numpy.abs(acrophases - acrophases_lower)
    upper_diff = numpy.abs(acrophases_upper - acrophases)

    apply_positive_rad = numpy.vectorize(positive_rad)
    acrophases = apply_positive_rad(acrophases)
    # acrophases_lower = apply_positive_rad(acrophases_lower)
    # acrophases_upper = apply_positive_rad(acrophases_upper)

    acrophases_zt = periods - (acrophases*periods)/(2*numpy.pi)
    # acrophases_lower_zt = periods - (acrophases_lower*periods)/(2*numpy.pi)
    # acrophases_upper_zt = periods - (acrophases_upper*periods)/(2*numpy.pi)

    acrophases_lower_zt = acrophases_zt - (lower_diff*periods)/(2*numpy.pi)
    acrophases_upper_zt = acrophases_zt + (upper_diff*periods)/(2*numpy.pi)

    best_models['acrophase_zt'] = acrophases_zt
    best_models['acrophase_zt_lower'] = acrophases_lower_zt
    best_models['acrophase_zt_upper'] = acrophases_upper_zt

    return best_models

def total_activity_per_day(protocol, save_folder = None, save_suffix = ''):
    print(protocol)
    protocol_df = protocol.data[['values','day']].copy()

    sum_by_group = protocol_df.groupby('day')['values'].sum()

    if save_folder != None:
        save_file = save_folder + '/total_activity_' + protocol.name.lower().replace(' ', '_') + '_' + save_suffix + '.xlsx'
        sum_by_group.to_excel(save_file)
    else:
        save_file = None

def fit_cosinor(protocol, dict = None, save_folder = None, save_suffix = ''):
    """
    Fit cosinor model to the data using the CosinorPy library.

    :param protocol: The protocol to fit the cosinor model to, if 0, the average of all protocols is used, defaults
    to 1
    :type protocol: int
    :param dict: A dictionary containing the parameters to fit the cosinor model with keys: record_type, time_shape,
    time_window,
    step, start_time, end_time, n_components. If None, the default values are used, defaults to None
    :type dict: dict
    :param save: If True, the cosinor model is saved in the cosinor_models folder, defaults to True
    :type save: bool
    :return: Dataframe containing the cosinor model parameters
    :rtype: pandas.DataFrame
    """
    warnings.filterwarnings("ignore")
    
    dict_default = {'time_shape': 'continuous',
                    'step': 0.1,
                    'start_time': 20,
                    'end_time': 30,
                    'n_components': [1]}

    if dict != None:
        for key in dict:
            if key in dict_default:
                dict_default[key] = dict[key]
            else:
                raise ValueError("The key " + key + " is not valid")

    time_shape = dict_default['time_shape']
    step = dict_default['step']
    start_time = dict_default['start_time']
    end_time = dict_default['end_time']
    n_components = dict_default['n_components']

    if start_time <= 0:
        raise ValueError("Start time must be greater than 0")

    period = numpy.arange(start_time, end_time, step)

    protocol_df = protocol.get_cosinor_df(time_shape = time_shape)
    protocol_name = protocol.name.replace('_', ' ').capitalize()

    result = cosinor.fit_group(protocol_df, n_components = n_components, period = period, plot = False)
    best = cosinor.get_best_fits(result, n_components = n_components)

    if len(best) == 0:
        best.loc[0] = pandas.np.nan
        best_models_extended = best
    else:
        best_models_extended = cosinor.analyse_best_models(protocol_df, best, analysis = "CI")

    best_models_extended.insert(1, 'first_hour', protocol_df['x'][0])
    best_models_extended = _set_significant_results(best_models_extended)
    best_models_extended = _acrophase_ci_in_zt(best_models_extended) #, protocol_df['y'][0])
    best_models_extended.reset_index(inplace = True, drop = True)

    if save_folder != None:
        save_file = save_folder + '/cosinor_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '.xlsx'
        best_models_extended.to_excel(save_file)
    else:
        save_file = None

    warnings.filterwarnings("always")
    return best_models_extended, save_file

def fit_cosinor_fixed_period(protocol, best_models, save_folder = None, save_suffix = ''):
    """
    Plot the cosinor period and acrophase for each day of the protocol

    :param best_models_per_day: The best models per day (output of the function get_cosinor_per_day)
    :type best_models_per_day: dict
    """
    warnings.filterwarnings("ignore")

    protocol_df = protocol.get_cosinor_df(time_shape = 'mean')
    protocol_name = protocol.name.replace('_', ' ').capitalize()

    best_models_per_day = pandas.DataFrame()

    change_test_day = protocol.cycle_days
    change_test_day = numpy.cumsum([0] + change_test_day[:-1])

    for count, label in enumerate(best_models['test']):
        test_df = protocol_df[protocol_df['test'] == label]
        index = test_df.index

        period = best_models['period'][count]

        days = []
        for i in range(0, len(index)):
            day = str(index[i].day)
            if len(day) == 1:
                day = '0' + day
            month = str(index[i].month)
            if len(month) == 1:
                month = '0' + month
            year = str(index[i].year)

            days.append(year + '-' + month + '-' + day)

        set_of_days = sorted(list(set(days)))
        test_df['day'] = days

        for count, day in enumerate(range(0, len(set_of_days))):
            day_df = test_df[test_df['day'] == set_of_days[day]]

            result_day = cosinor.fit_group(day_df, n_components = [1], period = [period], plot = False)
            best_model_day = cosinor.get_best_fits(result_day, n_components = [1])

            if len(best_model_day) == 0 or best_model_day['amplitude'][0] <= 0.01:                                      # If the model can't be fitted or the amplitude is too low (threshold), the model isn't considered
                best_model_day.loc[0] = pandas.np.nan
                best_model_day_extended = best_model_day
            else:
                best_model_day_extended = cosinor.analyse_best_models(day_df, best_model_day, analysis="CI")

            best_model_day_extended.insert(1, 'day', set_of_days[day])
            best_model_day_extended.insert(2, 'first_hour', day_df['x'][0])
            best_models_per_day = pandas.concat([best_models_per_day, best_model_day_extended], axis=0) 

    best_models_per_day = _set_significant_results(best_models_per_day)                                                 # Create a column indicating if the results are significant
    best_models_per_day = _acrophase_ci_in_zt(best_models_per_day)
    best_models_per_day.reset_index(drop = True, inplace = True)                                                        # Reset the index

    if save_folder != None:
        save_file = save_folder + '/cosinor_per_day_fixed_period_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '.xlsx'
        best_models_per_day.to_excel(save_file)
    else:
        save_file = None

    warnings.filterwarnings("always")
    return best_models_per_day, save_file

def fit_cosinor_per_day(protocol, dict = None, plot = False, save_folder = None, save_suffix = ''):
    """
    Fits a cosinor model to the data for each day of the protocol

    :param protocol: The protocol to fit the cosinor model parameters for, if 0, the average of all protocols is
    used, defaults to 1
    :type protocol: int
    :param dict: A dictionary containing the parameters to fit the cosinor model with keys: record_type, time_shape,
    time_window,
    step, start_time, end_time, n_components. If None, the default values are used, defaults to None
    :type dict: dict
    """
    warnings.filterwarnings("ignore")

    dict_default = {'day_window': 1,
                    'step': 0.1,
                    'start_time': 20,
                    'end_time': 30,
                    'n_components': [1]}

    if dict != None:
        for key in dict:
            if key in dict_default:
                dict_default[key] = dict[key]
            else:
                raise ValueError("The key " + key + " is not valid")

    day_window = dict_default['day_window']
    step = dict_default['step']
    start_time = dict_default['start_time']
    end_time = dict_default['end_time']
    n_components = dict_default['n_components']

    if start_time <= 0:
        raise ValueError("Start time must be greater than 0")

    periods = numpy.arange(start_time, end_time, step)

    protocol_df = protocol.get_cosinor_df(time_shape = 'mean')
    protocol_name = protocol.name.replace('_', ' ').capitalize()

    best_models_per_day = pandas.DataFrame()                                                                            # Dataframe containing the best model parameters for each day
    protocol_df['test'] = 'all'                                                                                         # Set the test label as 'all' for all the data
    index = protocol_df.index                                                                                           # Get the index (date) of the data

    days = []                                                                                                           # Create a list to store the days
    for i in range(0, len(index)):                                                                                      # Loop through the dates in the index
        day = str(index[i].day)                                                                                         # Get the day
        if len(day) == 1:                                                                                               # If the day is a single digit, add a 0 in front
            day = '0' + day                                                                                             # This is to make sure the date is in the format YYYY-MM-DD
        month = str(index[i].month)                                                                                     # Get the month
        if len(month) == 1:                                                                                             # If the month is a single digit, add a 0 in front
            month = '0' + month                                                                                         # This is to make sure the date is in the format YYYY-MM-DD
        year = str(index[i].year)                                                                                       # Get the year

        days.append(year + '-' + month + '-' + day)                                                                     # Add the date to the list of days

    set_of_days = sorted(list(set(days)))                                                                               # Get the unique days in the data and sort them

    if plot:                                                                                                            # If plot is True, create a figure
        fig, ax = plt.subplots(round(len(set_of_days)/10) + 1, 10, figsize=(40, 40), sharey = True)                     # Create a figure with a subplot for each day
        ax = ax.flatten()                                                                                               # Flatten the axes to make it easier to loop through them

    protocol_df['day'] = days                                                                                           # Add the day column to the dataframe

    for count, day in enumerate(range(0, len(set_of_days) - day_window + 1)):                                           # Loop through the days
        day_df = protocol_df[protocol_df['day'] == set_of_days[day]]                                                    # Get the data for the current day

        if day_window >= 2:                                                                                             # If the user want to use data from multiple days (set by day_window)
            for d in range(1, day_window):                                                                              # Loop through the days
                day_for_window_df = protocol_df[protocol_df['day'] == set_of_days[day + d]]                             # Get the data from the next day
                day_for_window_df['x'] = day_for_window_df['x'] + 24 * d                                                # Add 24 hours to the time for the next day
                day_df = pandas.concat([day_df, day_for_window_df], axis=0)                                             # Concatenate the current day data with the data from the next day

        result_day = cosinor.fit_group(day_df, n_components = n_components, period = periods, plot = False)             # Fit the cosinor model to the data for the current day
        best_model_day = cosinor.get_best_fits(result_day, n_components = n_components)                                 # Get the best model parameters for the current day

        if len(best_model_day) == 0 or best_model_day['amplitude'][0] <= 0.1:                                           #  If the model can't be fitted or the amplitude is too low (threshold), the model isn't considered
            best_model_day.loc[0] = pandas.np.nan
            best_model_day_extended = best_model_day
        else:
            best_model_day_extended = cosinor.analyse_best_models(day_df, best_model_day, analysis="CI")        
        
        best_model_day_extended.insert(1, 'day', set_of_days[day])                                                      # Add the day to the best model parameters dataframe
        best_model_day_extended.insert(2, 'first_hour', day_df['x'][0])                                                 # Add the first hour of the day to the best model parameters dataframe
        best_model_day_extended = _acrophase_ci_in_zt(best_model_day_extended)                                          # Reset the index of the dataframe
        best_models_per_day = pandas.concat([best_models_per_day, best_model_day_extended], axis=0)                     # Concatenate the best model parameters for the current day with the best model parameters for all the days

        if plot:                                                                                                        # If plot is True
            ax[count].bar(day_df['x'], day_df['y'], color = 'dimgray')
            ticks_to_plot = numpy.linspace(day_df['x'][0], day_df['x'][-1], num=5, endpoint=True)                       # Create a list of time points change the ticks on the x-axis
            ticks_to_plot = numpy.round(ticks_to_plot, 0)                                                               # Round the time points to 2 decimal places

            m_p_value = best_model_day_extended['p'][0]

            if not numpy.isnan(m_p_value) and m_p_value < 0.05:            
                m_acrophase = best_model_day_extended['acrophase'][0]                                                   # Get the acrophase estimate
                m_period = best_model_day_extended['period'][0]                                                         # Get the period estimate
                m_acrophase_zt = best_model_day_extended['acrophase_zt'][0]                                             # Convert the acrophase to zt

                m_frequency = 1/(m_period)                                                                              # Get the frequency estimate
                m_amplitude = best_model_day_extended['amplitude'][0]                                                   # Get the amplitude estimate
                model = m_amplitude*numpy.cos(numpy.multiply(2*numpy.pi*m_frequency, day_df['x']) + m_acrophase)        # Get the model
                offset = best_model_day_extended['mesor'][0]                                                            # Get the mesor
                model = model + offset                                                                                  # Add the mesor to the model

                ax[count].plot(day_df['x'], model, color = 'midnightblue', linewidth = 3)
                ax[count].axvline(m_acrophase_zt, color = 'black', linestyle = '--', linewidth = 1)
                ax[count].set_title(set_of_days[day] + '\n(PR: ' + str(round(m_period, 2))
                                + ', AC: ' + str(round(m_acrophase_zt, 2)) + ')', fontsize = 20)
            else:
                ax[count].set_title(set_of_days[day] + '\n(PR: NS, AC: NS)', fontsize = 20)
            
            ax[count].set_xticks(ticks_to_plot)
            ax[count].tick_params(axis='both', which='major', labelsize=20)
            ax[count].spines[['right', 'top']].set_visible(False)

    if plot:
        for c in range(len(ax)):
            if c > count:
                ax[c].axis('off')

        fig.suptitle('COSINOR MODEL PER DAY - ' + protocol_name.upper(), fontsize = 40)
        fig.supxlabel('TIME (ZT)', fontsize = 30)
        fig.supylabel('AMPLITUDE', fontsize = 30)
        plt.tight_layout(rect=[0.02, 0.01, 0.98, 0.98])

    if plot and save_folder == None:
        plt.show()

    if save_folder != None:
        save_file = save_folder + '/cosinor_per_day_w' + str(day_window) + '_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '.xlsx'
        best_models_per_day.to_excel(save_file, index = False)
        if plot:
            plt.savefig(save_folder + '/cosinor_per_day_w' + str(day_window) + '_' + 
                        protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '.png', backend=None)
            plt.close()
    else:
        save_file = None

    warnings.filterwarnings("always")
    return best_models_per_day, save_file

def _set_significant_results(best_models_extended):
    """
    Set the significant results of the cosinor analysis

    :param best_models_extended: The extended results of the cosinor analysis
    :type best_models_extended: pandas.DataFrame
    :return: The extended results of the cosinor analysis with the significant results
    :rtype: pandas.DataFrame
    """
    best_models_extended.insert(0, 'significant', 0)

    best_models_extended.loc[(best_models_extended['p'] <= 0.05) & 
                            (best_models_extended['p(amplitude)'] <= 0.05) & 
                            (best_models_extended['p(acrophase)'] <= 0.05) & 
                            (best_models_extended['p(mesor)'] <= 0.05) &
                            (best_models_extended['amplitude'] > 0.01), 'significant'] = 1

    return best_models_extended

def derivate_acrophase(best_models_per_day):
    
    acrophases_zt = numpy.array(best_models_per_day['acrophase_zt'].interpolate(method = 'spline', limit_direction = 'both', order = 3))
    acrophases_zt_smooth = savgol_filter(acrophases_zt, window_length = 10, polyorder = 3, mode = 'nearest')

    first_derivate = numpy.diff(acrophases_zt_smooth)

    plt.plot(first_derivate)
    plt.plot(acrophases_zt)
    plt.plot(acrophases_zt_smooth)

# file_activity = "F:\\github\\chronobiology_analysis\\protocols\\data_expto_5\\data\\1_control_dl\\control_dl_ale_animal_01.asc"
# file_temperature = "F:\\github\\chronobiology_analysis\\protocols\\data_expto_5\\data\\1_control_dl\\control_dl_temp_animal_01.asc"

# protocol = protocol('test', file_activity, file_temperature, 20, 18, 'DL', 'control')
# protocol.get_cosinor_df(time_shape = 'median', time_window = 24)
# pass