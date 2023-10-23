import numpy
import pandas
import pymice
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from math import ceil
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import MinMaxScaler
from io import StringIO
plt.ion()

class read_protocol():
    """
    Return a list of random ingredients as strings.

    :param name: Name of the protocol
    :type name: str
    :param file_activity: Path to the data file (can be a .asc if the type is generic)
    :type file_activity: str
    :param zt_0_time: Time of day that corresponds to ZT0 reference (can only be integers from 0 to 23)
    :type zt_0_time: int
    :param labels_dict: Dictionary with the cycle types, cycle days and test labels. The cycle types must be a list of strings
    with the cycle types (DD, LL, DL or LD), the cycle days must be a list of integers with the number of days of each cycle
    and the test labels must be a list of strings with the test labels (e.g. ['test1', 'test2', 'test3'])
    :type labels_dict: dict
    :param consider_first_day: Consider the first day of the experiment as a complete day, defaults to False
    :type consider_first_day: bool
    :param set_nans: List of integers with the indexes of the NaN values that will be set to 0, defaults to []
    :type set_nans: list
    
    :return: A protocol object
    :rtype: protocol object
    """
    def __init__(self, name, file, zt_0_time, labels_dict, type, consider_first_day = False, set_nans = numpy.array([])):
        """
        Constructor method
        """
        self.file = file                                                                                                # Define the path to the temperature file
        self.name = name                                                                                                # Define the name of the protocol
        if isinstance(zt_0_time, int) and zt_0_time >= 0 and zt_0_time <= 23:
            self.zt_0_time = zt_0_time                                                                                  # Define the time of day that corresponds to ZT 0
        else:
            raise ValueError('The ZT 0 time must be an integer between 0 and 23')
        if isinstance(consider_first_day, bool):
            self.consider_first_day = consider_first_day                                                                # Define if the first day of the experiment is considered as a complete day
        else:
            raise TypeError('The consider_first_day must be a boolean')

        if isinstance(labels_dict, dict):
            if 'cycle_types' not in labels_dict and 'cycle_days' not in labels_dict and 'test_labels' not in labels_dict:
                raise ValueError('The labels_dict must have the cycle_types, cycle_days and test_labels keys')
        else:
            raise TypeError('The labels_dict must be a dictionary')

        self.cycle_types = labels_dict['cycle_types'].copy()                                                            # Convert the cycle type to a number
        self.cycle_days = labels_dict['cycle_days'].copy()                                                              # Convert the cycle days to a number
        self.test_labels = labels_dict['test_labels'].copy()                                                            # Convert the test label to a number

        if not isinstance(self.cycle_days, list):
            raise ValueError('The cycle days must be a list of integers or a empty list')
        for cycle_type in self.cycle_types:
            if cycle_type not in ['DD', 'LL', 'DL', 'LD']:
                raise ValueError('The cycle type must be DD, LL, DL or LD')
        if len(self.cycle_types) != len(self.test_labels):
            raise ValueError('The cycle type and test label must have the same length')

        if isinstance(set_nans, numpy.ndarray):
            self.set_nans = set_nans.copy()                                                                             # Define if the NaN values will be set to 0 or not
        else:
            raise TypeError('The set_nans must be a numpy.ndarray')

        if type == 'er4000':
            self.read_asc_0()
        # elif type == 'ko':
        #     self.read_asc_1()
        elif type == 'intellicage':
            self.read_asc_2()
        elif type == 'generic':
            self.read_asc_3()
        else:
            raise ValueError('The type selected is not valid. Try intellicage or generic or see the documentation for more information')
            
    def read_asc_0(self):
        '''
        Function to read the activity file from the ER4000 system. The file must be a .asc file.
        '''        
        
        file = open(self.file, 'r')                                                                                     # Open the temperature file to be read
        self._lines = file.readlines()                                                                                  # Read the lines of the temperature file
        self._lines = [line.strip() for line in self._lines]                                                            # Remove the line breaks
        file.close()                                                                                                    # Close the temperature file

        date_type = "%m/%d/%y %H:%M:%S"
        start_date = datetime.strptime(self._lines[3].split("Start Date/Time ", 1)[1].strip(), date_type)               # Get the start date from the temperature file
        self.start_date = numpy.datetime64(start_date)                                                                  # Convert the start date to numpy datetime64
        end_date = datetime.strptime(self._lines[4].split("End Date/Time ", 1)[1].strip(), date_type)                   # Get the end date from the temperature file
        self.end_date = numpy.datetime64(end_date)                                                                      # Convert the end date to numpy datetime64
        sampling_freq = self._lines[9].split("Sampling Interval:", 1)[1].strip()                                        # Get the sampling frequency from the temperature file
        
        ts_1 = pandas.Timestamp(sampling_freq)                                                                          # Convert the sampling frequency to a pandas timestamp
        ts_0 = pandas.Timestamp('00:00:00')                                                                             # Set a pandas timestamp to 00:00:00 to calculate the sampling frequency
        period = ts_1 - ts_0                                                                                            # Calculate the sampling interval
        resolution = period.resolution_string                                                                           # Set the sampling interval for the object as a string (e.g. '1S' for 1 second)
        range = pandas.date_range(ts_0, ts_1, freq = resolution)                                                        # Create a pandas date range with the start date, end date and sampling interval
        self.sampling_interval = str(len(range) - 1) + resolution                                                       # Set the sampling interval for the object as a string (e.g. '1S' for 1 second)      
        self.sampling_frequency = self._get_sampling_frequency(self.sampling_interval)                                  # Set the sampling frequency for the object as a float (e.g. 1.0 for 1 Hz)

        data_string = self._lines[15:]
        data_string = [sample.replace('\t', ',') for sample in data_string]
        raw_data = [sample.replace(',', ' ', 1).split(",", 1) for sample in data_string]

        real_time = []
        data = []
        for sample in raw_data:
            try:
                real_time.append(datetime.strptime(sample[0], "%m/%d/%y %H:%M:%S"))
            except:
                real_time.append(datetime.strptime(sample[0], "%m/%d/%Y %H:%M:%S"))
        
            value = sample[1].replace(',', '.', 1)                                                                      # Replace the comma with a dot in the activity value
            value = value.split(',')[0]
            try:
                data.append(float(value))                                                                               # If it is possible, append the activity to the list
            except:
                if value == 'NaN.':
                    data.append(numpy.nan)
                else:
                    print("AO PAU AI GEEEENTE")

            #data.append(float(sample[1].replace(',', '.', 1)))

        real_time = [numpy.datetime64(date) for date in real_time]

        zt_correction = numpy.timedelta64(self.zt_0_time , 'h')
        time = [date - zt_correction for date in real_time]

        # Fill NaN method: use the mean of the 48 hours before and after the NaN value if it is possible,
        # otherwise use the mean of all the values before or after (depends if the n-th sample is on head or tail)
        # the NaN value in a sample of 24 hours
        
        if self.set_nans.size > 0:
            data = self._set_is_nan_indexes(self.set_nans, data)                                                        #This is a ugle hack to set the NaN values to acticity, because the activity file has 0 values instead of NaN

        self.is_nan = numpy.argwhere(numpy.isnan(data)).flatten()

        for n in self.is_nan:
            data[n] = 0
            # if n + round(48*60*60*self.sampling_frequency) > len(data) - 1:
            #     value = numpy.nanmean(data[n::-round(24*60*60*self.sampling_frequency)])
            #     data[n] = value
            # elif n - round(48*60*60*self.sampling_frequency) < 0:
            #     value = numpy.nanmean(data[n::round(24*60*60*self.sampling_frequency)])
            #     data[n] = value
            # else:
            #     next_value = data[n + round(48*60*60*self.sampling_frequency)]
            #     previus_value = data[n - round(48*60*60*self.sampling_frequency)]
            #     if numpy.isnan(next_value) == False and numpy.isnan(previus_value) == False:
            #         value = (previus_value + next_value)/2
            #         data[n] = value
            #     else:
            #         if n > len(data)/2:
            #             value = numpy.nanmean(data[n::-round(24*60*60*self.sampling_frequency)])
            #             data[n] = value
            #         else:
            #             value = numpy.nanmean(data[n::round(24*60*60*self.sampling_frequency)])
            #             data[n] = value

        self.data_column = pandas.DataFrame({'values': data}, index = time)
        self.data_column = self.data_column.interpolate(method= 'time')
        real_time_column = pandas.DataFrame({'real_time': real_time}, index = time)
        real_time_column = real_time_column.reset_index(drop=True)

        self.days = [str(row).split(" ")[0] for row in self.data_column.index]                                          # Get the day from the index

        if len(self.cycle_days) == 0:
            self.cycle_days = [len(numpy.unique(self.days))]

        self.labels = self._set_labels()                                                                                # Get if is night or not using the cycle type given as input (True if is night, False if is day)

        self.data = pandas.merge(self.data_column, self.labels, how='outer', right_index=True, left_index=True)         # Merge the activity, temperature and if is night or not

        self.data['real_date'] = real_time_column['real_time'].to_list()                                                # Get the real date from the index
        self.data['day'] = self.days                                                                                    # Add the day to the data

        if not self.consider_first_day:                                                                                 # If the first day is not to be considered
            self.data = self.data[self.data['day'] != self.data['day'].iloc[0]]                                         # Remove the first day from the data
            self.cycle_days[0] = self.cycle_days[0] - 1                                                                 # Remove the first day from the cycle days

    def read_asc_2(self):
        '''
        Function to read the activity file from the Intellicage system. The file must be a .asc file. Before executing 
        this function, the Intellicage output file must be converted to a .asc file using the intellicage_unwrapper 
        functions.
        '''

        file = open(self.file, 'r')                                                                                     # Open the temperature file to be read
        self._lines = file.readlines()                                                                                  # Read the lines of the temperature file
        self._lines = [line.strip() for line in self._lines]                                                            # Remove the line breaks
        file.close()                                                                                                    # Close the temperature file

        date_type = "%Y-%m-%d %H:%M:%S"        
        start_date = datetime.strptime(self._lines[1].split("Start date: ", 1)[1].strip(), date_type)                   # Get the start date from the temperature file
        self.start_date = numpy.datetime64(start_date)                                                                  # Convert the start date to numpy datetime64
        
        end_date = datetime.strptime(self._lines[2].split("End date: ", 1)[1].strip(), date_type)                       # Get the end date from the temperature file
        self.end_date = numpy.datetime64(end_date)                                                                      # Convert the end date to numpy datetime64
        
        self.sampling_interval = self._lines[3].split("Sampling interval: ", 1)[1].strip()                              # Set the sampling interval for the object as a string (e.g. '1S' for 1 second)
        self.sampling_frequency = self._get_sampling_frequency(self.sampling_interval)                                  # Set the sampling frequency for the object as a float (e.g. 1.0 for 1 Hz)

        data_string = self._lines[11:]
        tuplas = [string.split(',') for string in data_string]
        self.raw_data = pandas.DataFrame.from_records(tuplas)

        self.raw_data = self.raw_data.rename(columns={0: 'real_time', 1: 'duration', 2: 'value', 3: 'day'})
        self.raw_data['real_time'] = [datetime.strptime(date, date_type) for date in self.raw_data['real_time']]

        zt_correction = numpy.timedelta64(self.zt_0_time , 'h')
        self.raw_data.index = [date - zt_correction for date in self.raw_data['real_time']]

        self.data_column = pandas.DataFrame({'values': self.raw_data['value'].astype('float')}, index = self.raw_data.index)
        self.days = [str(row).split(" ")[0] for row in self.data_column.index]

        if len(self.cycle_days) == 0:
            self.cycle_days = [len(self.days.unique())]
        
        self.labels = self._set_labels()                                                                                # Get if is night or not using the cycle type given as input (True if is night, False if is day)

        self.data = pandas.merge(self.data_column, self.labels, how='outer', right_index=True, left_index=True)         # Merge the activity, temperature and if is night or not
        self.data['real_date'] = self.raw_data['real_time'].to_list()                                                   # Get the real date from the index
        self.data['day'] = self.days                                                                                    # Add the days to the dataframe

        if not self.consider_first_day:                                                                                 # If the first day is not to be considered
            self.data = self.data[self.data['day'] != self.data['day'].iloc[0]]                                         # Remove the first day from the data
            self.cycle_days[0] = self.cycle_days[0] - 1                                                                 # Remove the first day from the cycle days

    def read_asc_3(self):
        '''
        Function to read the data file from the generic file. The file must be a .asc file with the following format 
        (date,hour,activity) where the date is in the format mm/dd/yy, the hour is in the format hh:mm:ss and the
        activity is a float number.
        01/01/22,00:00:00,36.00
        01/01/22,00:30:00,41.00
        01/01/22,01:00:00,36.00
        '''
        
        file = open(self.file, 'r')                                                                                     # Open the temperature file to be read
        self._lines = file.readlines()                                                                                  # Read the lines of the temperature file
        self._lines = [line.strip() for line in self._lines]                                                            # Remove the line breaks
        file.close()                                                                                                    # Close the temperature file

        data_string = self._lines[0:]
        data_string = [sample.replace('\t', ',') for sample in data_string]
        raw_data = [sample.replace(',', ' ', 1).split(",", 1) for sample in data_string]

        real_time = []
        data = []
        for sample in raw_data:
            try:
                real_time.append(datetime.strptime(sample[0], "%m/%d/%y %H:%M:%S"))
            except:
                real_time.append(datetime.strptime(sample[0], "%m/%d/%Y %H:%M:%S"))
        
            value = sample[1]
            try:
                data.append(float(value))                                                                               # If it is possible, append the activity to the list
            except:
                if value == 'NaN.':
                    data.append(numpy.nan)
                else:
                    raise ValueError('The activity file has a value that is not a number or NaN')

        real_time = [numpy.datetime64(date) for date in real_time]

        self.start_date = real_time[0]                                                                                  # Set the start date of the data
        self.end_date = real_time[-1]                                                                                   # Set the end date of the data
        ts_1 = pandas.Timestamp(real_time[1])                                                                           # Convert the sampling frequency to a pandas timestamp
        ts_0 = pandas.Timestamp(real_time[0])                                                                           # Set a pandas timestamp to 00:00:00 to calculate the sampling frequency
        period = ts_1 - ts_0                                                                                            # Calculate the sampling interval        
        resolution = period.resolution_string                                                                           # Set the sampling interval for the object as a string (e.g. '1S' for 1 second)
        range = pandas.date_range(ts_0, ts_1, freq = resolution)                                                        # Create a pandas date range with the start date, end date and sampling interval
        self.sampling_interval = str(len(range) - 1) + resolution                                                       # Set the sampling interval for the object as a string (e.g. '1S' for 1 second)      
        self.sampling_frequency = self._get_sampling_frequency(self.sampling_interval)                                  # Set the sampling frequency for the object as a float (e.g. 1.0 for 1 Hz)

        zt_correction = numpy.timedelta64(self.zt_0_time , 'h')
        time = [date - zt_correction for date in real_time]

        # Fill NaN method: use the mean of the 48 hours before and after the NaN value if it is possible,
        # otherwise use the mean of all the values before or after (depends if the n-th sample is on head or tail)
        # the NaN value in a sample of 24 hours
        
        if len(self.set_nans) > 0:
            data = self._set_is_nan_indexes(self.set_nans, data)                                                        #This is a ugle hack to set the NaN values to acticity, because the activity file has 0 values instead of NaN

        self.is_nan = numpy.argwhere(numpy.isnan(data)).flatten()

        for n in self.is_nan:
            # data[n] = 0
            if n + round(48*60*60*self.sampling_frequency) > len(data) - 1:
                value = numpy.nanmean(data[n::-round(24*60*60*self.sampling_frequency)])
                data[n] = value
            elif n - round(48*60*60*self.sampling_frequency) < 0:
                value = numpy.nanmean(data[n::round(24*60*60*self.sampling_frequency)])
                data[n] = value
            else:
                next_value = data[n + round(48*60*60*self.sampling_frequency)]
                previus_value = data[n - round(48*60*60*self.sampling_frequency)]
                if numpy.isnan(next_value) == False and numpy.isnan(previus_value) == False:
                    value = (previus_value + next_value)/2
                    data[n] = value
                else:
                    if n > len(data)/2:
                        value = numpy.nanmean(data[n::-round(24*60*60*self.sampling_frequency)])
                        data[n] = value
                    else:
                        value = numpy.nanmean(data[n::round(24*60*60*self.sampling_frequency)])
                        data[n] = value

        self.data_column = pandas.DataFrame({'values': data}, index = time)
        self.data_column = self.data_column.interpolate(method= 'time')
        real_time_column = pandas.DataFrame({'real_time': real_time}, index = time)
        real_time_column = real_time_column.reset_index(drop=True)

        self.days = [str(row).split(" ")[0] for row in self.data_column.index]                                          # Get the day from the index

        if len(self.cycle_days) == 0:
            self.cycle_days = [len(numpy.unique(self.days))]

        self.labels = self._set_labels()                                                                                # Get if is night or not using the cycle type given as input (True if is night, False if is day)

        self.data = pandas.merge(self.data_column, self.labels, how='outer', right_index=True, left_index=True)         # Merge the activity, temperature and if is night or not

        self.data['real_date'] = real_time_column['real_time'].to_list()                                                # Get the real date from the index
        self.data['day'] = self.days                                                                                    # Add the day to the data

        if not self.consider_first_day:                                                                                 # If the first day is not to be considered
            self.data = self.data[self.data['day'] != self.data['day'].iloc[0]]                                         # Remove the first day from the data
            self.cycle_days[0] = self.cycle_days[0] - 1                                                                 # Remove the first day from the cycle days

    def _set_labels(self):
        """
        Get a bool list with True if the hour correspond to night (dark) period and False if it correspond to day
        (light) period.

        :return: pandas.DataFrame with is_night data
        :rtype: pandas.DataFrame
        """        
        unique_days = pandas.unique(self.days)
        
        if isinstance(self.cycle_days , list):
            number_of_days = sum(self.cycle_days)
            if len(self.cycle_types) != len(self.cycle_days):
                raise ValueError('The number of cycle types is not equal to the number of cycle days')
            elif len(self.cycle_types) != len(self.test_labels):
                raise ValueError('The number of cycle types is not equal to the number of test labels')
            elif len(self.cycle_days) != len(self.test_labels):
                raise ValueError('The number of cycle days is not equal to the number of test labels')
            if number_of_days != len(unique_days):
                raise ValueError('The number of days (' + str(number_of_days) + ') in cycle_days is not equal to the' + 
                                 'number of days (' + str(len(unique_days)) + ') in the data. If your ZT is not 0, try ' +
                                 'to add 1 to the first day in cycle_days')
            else:
                cycle_days = self.cycle_days
                cycle_types = self.cycle_types
                test_labels = self.test_labels
        else: 
            raise TypeError('The cycle_days must be a list of integers, a single integer or None')
            
        if isinstance(self.cycle_types , list):
            cycle_types = self.cycle_types
        elif isinstance(self.cycle_types , int):
            cycle_types = [self.cycle_types]
        else:
            raise TypeError('The cycle_types must be a list of integers or a single integer')
        
        if isinstance(self.test_labels , list):
            test_labels = self.test_labels
        elif isinstance(self.test_labels , str):
            test_labels = [self.test_labels]
        else:
            raise TypeError('The test_labels must be a list of strings or a single string')

        first_hour = numpy.datetime64(0, 'h')
        last_hour = first_hour + numpy.timedelta64(12, 'h')

        first_hour = pandas.Timestamp(first_hour).hour
        last_hour = pandas.Timestamp(last_hour).hour

        days_in_data = sorted(unique_days)

        last_day = 0
        is_night_list = []
        cycle_types_list = []
        test_labels_list = []

        for type, days, label in zip(cycle_types, cycle_days, test_labels):
            days_to_set_type = days_in_data[last_day:last_day + days]
            peace_of_data = self.data_column.loc[days_to_set_type[0]:days_to_set_type[-1]]

            if type == 'DD':
                is_night = [True]*len(peace_of_data)
            elif type == 'LL':
                is_night = [False]*len(peace_of_data)
            elif type == 'DL':
                is_night = []
                for date in peace_of_data.index.hour:
                    if date >= first_hour and date < last_hour:
                        is_night.append(True)
                    else:
                        is_night.append(False)
            elif type == 'LD':
                is_night = []
                for date in peace_of_data.index.hour:
                    if date >= first_hour and date < last_hour:
                        is_night.append(False)
                    else:
                        is_night.append(True)
            else:
                raise(ValueError("Cycle must be 'DD', 'LL', 'DL' or 'LD'"))
            last_day += days
            is_night_list.extend(is_night)
            cycle_types_list.extend([type]*len(is_night))
            test_labels_list.extend([label]*len(is_night))

        labels = pandas.DataFrame({'is_night': is_night_list, 'cycle_types': cycle_types_list, 
                                   'test_labels': test_labels_list}, index = self.data_column.index)

        return labels

    def _get_sampling_frequency(self, sampling_interval):
        """
        Convert sampling interval in seconds to sampling frequency. For example, if the sampling interval is 30 minutes,
        the sampling frequency is 1/(30*60).

        :param sampling_interval: Sampling interval of the data
        :type sampling_interval: str
        :return: Sampling frequency of the data
        :rtype: float
        """
        range = pandas.date_range(start = self.start_date, periods = 2, freq = sampling_interval)
        sampling_frequency = range[1] - range[0]
        frequency = 1/(sampling_frequency.total_seconds())

        return frequency

    def resample(self, new_sampling_interval, method = 'sum'):
        """
        Resample the data to a new sampling interval.

        :param new_sampling_interval: New sampling interval in pandas format (e.g. '30T' for 30 minutes)
        :type new_sampling_interval: str
        :param method: Method to resample the data. Use 'sum' to sum the values in the new sampling interval or 'last' to
        get the last value in the new sampling interval
        """
        if method == 'sum':
            resample_data = self.data['values'].resample(new_sampling_interval).sum()
        elif method == 'last':
            resample_data = self.data['values'].resample(new_sampling_interval).last()
        else:
            raise(ValueError("Method must be 'sum' or 'last'"))

        resample_is_night = self.data['is_night'].resample(new_sampling_interval).last()
        resample_cycle = self.data['cycle_types'].resample(new_sampling_interval).last()
        resample_test_labels = self.data['test_labels'].resample(new_sampling_interval).last()
        resample_real_date = self.data['real_date'].resample(new_sampling_interval).first()

        self.data = pandas.merge(resample_data, resample_is_night, how='outer', right_index=True, left_index=True)
        self.data = pandas.merge(self.data, resample_cycle, how='outer', right_index=True, left_index=True)
        self.data = pandas.merge(self.data, resample_test_labels, how='outer', right_index=True, left_index=True)
        self.data = pandas.merge(self.data, resample_real_date, how='outer', right_index=True, left_index=True)

        self.data['day'] = [str(row).split(" ")[0] for row in self.data.index]

        self.start_date = self.data.index[0]
        self.end_date = self.data.index[-1]
        self.sampling_interval = new_sampling_interval
        self.sampling_frequency = self._get_sampling_frequency(new_sampling_interval)

    def concat_protocols(self, protocol, method):
        """
        Concatenate two protocols. The protocols must have the same units of activity, temperature and sampling interval.
        The second protocol will be concatenated to the first one inplace.

        :param protocol: Protocol to concatenate
        :type protocol: Protocol
        :param method: Method to concatenate the data. Use 'sum' to sum the values in the new sampling interval or 
        'last' to get the last value in the new sampling interval
        :type method: str
        """
        if self.sampling_interval != protocol.sampling_interval:
            raise("Sampling interval is not the same")

        if self.test_labels[-1] == protocol.test_labels[0] and self.cycle_types[-1] == protocol.cycle_types[0]:
            self.cycle_days[-1] += protocol.cycle_days[0]
            self.cycle_days.extend(protocol.cycle_days[1:])
            self.cycle_types.extend(protocol.cycle_types[1:])
            self.test_labels.extend(protocol.test_labels[1:])
        else:
            self.cycle_days.extend(protocol.cycle_days)
            self.cycle_types.extend(protocol.cycle_types)
            self.test_labels.extend(protocol.test_labels)

        if self.start_date < protocol.start_date:
            if numpy.where(self.data.index == protocol.data.index[0])[0].size > 0:
                index = numpy.where(self.data.index == protocol.data.index[0])[0][0]
                for sample_1, sample_0 in enumerate(range(index, len(self.data.index))):                    
                    data_0 = self.data['values'][sample_0]
                    data_1 = protocol.data['values'][sample_1]
                    if method == 'sum':
                        if numpy.isnan(data_0) == False and numpy.isnan(data_1) == False:
                            self.data.iloc[sample_0, 0] = data_0 + data_1
                        elif numpy.isnan(data_0) == False and numpy.isnan(data_1) == True:
                            self.data.iloc[sample_0, 0] = data_0
                        elif numpy.isnan(data_0) == True and numpy.isnan(data_1) == False:
                            self.data.iloc[sample_0, 0] = data_1
                        else:
                            self.data.iloc[sample_0, 0] = numpy.nan
                    elif method == 'last':
                        if numpy.isnan(data_0) == False and numpy.isnan(data_1) == False:
                            self.data.iloc[sample_0, 1] = data_1
                        elif numpy.isnan(data_0) == False and numpy.isnan(data_1) == True:
                            self.data.iloc[sample_0, 1] = data_0
                        elif numpy.isnan(data_0) == True and numpy.isnan(data_1) == False:
                            self.data.iloc[sample_0, 1] = data_1
                        else:
                            self.data.iloc[sample_0, 1] = numpy.nan
                    else:
                        raise(ValueError("Method must be 'sum' or 'last'"))

                    is_night_0 = self.data['is_night'][sample_0]
                    is_night_1 = protocol.data['is_night'][sample_1]
                    if is_night_0 != is_night_1:
                        self.data.iloc[sample_0, 1] = is_night_1

                self.data = pandas.concat([self.data, protocol.data[len(self.data.index)-index:]])
            else:
                self.data = pandas.concat([self.data, protocol.data])
            self.end_date = protocol.end_date
        else:
            raise ValueError("The protocol that is being concatenated ocurred before the current protocol")

    def get_last_days_data(self, num_days, test_labels):
        '''
        Function to get the last n days of the data. The number of days must be equal or less than the number of days
        in the data. The test labels must be a list with the test labels to get the data from. The test labels must be
        in the data.

        :param num_days: Number of days to get the data from
        :type num_days: int
        :param test_labels: List with the test labels to get the data from
        :type test_labels: list
        '''
        if not isinstance(num_days, int) or not isinstance(test_labels, list):
            raise TypeError("Days and test labels must be lists.")

        for test_label in test_labels:
            if test_label not in self.data['test_label'].unique():
                raise ValueError("Test label not found in data.")

        new_data = pandas.DataFrame()

        for test_label in test_labels:
            test_label_data = self.data[self.data['test_label'] == test_label]
            list_of_days = sorted(test_label_data['day'].unique())
            real_date = [d.date() for d in test_label_data['real_date']]
            list_of_real = sorted(set(real_date))

            if num_days > len(list_of_days):
                raise ValueError("Number of days is greater than the number of days in the data.")

            day_selected = list_of_days[-(num_days+1):-1]
            print(day_selected)
            print(list_of_real[-(num_days+1):-1])

            day_data = test_label_data.query('day in @day_selected')
            #day_data = test_label_data[test_label_data['day'] == day_selected]
            new_data = pandas.concat([new_data, day_data])

        self.data = new_data

    def get_specific_days_data(self, days, test_labels):
        '''
        Function to get specific days of the data. The days must be a list with the days to get the data from and
        the test labels must be a list with the test labels to get the data from.

        :param days: List with the days to get the data from
        :type days: list
        :param test_labels: List with the test labels to get the data from
        :type test_labels: list
        '''
        if not isinstance(days, list) or not isinstance(test_labels, list):
            raise TypeError("Days and test labels must be lists.")

        if len(days) != len(test_labels):
            raise ValueError("Number of days must be equal to the number of test labels (one day for each test label).")

        for test_label in test_labels:
            if test_label not in self.data['test_label'].unique():
                raise ValueError("Test label not found in data.")

        new_data = pandas.DataFrame()

        for day, test_label in zip(days, test_labels):
            test_label_data = self.data[self.data['test_label'] == test_label]
            list_of_days = sorted(test_label_data['day'].unique())
            real_date = [d.date() for d in test_label_data['real_date']]
            list_of_real = sorted(set(real_date))
            if day < 0 or day > len(list_of_days):
                raise ValueError("Day not found in data.")
            elif day == 0:
                continue
            else:
                day_selected = list_of_days[day - 1]
                print(day_selected)
                print(list_of_real[day - 1])

            day_data = test_label_data[test_label_data['day'] == day_selected]
            new_data = pandas.concat([new_data, day_data])

        self.data = new_data

    def correct_labels(self):
        '''
        This function is used to correct labels that may be wrong after importing the data and using the protocol concatenation function. Generally, it will be used when the experimental protocol was performed by separating the files. In these cases, the exact time that the experimenter cut the record may not match the correct labels, since each file will be imported with a specific label.

        Example:

        +------------+----------+--------+-----------+-------+--------------------------+
        |    date    |   hour   | cycle  | is_night  | label |                          |
        +============+==========+========+===========+=======+==========================+
        | 01/01/2023 | 23:00:00 |   DL   |   False   |   A   |                          |
        +------------+----------+--------+-----------+-------+--------------------------+
        | 01/01/2023 | 23:30:00 |   DD   |   True    |   B   | <- This label is wrong   |
        +------------+----------+--------+-----------+-------+--------------------------+
        | 01/02/2023 | 00:00:00 |   DD   |   True    |   B   |                          |
        +------------+----------+--------+-----------+-------+--------------------------+
        | 01/02/2023 | 00:30:00 |   DD   |   True    |   B   |                          |
        +------------+----------+--------+-----------+-------+--------------------------+

        After using the correct_labels function, the table is updated as follows:

        +------------+----------+--------+-----------+-------+-----------------------------+
        |    date    |   hour   | cycle  | is_night  | label |                             |
        +============+==========+========+===========+=======+=============================+
        | 01/01/2023 | 23:00:00 |   DL   |   False   |   A   |                             |
        +------------+----------+--------+-----------+-------+-----------------------------+
        | 01/01/2023 | 23:30:00 |   DL   |   False   |   A   | <- This label is corrected  |
        +------------+----------+--------+-----------+-------+-----------------------------+
        | 01/02/2023 | 00:00:00 |   DD   |   True    |   B   |                             |
        +------------+----------+--------+-----------+-------+-----------------------------+
        | 01/02/2023 | 00:30:00 |   DD   |   True    |   B   |                             |
        +------------+----------+--------+-----------+-------+-----------------------------+

        :return: None
        '''
        labels = self.test_labels
        cycles = self.cycle_types
        labels = numpy.flip(labels, axis=0)
        cycles = numpy.flip(cycles, axis=0)
        for count, (label, cycle) in enumerate(zip(labels, cycles)):
            if count != len(labels) - 1:
                data = self.data[self.data['test_labels'] == label].copy()
                
                first_day_dif = data.index[0] - data.index[0].normalize()
                
                samples_per_day = (24*60*60)/(1/self.sampling_frequency)
                samples_day_dif = first_day_dif.total_seconds()/(1/self.sampling_frequency)

                if samples_day_dif <= samples_per_day/2:
                    index_to_change = self.data[self.data['test_labels'] == labels[count + 1]][-int(samples_day_dif):].index
                    self.data.loc[index_to_change,'test_labels'] = label
                    self.data.loc[index_to_change,'cycle_types'] = cycle
                else:
                    index_to_change = self.data[self.data['test_labels'] == label][0:int(samples_per_day - samples_day_dif)]['test_labels'].index
                    self.data.loc[index_to_change,'test_labels'] = labels[count + 1]
                    self.data.loc[index_to_change,'cycle_types'] = cycles[count + 1]
        
        count_days = list(self.data.groupby((self.data['test_labels'] != self.data['test_labels'].shift()).cumsum())['test_labels'].count())
        count_days = numpy.array(count_days)/self.sampling_frequency/60/60/24
        count_days = [ceil(x) for x in count_days]
        self.cycle_days = count_days


    def apply_filter(self, type = 'savgol', window = 5, order = 3, reverse = False):
        """
        Apply filters to the data. The filters available are Savitzky-Golay filter and moving average filter. The 
        Savitzky-Golay and moving average filter are a type of low-pass filter, particularly suited for smoothing noisy.
        (to reverse the filter, use the reverse parameter).

        :param type: Type of filter to apply: Savitzky-Golay ('savgol') filter or move average ('moving_average') filter, 
        defaults to 'savgol'.
        :type type: str
        :param window: The length of the filter window (i.e. the number of coefficients). Must be a positive odd integer,
        defaults to 5
        :type window: int
        :param order: The order of the polynomial used to fit the samples. Must be less then window size, defaults to 3
        :type order: int
        :param reverse: Revert the filter, defaults to False
        :type reverse: bool
        """
        if not isinstance(window, int) or not isinstance(order, int):
            raise TypeError("Window and order must be integers.")
        if window < 0 or order < 0:
            raise ValueError("Window and order must be positive.")
        if not isinstance(reverse, bool):
            raise TypeError("Reverse must be a boolean.")
        if not isinstance(type, str):
            raise TypeError("Type must be 'savgol' or 'moving_average.")

        if reverse == True:
            if hasattr(self, '_old_data'):
                self.data['values'] = self._old_data
            else:
                raise ValueError("No filter applied to revert.")
        else:
            if type == 'savgol':
                self._old_data = self.data['values'].copy()
                self.data['values'] = savgol_filter(self.data['values'], window, order)
            elif type == 'moving_average':
                self._old_data = self.data['values'].copy()
                self.data['values'] = uniform_filter1d(self.data['values'], window, mode = 'nearest')
            else:
                raise ValueError("Filter type must be 'savgol' or 'moving_average'.")

    def normalize_data(self, type = 'minmax', per_day = False):
        """
        Normalize the data using the z-score or minmax method. The z-score method subtracts the mean of the data and
        divides by the standard deviation. The minmax method subtracts the minimum value and divides by the difference
        between the maximum and minimum values.

        :param type: Type of normalization to apply, can be 'zscore' or 'minmax', defaults to 'minmax'
        :type type: str
        :param per_day: Normalize the data. If True, the normalization will be done per day, if False, the normalization
        will be done for the whole data, defaults to False
        :type per_day: bool
        """
        if not isinstance(type, str):
            raise TypeError("Type must be 'zscore' or 'minmax'.")
        if not isinstance(per_day, bool):
            raise TypeError("per_day must be a boolean.")
        
        if per_day == False:
            if type == 'zscore':
                values = self.data['values']
                self.data['values'] = (values - values.mean())/values.std()
            if type == 'minmax':
                values = self.data['values']
                scaler = MinMaxScaler()
                scaler.fit(values.values.reshape(-1,1))
                self.data['values'] = scaler.transform(values.values.reshape(-1,1))
            else:
                raise ValueError("Type must be 'zscore' or 'minmax'.")
        elif per_day == True:
            if type == 'zscore':
                for day in self.data['day'].unique():
                    data = self.data[self.data['day'] == day]
                    values = data['values']
                    self.data.loc[data.index, 'values'] = (values - values.mean())/values.std()
            if type == 'minmax':
                for day in self.data['day'].unique():
                    data = self.data[self.data['day'] == day]
                    values = data['values']
                    scaler = MinMaxScaler()
                    scaler.fit(values.values.reshape(-1,1))
                    self.data.loc[data.index, 'values'] = scaler.transform(values.values.reshape(-1,1))
            else:
                raise ValueError("Type must be 'zscore' or 'minmax'.")
        else:
            raise ValueError("per_day must be a boolean.")


    def specify_test_labels(self, number_of_days, test_labels, cycle_types):
        """
        Specify the test labels to be used in the analysis. This function can be used to set the test labels, the cycle 
        type after the data is imported. 
        
        :param number_of_days: Number of days to be used in each test label, the sum of the days must be equal to the
        number of days in the experiment
        :type number_of_days: list
        :param test_labels: Test labels to be used in the analysis, must be a list of strings with the same length of
        number_of_days
        :type test_labels: list
        :param cycle_types: Cycle types to be used in the analysis, must be a list of strings with the same length of
        number_of_days
        :type cycle_types: list
        """
        if not isinstance(number_of_days, list) or not isinstance(test_labels, list) or not isinstance(cycle_types, list):
            raise TypeError("Number of days, test labels and cycle types must be lists.")
        if len(number_of_days) != len(test_labels) or len(number_of_days) != len(cycle_types):
            raise ValueError("Number of days, test labels and cycle types must have the same length.")

        if sum(number_of_days) != len(self.data['day'].unique()):
            raise ValueError("The sum of the days must be equal to the number of days in the experiment.")

        number_of_days = numpy.cumsum(number_of_days)

        days_in_data = sorted(self.data['day'].unique())

        lower_limit = 0
        for number, label in zip(number_of_days, test_labels):
            days_to_change = days_in_data[lower_limit:number]
            self.data.loc[self.data['day'].isin(days_to_change), 'test_labels'] = label
            self.data.loc[self.data['day'].isin(days_to_change), 'cycle_types'] = cycle_types
            lower_limit = number     

    def get_is_nan_indexes(self):
        """
        Get a boolean array indicating if each data point is nan or not. The array is True if the data point is nan and
        False if the data point is not nan.

        :return: Boolean array indicating if the activity data is nan or not
        :rtype: numpy.ndarray
        """
        return self.is_nan

    def _set_is_nan_indexes(self, is_nan, data):
        """
        Set the boolean array indicating if each data point is nan or not.

        :param is_nan: Boolean array indicating if the activity data is nan or not
        :type is_nan: numpy.ndarray
        :param data: Experimental data
        :type data: numpy.ndarray

        :return: Experimental data with nan values where is_nan is True
        :rtype: numpy.ndarray
        """
        data = numpy.array(data)
        data[is_nan] = numpy.nan
        data = list(data)

        return data

    def delete_last_days(self, number_of_days):
        """
        Delete the last days of the data.

        :param number_of_days: Number of days to be deleted
        :type number_of_days: int
        """
        if not isinstance(number_of_days, int):
            raise TypeError("Number of days must be an integer.")

        days_to_delete = sorted(self.data['day'].unique())[-number_of_days:]
        self.data = self.data[~self.data['day'].isin(days_to_delete)]
        self.cycle_days[-1] = self.cycle_days[-1] - number_of_days

    def delete_first_days(self, number_of_days):
        """
        Delete the first days of the activity data.

        :param number_of_days: Number of days to be deleted
        :type number_of_days: int
        """
        if not isinstance(number_of_days, int):
            raise TypeError("Number of days must be an integer.")

        days_to_delete = sorted(self.data['day'].unique())[0:number_of_days]
        self.data = self.data[~self.data['day'].isin(days_to_delete)]
        self.cycle_days[0] = self.cycle_days[0] - number_of_days

    def delete_period(self, first_day_between, last_day_between, test_label):
        """
        Delete the days between the two parameters

        :param first_day_between: First day in the interval to be deleted
        :type number_of_days: int
        :param last_day_between: Last day in the interval to be deleted
        :type number_of_days: int
        """        
        if test_label not in self.test_labels:
            raise ValueError("The test_label select must be in the data") 
        if not isinstance(test_label, str):
            raise TypeError("test_label must be a string.")
        if not isinstance(first_day_between, int):
            raise TypeError("First day must be an integer.")
        if not isinstance(last_day_between, int):
            raise TypeError("Last day must be an integer.")

        number_of_days = last_day_between - first_day_between + 1
        index_test_label = self.test_labels.index(test_label)
        if number_of_days >= self.cycle_days[index_test_label]:
            print(number_of_days)
            print(self.cycle_days[index_test_label])
            raise ValueError("The period to be removed needs to be shorter than the total interval.")

        selected_data = self.data[self.data['test_labels'] == test_label] 
        days = sorted(selected_data['day'].unique())
        days_to_delete = days[first_day_between - 1:last_day_between]
        print(days_to_delete)
        
        first_day_to_delete = days_to_delete[0]

        self.new_data = self.data[~self.data['day'].isin(days_to_delete)]
        
        after_date = pandas.to_datetime(first_day_to_delete)
        self.new_data.index = self.new_data.index.where(self.new_data.index <= after_date, self.new_data.index - pandas.DateOffset(days=number_of_days))
        self.data = self.new_data

        self.cycle_days[index_test_label] = self.cycle_days[index_test_label] - number_of_days

    def get_cosinor_df(self, time_shape = 'continuous'):
        """
        Get the dataframe to be used in cosinor analysis (CosinorPy input). Each dataframe contains three columns: 
        "type" (protocol step label e.g 'Control'), "x" (time in hours) and "y" (activity or temperature). The "x" 
        columns can be 'continuous' (range from 0 to the total number of hours) or 'mean'/'median' (setted in clycles of
        24 hours). With the time_shape is setted to 'mean' or 'median', the data will be grouped by day and the mean or
        median will be calculated for each day.

        :param time_shape: Shape of the time variable. Set 'continuous' for a continuous variable or 'mean'/'median' for
        a clycic time columns, defaults to 'continuous'
        :type time_shape: str

        :return: Dataframes with activity and temperature data
        :rtype: pandas.DataFrame, pandas.DataFrame
        """
        if time_shape != 'continuous' and time_shape != 'mean' and time_shape != 'median':
            raise ValueError("Time shape must be 'continuous', 'median' or 'mean'")

        sampling_interval_hour = (1/self.sampling_frequency)/3600

        if time_shape == 'continuous':
            time_in_hour = numpy.arange(0, len(self.data.index)*sampling_interval_hour, sampling_interval_hour)
        else:
            time_in_hour = []
            for time in self.data.index:
                reference = time.replace(hour=0, minute=0, second=0, microsecond=0)
                time_in_hour.append((time - reference).total_seconds()/3600)

        protocol_df = pandas.DataFrame({'test': self.data['test_labels'], 'x': time_in_hour, 'y': self.data['values']})

        return protocol_df

    def save_data(self, save_file):
        """
        Save the data of the protocol in a csv file

        :param save_file: Path to save the file
        :type save_file: str
        """
        self.data.to_csv(save_file + '.csv')

def intellicage_unwrapper(files, name_to_save = '', sampling_interval = '1H'):
    '''
    This function is used to unwrap the data from the Intellicage system, which is saved in a compact format. The data
    is extraxted using the pymice package and the data from each animal is saved in a separated txt file. An important
    subfunction used in this function is the visits_by_intervals, which is used to get the visits in a specific interval
    of time, because the Intellicage system saves the visits with irregular intervals.

    :param files: List with the files to be unwrapped
    :type files: list
    :param name_to_save: Folder to save the data, if not specified, the data will be saved in the same folder as the
    files in a folder called 'data_unwrapped'
    :type name_to_save: str
    :param sampling_interval: Sampling interval to get the visits (output sampling interval), defaults to '1H'
    :type sampling_interval: str
    '''
    if not isinstance(files, list):
        raise TypeError("Files must be a list of files.")
    if not isinstance(name_to_save, str):
        raise TypeError("The folder to save the data must be a string.")
    if not isinstance(sampling_interval, str):
        raise TypeError("Sampling interval must be a string (e.g. '1H', '30T').")

    if name_to_save == '':
        root_folder = os.path.dirname(files[0])
        name_to_save = root_folder + '\\data_unwrapped'
    else:
        root_folder = os.path.dirname(files[0])
        name_to_save = root_folder + '\\' + name_to_save

    visits_df = {}
    for count, file in enumerate(files):
        data_raw = pymice.Loader(file, getNp=True, getLog=True, getEnv=True, getHw=True, verbose=False)

        if count == 0:
            start_date = data_raw.getStart().replace(hour=0, minute=0, second=0, microsecond=0)           
        if count == len(files) - 1:
            end_date = data_raw.getEnd().replace(hour=0, minute=0, second=0, microsecond=0)

        animals = sorted(list(data_raw.getAnimal()))
        for animal in animals:
            if animal not in visits_df.keys():
                visits_df[animal] = {}
                visits_df[animal]['file_to_save'] = name_to_save + '_' + animal.replace(' ', '_').lower() + '.txt'
                visits_df[animal]['data'] = pandas.DataFrame(columns = ['corner', 'duration_seconds', 'duration_date', 'visit_start', 'visit_end'])
            
            visits = data_raw.getVisits(order = 'Start', mice = animal)
            visits_each_animal = pandas.DataFrame(columns = ['corner', 'duration_seconds', 'duration_date', 'visit_start', 'visit_end'])
            for count, visit in enumerate(visits):              
                if count == 0:    
                    visits_df[animal]['tag'] = str(list(visit.Animal.Tag)[0])
                    visits_df[animal]['name'] = str(visit.Animal.Name)
                    visits_df[animal]['sex'] = str(visit.Animal.Sex)
                    visits_df[animal]['cage'] = str(int(visit.Cage))
                else:
                    if visits_df[animal]['tag'] != str(list(visit.Animal.Tag)[0]):
                        raise ValueError('The animal tag is not consistent')
                    if visits_df[animal]['name'] != str(visit.Animal.Name):
                        raise ValueError('The animal name is not consistent')
                    if visits_df[animal]['sex'] != str(visit.Animal.Sex):
                        raise ValueError('The animal sex is not consistent')
                    if visits_df[animal]['cage'] != str(int(visit.Cage)):
                        raise ValueError('The animal cage is not consistent')
    
                visits_each_animal.loc[count] = [int(visit.Corner), visit.Duration.total_seconds(), visit.Duration, visit.Start, visit.End]
            
            visits_df[animal]['data'] = pandas.concat([visits_df[animal]['data'], visits_each_animal], axis = 0)

    for animal in visits_df.keys():
        visits_by_intervals = _visits_by_intervals(visits_df[animal]['data'], start_date, end_date, sampling_interval = sampling_interval)

        if visits_by_intervals is not None:
            with open(visits_df[animal]['file_to_save'], 'w') as save:
                save.write('Description:\n')
                save.write('Start date: ' + str(start_date) + '\n')
                save.write('End date: ' + str(end_date) + '\n')
                save.write('Sampling interval: ' + sampling_interval + '\n')
                save.write('Animal: ' + animal + '\n')
                save.write('Tag: ' + visits_df[animal]['tag'] + '\n')
                save.write('Sex: ' + visits_df[animal]['sex'] + '\n')
                save.write('Cage: ' + str(visits_df[animal]['cage']) + '\n')
                save.write('\nData:\n')
                save.write(', '.join(list(visits_by_intervals.columns)) + '\n')
                visits_by_intervals.to_csv(save, header = False, index = False, lineterminator = '\n')

            print('File saved in ' + visits_df[animal]['file_to_save'])
        else:
            print('No data for ' + animal)

def _visits_by_intervals(visits_df, start_date, end_date, sampling_interval = '1H'):
    '''
    This function is used to get the visits in a specific interval of time, because the Intellicage system saves the
    visits with irregular intervals.

    :param visits_df: Dataframe with the visits
    :type visits_df: pandas.DataFrame
    :param start_date: Start date of the data
    :type start_date: datetime.datetime
    :param end_date: End date of the data
    :type end_date: datetime.datetime
    :param sampling_interval: Sampling interval to get the visits (output sampling interval), defaults to '1H'
    :type sampling_interval: str

    :return: Dataframe with the visits in the specified interval
    :rtype: pandas.DataFrame
    ''' 
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = end_date + timedelta(days = 1)
    date_range = pandas.date_range(start_date, end_date, freq = sampling_interval)
    date_range = pandas.DataFrame({'interval_start': date_range[:-1], 'interval_end': date_range[1:]})
        
    duration_per_range = []
    entries_per_range = []

    if len(visits_df) >= 1:
        for _, row in date_range.iterrows():

            entries_per_range.append(0)
            contained_visits_df = visits_df.loc[(visits_df['visit_start'] >= row['interval_start']) & (visits_df['visit_end'] < row['interval_end'])]['duration_date']
            entries_per_range[-1] += len(contained_visits_df)
            contained_visits_df = contained_visits_df.sum()
            edge_visits_df_0 = (visits_df.loc[(visits_df['visit_start'] < row['interval_start']) & (visits_df['visit_end'] < row['interval_end']) & (visits_df['visit_end'] >= row['interval_start'])]['visit_end'] - row['interval_start']).sum()
            edge_visits_df_1 = row['interval_end'] - visits_df.loc[(visits_df['visit_start'] >= row['interval_start']) & (visits_df['visit_start'] < row['interval_end']) & (visits_df['visit_end'] >= row['interval_end'])]['visit_start']
            entries_per_range[-1] += len(edge_visits_df_1)
            edge_visits_df_1 = edge_visits_df_1.sum()
            contains_visits_df = len(visits_df.loc[(visits_df['visit_start'] < row['interval_start']) & (visits_df['visit_end'] >= row['interval_end'])])*(row['interval_end'] - row['interval_start'])
            duration_per_range.append((contained_visits_df + edge_visits_df_0 + edge_visits_df_1 + contains_visits_df).total_seconds())

        visits_by_intervals = pandas.DataFrame({'date': date_range['interval_start'], 'duration': duration_per_range, 'values': entries_per_range})
        visits_by_intervals['day'] = [str(row).split(" ")[0] for row in visits_by_intervals['date']]
    else:
        visits_by_intervals = None

    return visits_by_intervals

