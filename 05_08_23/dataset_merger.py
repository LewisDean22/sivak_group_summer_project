# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 23:12:22 2023

@author: lewis
"""

import numpy as np

FILE_DIRECTORY = "data_files/"

def data_file_properties(filename, directory=FILE_DIRECTORY):
    
    file_array = np.genfromtxt(directory + filename,
                               delimiter=',', names=True)
    header = file_array.dtype.names
    data = np.array([file_array[name] for name in header])
    data = data.transpose()
    
    return np.array(header), data


def merge_data(file_list, merged_file_name, directory=FILE_DIRECTORY):
    
    first_file = file_list[0]
    header, data = data_file_properties(first_file)
    
    merged_data = data
    for file in file_list[1:]:
        file_header, file_data = data_file_properties(file)
        merged_data = np.vstack((merged_data, file_data))
    
    sorted_indices = np.argsort(merged_data[:, 0])
    sorted_merged_data = merged_data[sorted_indices]
    merged_file = np.vstack((header, sorted_merged_data))
        
    np.savetxt(directory + merged_file_name, merged_file, delimiter=',',
               fmt='%s')
    
    return None
    