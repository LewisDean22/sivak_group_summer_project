# -*- coding: utf-8 -*-
"""
This script can be used once to set-up the necessary directories for
these simulations

@author: Lewis Dean - lewis.dean@manchester.student.ac.uk
"""
import os
  
DIRECTORIES = ("data_files", "plots")

def main(directories=DIRECTORIES):
  
    parent_directory = os.getcwd()
      
    for directory in directories:
        path = os.path.join(parent_directory, directory)
        os.mkdir(path)
        
    return None


if __name__ == '__main__':
    main()
