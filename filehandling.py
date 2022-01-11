

import os
import pandas as pd
import numpy as np
import re
import datetime
"""
This is a test, to test git pushes - Alex 2022-01-11
"""


def ensure_dir_exists(dir_name):
    """creates directory if not already existing"""    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
def dat_to_pd(filepath):
    """reads typical LAP measurements output file and returns it as pandas dataframe"""  
    
    return(pd.read_csv(filepath,header=0,delimiter='\t'))
    
def spectrum_to_pd(filepath):
    """reads typical LAP measurements Spectrum output file and returns it as pandas dataframe"""  
    spec_df=pd.read_csv(filepath,names=['wavelength','counts'],skiprows=4,delimiter='\t')
    spec_df['wavelength']=spec_df['wavelength']*1e-9
    return(spec_df)
    
def get_calibration_function(path,regex):
    
    """gets a calibration cal_func(wavelength) by which counts should be
    multiplied for calibration using the scipy interp1d method"""
    
    from scipy.interpolate import interp1d
    regex_condition = re.compile(regex)
    for item in os.listdir(path):
        if regex_condition.match(item):
            lamb, cal_factor, sensitivity = np.loadtxt(path+'/'+item,skiprows=1).T
            cal_func = interp1d(lamb,cal_factor)
    return cal_func
        
class dat_to_object():
    """old, use dat_to_pd instead"""  
    def __init__(self,filepath,comment_discriminator='#'):
        self.filepath=filepath
        try:
            self.load_with_delimiter('\t')
        except:
            try:
                self.load_with_delimiter(',')
            except:
                try:
                    self.load_with_delimiter(' ')
                except ValueError:
                    print("could not read datafile with filepath: "+filepath)
    def load_with_delimiter(self,delimiter):
        file = open(self.filepath,'r')
        filedata = file.readlines()
        self.headers = filedata[0].split(delimiter)
        ## remeove \n at end of headers (artefact of saving textfiles with csv.writer
        #print(self.headers)
        for i in range(len(self.headers)):
        
            if self.headers[i].endswith('\n'):
                self.headers[i] = self.headers[i][:-1]

        ## lade Daten in dictionary
        self.data = {}
        for i in range(len(self.headers)):
            self.data[self.headers[i]]=[]

        for line in filedata[1:]:
            lspl=line.split(delimiter)
        
            for i in range(len(self.headers)):
                self.data[self.headers[i]].append(float(lspl[i]))
                
def float_from_string(string,start='',end=''):
    """extracts a float from a string identified by start and end string"""
    regex = start+'(\d+\.*\d*)'+end
    print('regex:',regex)
    extracted=re.findall(regex,string)
    if len(extracted) ==0:
        raise Exception('regex identifier '+regex+' was not found in string '+string)
    if len(extracted) >1:
        raise Exception('regex identifier '+regex+' was found more than once in '+string+':',extracted)
    return(float(extracted[0]))
    
                
def data_from_directory(path,read_regex='', read_function=spectrum_to_pd, var_strings=[], var_regex_dict={}):
    """loads all files in path resulting in a pd dataframe of descriptions and dataframes. 
    only files with 'read_regex' in filename are included.
    'read_function' shall return a pandas dataframe with filepath as argument. 
    With 'start_strings','end_strings', variables with 'extractor_names' are 
    extracted from filenames and added to final dataframe
    """
    print(path)
    path_list=[]
    modify_time_list=[]
    df_list=[]
    var_dict={}
    for var_string in var_strings:
        var_dict[var_string]=[]
    for key in var_regex_dict.keys():
        var_dict[key]=[]
    
    for filename in os.listdir(path):
        if len(re.findall(read_regex,filename))>0:
            print(filename)
            filepath=path+'/'+filename
            modify_time_list.append(datetime.datetime.fromtimestamp(os.stat(filepath)[8])) ## os.stat()[8] gets the modify time
            path_list.append(filepath)
            df_list.append(read_function(filepath))
            for var_string in var_strings:
                var_val=re.findall(var_string+'\D*(\d+\.?\d*)',filename)
                if len(var_val)==0:
                    raise Exception('var_string "'+var_string+'" could not be found with value in "'+filename+'"')
                else:
                    var_val=float(var_val[0])
                var_dict[var_string].append(var_val)
            for key,var_regex in var_regex_dict.items():
                var_val=re.findall(var_regex,filename)
                if len(var_val)==0:
                    raise Exception('var_regex "'+var_regex+'" could not be found with value in "'+filename+'"')
                else:
                    var_val=float(var_val[0])
                var_dict[key].append(var_val)
    data_dict={'filepath':path_list,'modify_time':modify_time_list,'data':df_list}
    data_dict={**data_dict,**var_dict}
    data=pd.DataFrame.from_dict(data_dict)
    data=data.sort_values('modify_time').reset_index(drop=True)
    return data

if __name__ == "__main__":
    ## do test cases of all functions
    ensure_dir_exists('dummy_directory')
    data_object=dat_to_object('test_data/LAP_Measurment_output.dat')
    data_object=dat_to_object('test_data/space_separated.dat')
    dataframe=dat_to_pd('test_data/LAP_Measurment_output.dat')
    spec_data=spectrum_to_pd('test_data/spectrum.txt')
    data=data_from_directory('test_data',read_regex='spectrum_30s',
                              read_function=spectrum_to_pd,var_strings=['V_Piezo','V_SMU'])
    print(data) 
    input('test finished, press Enter to quit')
    
    