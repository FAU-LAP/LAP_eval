# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 19:08:20 2016

@author: jo28dohe
"""

import numpy as np
import csv
import os

class opendat:

    def __init__(self,filename):
        try:
            file = open(filename,'r')
            filedata = file.readlines()
            self.headers = filedata[0].split('\t')
            ## remeove \n at end of headers (artefact of saving textfiles with csv.writer
            #print(self.headers)
            for i in range(len(self.headers)):
            
                if self.headers[i].endswith('\n'):
                    self.headers[i] = self.headers[i][:-1]

            ## lade Daten in dictionary
            self.data = {}
            names = []
            for i in range(len(self.headers)):
                self.data[self.headers[i]]=[]

            for line in filedata[1:]:
                lspl=line.split('\t')
            
                for i in range(len(self.headers)):
                    self.data[self.headers[i]].append(float(lspl[i]))
        except:
            try:
                file = open(filename,'r')
                filedata = file.readlines()
                self.headers = filedata[0].split(',')
                ## remeove \n at end of headers (artefact of saving textfiles with csv.writer
                #print(self.headers)
                for i in range(len(self.headers)):
                
                    if self.headers[i].endswith('\n'):
                        self.headers[i] = self.headers[i][:-1]
    
                ## lade Daten in dictionary
                self.data = {}
                names = []
                for i in range(len(self.headers)):
                    self.data[self.headers[i]]=[]
    
                for line in filedata[1:]:
                    lspl=line.split(',')
                
                    for i in range(len(self.headers)):
                        self.data[self.headers[i]].append(float(lspl[i]))
            except:
                file = open(filename,'r')
                filedata = file.readlines()
                self.headers = filedata[0].split(' ')
                ## remeove \n at end of headers (artefact of saving textfiles with csv.writer
                #print(self.headers)
                for i in range(len(self.headers)):
                
                    if self.headers[i].endswith('\n'):
                        self.headers[i] = self.headers[i][:-1]
    
                ## lade Daten in dictionary
                self.data = {}
                names = []
                for i in range(len(self.headers)):
                    self.data[self.headers[i]]=[]
    
                for line in filedata[1:]:
                    lspl=line.split(' ')
                
                    for i in range(len(self.headers)):
                        self.data[self.headers[i]].append(float(lspl[i]))



class savedat:

    def __init__(self,headers,lists,filename):
        
        array=np.array(lists).transpose()
        print('bin in savedat,',headers)
        print(filename)

        csvfile = open(filename,'w',newline='')
        print('csvfile geöffnet')
        writer = csv.writer(csvfile,delimiter='\t')
        writer.writerow(headers)
        
        for i in range(len(lists[0])):
            writer.writerow(array[i])
        csvfile.close()


class filenames:
    def __init__(self,path):
        self.path = path
        if self.path.endswith('.txt') or self.path.endswith('.dat'):
            self.filepaths=[self.path]
        else:
            self.filepaths=[os.path.join(dirpath, f)
                for dirpath, dirnames, files in os.walk(self.path)
                for f in files if (f.endswith('.dat') or f.endswith('.txt')) ]
        
    def show(self):      
        for names in self.filepaths:
            print(names)

    def show_time(self):
        for path in self.filepaths:
            print(os.stat(path)[8])

    def sort_by_modify_time(self):
        self.filepaths=sorted(self.filepaths, key=lambda fpath: os.stat(fpath)[8])

    def sort_by_nametime(self):
        self.filepaths=sorted(self.filepaths, key=lambda fpath: float(fpath.split('.txt')[0].split('=')[-1]))

    def sort_by_V_Piezo(self):
        try:
            self.filepaths=sorted(self.filepaths, key=lambda fpath: float(fpath.split('V_Piezo=')[1].split('_')[0 ]))
        except:
            self.filepaths=sorted(self.filepaths, key=lambda fpath: float(fpath.split('V_Piezo_')[1].split('_')[0 ]))
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

