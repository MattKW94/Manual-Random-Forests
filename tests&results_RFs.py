# Testing
# Followed by results for analysis in R

import os
#os.chdir('H:/My Documents')
import csv # for writing test cases to csv
import methods_461_52315235_RFs as work

# data: https://archive.ics.uci.edu/ml/datasets/HIV-1+protease+cleavage

###################### Reading in data

data = [] #Raw data
DataFile =\
open("746Data.txt", "r")

while True:
    theline = DataFile.readline()
    if len(theline) == 0:
        break
    readData = theline.split(",")
    temp = []
    # '\n's are removed
    if readData[1][0] == '1':
        temp.append(1)
    else:
        temp.append(0) # actually -1s in the data
    for i in range(len(readData[0])):
        temp.append(readData[0][i])
    data.append(temp)

DataFile.close()

## data for lib_rf

x_data_orig = []
y_data = []

for entry in data:
    x_data_orig.append(entry[1:])
    y_data.append(entry[0])
    
# data is quite well balanced. 402 positive cases, 344 negative cases
    
# One hot encoding

# col_values to be appended for each one hot encoding entry
# entry [0] is the variable that one hot coding will signify if a data entry
# has or not
# [1] is the column number this will be in the new data
# [2] is the relevant column from the original data
        
col_values = []
new_col_number = 0 # initialising value

for i in range(0,len(x_data_orig[0])):
    col_values.append([])
    temp = []
    for j in x_data_orig:
        if j[i] not in temp:
            temp.append(j[i])
            col_values[i].append([j[i]])
    for x in col_values[i]:
        x.append(new_col_number)
        x.append(i)
        new_col_number += 1

total_cols_needed = new_col_number # total cols needed for one hot encoding

x_data = []

for i in range(0,len(x_data_orig)):
    x_data.append([0] * total_cols_needed)
    for j in range (0, len(x_data_orig[0])):
        for k in range (0, len(col_values[j])):
            if x_data_orig[i][col_values[j][k][2]] == col_values[j][k][0]:
                x_data[i][col_values[j][k][1]] = 1

###################### Tests
                
class Lib_DT_Case(work.Lib_DT):
    
    def test_stats(self):
        return self.accuracy

class Man_DT_Case(work.Man_DT):
    
    def test_stats(self):
        return self.accuracy
    
class Lib_RF_Case(work.Lib_RF):
    
    def test_stats(self):
        return self.accuracy
    
    def save_results(self, test_name):
        
        import time
        self.time = time.time() - self.start_time

        with open('python_results_1.csv', 'a', newline='') as m:
            thewriter = csv.writer(m)
            
            thewriter.writerow([test_name, self.tp, self.tn, self.fp, self.fn,\
                                self.accuracy, self.precision,\
                                self.recall, self.f1_score, self.time])
    
    def save_results2(self, test_name):
        
        import time
        self.time = time.time() - self.start_time

        with open('python_results_2.csv', 'a', newline='') as m:
            thewriter = csv.writer(m)
            
            thewriter.writerow([test_name, self.for_no, self.tp, self.tn,\
                                self.fp, self.fn, self.accuracy,\
                                self.precision, self.recall, self.f1_score,\
                                self.time])
    
    def save_results3(self, test_name):
        
        import time
        self.time = time.time() - self.start_time

        with open('python_results_3.csv', 'a', newline='') as m:
            thewriter = csv.writer(m)
            
            thewriter.writerow([test_name, self.layers, self.tp, self.tn,\
                                self.fp, self.fn, self.accuracy,\
                                self.precision, self.recall, self.f1_score,\
                                self.time])
    
    def save_ROC_data(self, test_name):
        
        with open('python_ROC_Library2.csv', 'w', newline='') as m:
            thewriter = csv.writer(m)
            
            thewriter.writerow(['Test Name', 'Number of Forests',\
                                'Threshold', 'TP', 'TN', 'FP', 'FN',\
                                'TP Rate', 'FP Rate'])
        
        for i in range(0, len(self.thresholds)):
            
            with open('python_ROC_Library2.csv', 'a', newline='') as m:
                thewriter = csv.writer(m)
                
                thewriter.writerow([test_name, self.for_no,\
                                    self.thresholds[i], self.tp, self.tn,\
                                    self.fp, self.fn, self.tp_rates[i],\
                                    self.fp_rates[i]])
    
class Man_RF_Case(work.Man_RF):
    
    def test_stats(self):
        return self.bag_data, self.col_values, self.for_probs,\
    self.temp_probs, self.temp_mean_prob, self.preds
    
    def save_results(self, test_name):
    
        import time
        self.time = time.time() - self.start_time
    
        with open('python_results_1.csv', 'a', newline='') as m:
            thewriter = csv.writer(m)
            
            thewriter.writerow([test_name, self.tp, self.tn, self.fp, self.fn,\
                                self.accuracy, self.precision,\
                                self.recall, self.f1_score, self.time])
    
    def save_results2(self, test_name):
    
        import time
        self.time = time.time() - self.start_time
    
        with open('python_results_2.csv', 'a', newline='') as m:
            thewriter = csv.writer(m)
            
            thewriter.writerow([test_name, self.for_no, self.tp, self.tn,\
                                self.fp, self.fn, self.accuracy,\
                                self.precision, self.recall, self.f1_score,\
                                self.time])
    
    def save_results3(self, test_name):
    
        import time
        self.time = time.time() - self.start_time
    
        with open('python_results_3.csv', 'a', newline='') as m:
            thewriter = csv.writer(m)
            
            thewriter.writerow([test_name, self.layers, self.tp, self.tn,\
                                self.fp, self.fn, self.accuracy,\
                                self.precision, self.recall, self.f1_score,\
                                self.time])
    
    def save_ROC_data(self, test_name):
        
        with open('python_ROC_Manual2.csv', 'w', newline='') as m:
            thewriter = csv.writer(m)
            
            thewriter.writerow(['Test Name', 'Number of Forests',\
                                'Threshold', 'TP', 'TN', 'FP', 'FN',\
                                'TP Rate', 'FP Rate'])
        
        thresholds = [0] # 0 as the starting threshold
        
        for i in self.probs:
            if i not in thresholds:
                thresholds.append(i)
        thresholds.sort()
        
        for thresh in thresholds:
            self.results(thresh)
            self.metrics()
            
            # ROC curve looks at positive classes
            
            self.tp_rate = self.tp / (self.tp + self.fn)
            self.fp_rate = self.fp / (self.fp + self.tn)
            
            with open('python_ROC_Manual2.csv', 'a', newline='') as m:
                thewriter = csv.writer(m)
                
                thewriter.writerow([test_name, self.for_no, thresh, self.tp,\
                                    self.tn, self.fp, self.fn, self.tp_rate,\
                                    self.fp_rate])
            

# Decision Tree manual testing vs SK Learn

lib_dt = Lib_DT_Case(layers=2, min_items=10, min_imp_dec=0.01)
lib_dt.train_test_split(x_data, y_data, test_prop=0.2, shuffle_state=False)
lib_dt.model_fit()
lib_dt.results()
lib_dt.metrics()
print('Test case 1 library accuracy is: ' + str(lib_dt.test_stats()))

man_dt = Man_DT_Case(data, test_prop=0.2, layers=2, min_items=10,\
                     min_imp_dec=0.01, shuffle_state=False)
man_dt.tree_fit()
man_dt.tree_results()
man_dt.metrics()
print('Test case 1 manual accuracy is: ' + str(man_dt.test_stats()))

lib_dt = Lib_DT_Case(2, 2, 0.4)
lib_dt.train_test_split(x_data, y_data, 0.3, False)
lib_dt.model_fit()
lib_dt.results()
lib_dt.metrics()
print('Test case 2 library accuracy is: ' + str(lib_dt.test_stats()))

man_dt = Man_DT_Case(data, 0.3, 2, 2, 0.4, False)
man_dt.tree_fit()
man_dt.tree_results()
man_dt.metrics()
print('Test case 2 manual accuracy is: ' + str(man_dt.test_stats()))

lib_dt = Lib_DT_Case(4, 8, 0.02)
lib_dt.train_test_split(x_data, y_data, 0.35, False)
lib_dt.model_fit()
lib_dt.results()
lib_dt.metrics()
print('Test case 3 library accuracy is: ' + str(lib_dt.test_stats()))

man_dt = Man_DT_Case(data, 0.35, 4, 8, 0.02, False)
man_dt.tree_fit()
man_dt.tree_results()
man_dt.metrics()
print('Test case 3 manual accuracy is: ' + str(man_dt.test_stats()))

lib_dt = Lib_DT_Case(7, 11, 0.02)
lib_dt.train_test_split(x_data, y_data, 0.15, False)
lib_dt.model_fit()
lib_dt.results()
lib_dt.metrics()
print('Test case 4 library accuracy is: ' + str(lib_dt.test_stats()))

man_dt = Man_DT_Case(data, 0.15, 7, 11, 0.02, False)
man_dt.tree_fit()
man_dt.tree_results()
man_dt.metrics()
print('Test case 4 manual accuracy is: ' + str(man_dt.test_stats()))

# RF test cases

man_rf = Man_RF_Case(data, 5, 0.2, 2, 8, 0.02, True)
man_rf.model_fit()
man_rf.results()
man_rf.metrics()
print('RF Test 1')
print('bag data for last 5 items:')
print(man_rf.test_stats()[0][-5:]) # bag data for last 5 items
print('column values to use:')
print(man_rf.test_stats()[1]) # col values
print('predictions of 1st forest for last 5 test items:')
print(man_rf.test_stats()[2][0][-5:]) # 1st for_preds for last 5 items
print('predictions of 2nd forest for last 5 test items:')
print(man_rf.test_stats()[2][1][-5:]) # 2nd for_preds for last 5 items
print('temporary probabilities stored for latest item:')
print(man_rf.test_stats()[3]) # temp_probs for latest item
print('temporary mean probability stored for latest item:')
print(man_rf.test_stats()[4]) # temp mean prob for latest item
print('classes predicted for last 5 items:')
print(man_rf.test_stats()[5][-5:]) # predictions for last 5 items

man_rf = Man_RF_Case(data, 5, 0.2, 2, 8, 0.02, True)
man_rf.model_fit()
man_rf.results()
man_rf.metrics()
print('RF Test 2')
print('bag data for last 5 items:')
print(man_rf.test_stats()[0][-5:]) # bag data for last 5 items
print('column values to use:')
print(man_rf.test_stats()[1]) # col values
print('predictions of 1st forest for last 5 test items:')
print(man_rf.test_stats()[2][0][-5:]) # 1st for_preds for last 5 items
print('predictions of 2nd forest for last 5 test items:')
print(man_rf.test_stats()[2][1][-5:]) # 2nd for_preds for last 5 items
print('temporary probabilities stored for latest item:')
print(man_rf.test_stats()[3]) # temp_probs for latest item
print('temporary mean probability stored for latest item:')
print(man_rf.test_stats()[4]) # temp mean prob for latest item
print('classes predicted for last 5 items:')
print(man_rf.test_stats()[5][-5:]) # predictions for last 5 items

man_rf = Man_RF_Case(data, 5, 0.2, 2, 8, 0.02, True)
man_rf.model_fit()
man_rf.results()
man_rf.metrics()
print('RF Test 3')
print('bag data for last 5 items:')
print(man_rf.test_stats()[0][-5:]) # bag data for last 5 items
print('column values to use:')
print(man_rf.test_stats()[1]) # col values
print('predictions of 1st forest for last 5 test items:')
print(man_rf.test_stats()[2][0][-5:]) # 1st for_preds for last 5 items
print('predictions of 2nd forest for last 5 test items:')
print(man_rf.test_stats()[2][1][-5:]) # 2nd for_preds for last 5 items
print('temporary probabilities stored for latest item:')
print(man_rf.test_stats()[3]) # temp_probs for latest item
print('temporary mean probability stored for latest item:')
print(man_rf.test_stats()[4]) # temp mean prob for latest item
print('classes predicted for last 5 items:')
print(man_rf.test_stats()[5][-5:]) # predictions for last 5 items

###################### Results for analysis

## Test results 1 - for metrics comparison, manual vs library

with open('python_results_1.csv', 'w', newline='') as m:
    thewriter = csv.writer(m)
    
    thewriter.writerow(['Test Name', 'TP', 'TN', 'FP', 'FN', 'Accuracy',\
                        'Precision', 'Recall', 'F1 Score',\
                        'Computational Time'])

for i in range(0, 100):
    lib_rf = Lib_RF_Case(10, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results('lib_res')

for i in range(0, 100):
    man_rf = Man_RF_Case(data, 10, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
#    man_rf.save_results('man_res')

## Test results 2 - for linear regression when changing number of forests

with open('python_results_2.csv', 'w', newline='') as m:
    thewriter = csv.writer(m)
    
    thewriter.writerow(['Test Name', 'Number of Forests', 'TP', 'TN', 'FP',\
                        'FN', 'Accuracy', 'Precision', 'Recall', 'F1 Score',\
                        'Computational Time'])

for i in range(0, 10):
    lib_rf = Lib_RF_Case(1, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results2('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(2, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results2('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(3, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results2('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(4, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results2('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(5, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results2('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(6, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results2('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(7, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results2('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(8, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results2('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(9, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results2('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(10, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results2('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(11, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results2('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(12, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results2('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(13, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results2('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(14, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results2('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(15, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results2('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(16, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results2('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(17, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results2('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(18, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results2('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(19, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results2('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(20, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results2('lib_res')

for i in range(0, 10):
    man_rf = Man_RF_Case(data, 1, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results2('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 2, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results2('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 3, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results2('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 4, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results2('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 5, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results2('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 6, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results2('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 7, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results2('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 8, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results2('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 9, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results2('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 10, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results2('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 11, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results2('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 12, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results2('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 13, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results2('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 14, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results2('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 15, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results2('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 16, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results2('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 17, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results2('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 18, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results2('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 19, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results2('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 20, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results2('man_res')
#
## Test results 3 - for linear regression when changing tree depths
#
with open('python_results_3.csv', 'w', newline='') as m:
    thewriter = csv.writer(m)
    
    thewriter.writerow(['Test Name', 'Tree Depth', 'TP', 'TN', 'FP',\
                        'FN', 'Accuracy', 'Precision', 'Recall', 'F1 Score',\
                        'Computational Time'])

for i in range(0, 10):
    lib_rf = Lib_RF_Case(10, 1, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results3('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(10, 2, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results3('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(10, 3, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results3('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(10, 4, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results3('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(10, 5, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results3('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(10, 6, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results3('lib_res')
    
for i in range(0, 10):
    lib_rf = Lib_RF_Case(10, 7, 5, 0.01)
    lib_rf.train_test_split(x_data, y_data, 0.25, True)
    lib_rf.model_fit()
    lib_rf.results()
    lib_rf.metrics(pos_label=0)
    lib_rf.save_results3('lib_res')

for i in range(0, 10):
    man_rf = Man_RF_Case(data, 10, 0.25, 1, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results3('man_res')

for i in range(0, 10):
    man_rf = Man_RF_Case(data, 10, 0.25, 2, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results3('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 10, 0.25, 3, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results3('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 10, 0.25, 4, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results3('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 10, 0.25, 5, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results3('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 10, 0.25, 6, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results3('man_res')
    
for i in range(0, 10):
    man_rf = Man_RF_Case(data, 10, 0.25, 7, 5, 0.01, True)
    man_rf.model_fit()
    man_rf.results()
    man_rf.metrics()
    man_rf.save_results3('man_res')
    
## ROC Manual Data

man_rf = Man_RF_Case(data, 10, 0.2, 3, 5, 0.01, True)
man_rf.model_fit()
man_rf.results()
man_rf.save_ROC_data('man_res')

## ROC Library Data

lib_rf = Lib_RF_Case(10, 3, 5, 0.01)
lib_rf.train_test_split(x_data, y_data, 0.2, True)
lib_rf.model_fit()
lib_rf.results()
lib_rf.metrics(pos_label=0)
lib_rf.save_ROC_data('lib_res')
