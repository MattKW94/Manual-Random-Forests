# Random Forests Manual Classifier vs SK Learn

###################### Decision Tree - SK Learn

class Lib_DT:
    
    def __init__(self, layers, min_items, min_imp_dec):
        from sklearn import tree
        self.tree = tree.DecisionTreeClassifier(max_depth = layers,\
        min_samples_split = min_items, min_impurity_decrease = min_imp_dec)
        
    def train_test_split(self, x_data, y_data, test_prop, shuffle_state):
        from sklearn.model_selection import train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = \
        train_test_split(x_data, y_data, test_size = test_prop,\
                         shuffle = shuffle_state)
        
    def model_fit(self):
        self.tree.fit(self.x_train, self.y_train)
        
    def results(self):
        preds = self.tree.predict(self.x_test)
        self.preds = preds.tolist()
        
    def metrics(self, pos_label=1):
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import f1_score
        CM = confusion_matrix(self.y_test, self.preds)
        self.tn = CM.item((0, 0))
        self.tp = CM.item((1, 1))
        self.fp = CM.item((0, 1))
        self.fn = CM.item((1, 0))
        
        self.accuracy = accuracy_score(self.y_test, self.preds)
        
        # precision, recall and F1 score are in terms of the negative case,\
        # as this is the minority class
        self.recall = recall_score(self.y_test, self.preds, pos_label)
        self.precision = precision_score(self.y_test, self.preds, pos_label)
        self.f1_score = f1_score(self.y_test, self.preds, pos_label)

###################### Random Forests - SK Learn

class Lib_RF:
    
    def __init__(self, number_of_forests, layers, min_items, min_imp_dec):
        import time
        self.start_time = time.time() # to start timing the method
        self.layers = layers
        self.for_no = number_of_forests # number of trees in the RF
        from sklearn.ensemble import RandomForestClassifier
        self.rf = RandomForestClassifier(n_estimators=number_of_forests,\
        max_depth = layers, min_samples_split = min_items,\
        min_impurity_decrease = min_imp_dec)
        
    def train_test_split(self, x_data, y_data, test_prop, shuffle_state):
        from sklearn.model_selection import train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = \
        train_test_split(x_data, y_data, test_size = test_prop,\
                         shuffle = shuffle_state)
        
    def model_fit(self):
        self.rf.fit(self.x_train, self.y_train)
        
    def results(self):
        preds = self.rf.predict(self.x_test)
        self.preds = preds.tolist()
        probs = self.rf.predict_proba(self.x_test)
        probs = probs.tolist()
        self.probs = []
        for i in probs:
            self.probs.append(i[1]) # probabilities of being class 1
        
    def metrics(self, pos_label=1):
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import f1_score
        from sklearn.metrics import roc_curve
        CM = confusion_matrix(self.y_test, self.preds)
        self.tn = CM.item((0, 0))
        self.tp = CM.item((1, 1))
        self.fp = CM.item((0, 1))
        self.fn = CM.item((1, 0))
        
        self.accuracy = accuracy_score(self.y_test, self.preds)
        
        # precision, recall and F1 score are in terms of the negative case,\
        # as this is the minority class
        self.recall = recall_score(self.y_test, self.preds, pos_label)
        self.precision = precision_score(self.y_test, self.preds, pos_label)
        self.f1_score = f1_score(self.y_test, self.preds, pos_label)
        
        self.fp_rates, self.tp_rates, self.thresholds = roc_curve(self.y_test,\
                                                        self.probs)
        self.fp_rates = self.fp_rates.tolist()
        self.tp_rates = self.tp_rates.tolist()
        self.thresholds = self.thresholds.tolist()

###################### Decision Tree - Manual

class Man_DT:
    
    def __init__(self, data, test_prop, layers, min_items,\
                 min_imp_dec, shuffle_state):
        
        self.shuffle_state = shuffle_state
        if self.shuffle_state == True:
            import random
            random.shuffle(data)
        
        self.test_prop = test_prop
        # int is used so that the train_number is rounded down
        # rounding down is consistent with sk learn
        train_number = int((1-self.test_prop) * len(data))
        
        self.data = data
        self.train_data = self.data[0:train_number]
        self.test_data = self.data[train_number:]
        self.layers = layers
        self.min_items = min_items
        self.min_imp_dec = min_imp_dec
        self.col_values = []
        new_col_number = 0 # initialising value
        
        for i in range(0,len(self.train_data[0])):
            self.col_values.append([])
            temp = []
            for j in self.train_data:
                if j[i] not in temp:
                    temp.append(j[i])
                    self.col_values[i].append([j[i]])
            for x in self.col_values[i]:
                x.append(new_col_number)
                x.append(i)
                new_col_number += 1
    
    def gini(self, data, col, value):
        """ gini split of a dataset based on
        asking if it includes a value in a certain column"""
        dl = [] #split to left (answer is no)
        dr = [] #split to right (answer is yes)
        dl_1s = 0 #count 1s in split to left
        dl_0s = 0 #count 0s in split to left
        dr_1s = 0 #count 1s in split to right
        dr_0s = 0 #count 0s in split to right
        for item in data:
            if item[col] != value:
                dl.append(item)
            else:
                dr.append(item)
        for item in dl:
            if item[0] == 1:
                dl_1s += 1
            else:
                dl_0s +=1
        for item in dr:
            if item[0] == 1:
                dr_1s += 1
            else:
                dr_0s += 1
        if len(dl) != 0 and len(dr) != 0:
            g = (len(dl)/len(data))*(1-(dl_1s/len(dl))**2-(dl_0s/len(dl))**2)\
            + (len(dr)/len(data))*(1-(dr_1s/len(dr))**2-(dr_0s/len(dr))**2)
        else:
            g = 1 #no partitions found
        return g, dl, dr #return gini value, split to left and split to right

    def bestPartition(self, data, RF='no'):
        """ looks at the best partition of data using the gini
        function and col vals """
        if len(data) < self.min_items:
            return [99, 99, 99, 99]
        else:
            data_0s = 0
            data_1s = 0
            for i in data:
                if i[0] == 1:
                    data_1s += 1
                else:
                    data_0s += 1
            # the gini impurity of the data as it is
            g_init = 1 - (data_0s / len(data))**2 - (data_1s / len(data))**2
            x = 1
            # col 0 not used (class entry)
            for i in range(1, len(self.col_values)):
                for j in range(0, len(self.col_values[i])):
                    if self.gini(data, self.col_values[i][j][2],\
                                 self.col_values[i][j][0])[0] < x:
                        #best current split found
                        x = self.gini(data, self.col_values[i][j][2],\
                                      self.col_values[i][j][0])[0]
                        #new threshhold
                        split_col = self.col_values[i][j][2] #split column
                        split_value = self.col_values[i][j][0] # split data
                        dl = self.gini(data, self.col_values[i][j][2],\
                                       self.col_values[i][j][0])[1]
                        #split to left
                        dr = self.gini(data, self.col_values[i][j][2],\
                                       self.col_values[i][j][0])[2]
                        #split to right
            
            imp_dec = (len(data) / len(self.train_data)) * (g_init - x)
            
            # for random forests, where the split may need to look at
            # further features
            if RF == 'yes':
                if imp_dec >= self.min_imp_dec:
                    pass
                else:
                    # other_col_values need to be considered until a solution
                    # is found, if possible
                    for i in range(1, len(self.other_col_values)):
                        for j in range(0, len(self.other_col_values[i])):
                            if self.gini(data, self.other_col_values[i][j][2],\
                                         self.other_col_values[i][j][0])[0]< x:
                                #best current split found
                                x = self.gini(data,\
                                              self.other_col_values[i][j][2],\
                                              self.other_col_values[i][j][0])[0]
                                #new threshhold
                                #split column
                                split_col = self.other_col_values[i][j][2]
                                # split data
                                split_value = self.other_col_values[i][j][0]
                                dl = self.gini(data,\
                                               self.other_col_values[i][j][2],\
                                               self.other_col_values[i][j][0])[1]
                                #split to left
                                dr = self.gini(data,\
                                               self.other_col_values[i][j][2],\
                                               self.other_col_values[i][j][0])[2]
                                #split to right
                                imp_dec = (len(data) / len(self.train_data))\
                                * (g_init - x)
                            if imp_dec >= self.min_imp_dec:
                                break
                    
            
            if imp_dec < self.min_imp_dec:
                return [99, 99, 99, 99]
            # No good partitions / none at all possible with this data
            else:
                return split_col, split_value, dl, dr
    
    def tree_fit(self, RF='no'):
        """ This builds a tree to be used for classificication using
        training data and number of layers as inputs"""
        # RF variable used in random forests for the random feature selection
        # at each layer
        self.tree = [] # output of the function
        temp1 = [] # entry for a layer
        temp2 = [] # entry within a layer
        # split col for q
        temp2.append(self.bestPartition(self.train_data, RF)[0])
        # split val for q
        temp2.append(self.bestPartition(self.train_data, RF)[1])
        # training L split
        temp2.append(self.bestPartition(self.train_data, RF)[2])
        # training R split
        temp2.append(self.bestPartition(self.train_data, RF)[3])
        temp1.append(temp2)
        self.tree.append(temp1) # layer appended
        # -1 since layer 0 here will be referred to
        for i in range(0, self.layers-1):
            # as layer 1
            temp1 = []# next layer
            for j in range(0, len(self.tree[-1])): # looks @ the previous layer
                temp2 = []
                if self.tree[-1][j][0] == 99: # if the relevant question was 99
                    # (which means N/A), then make these questions N/A too
                    temp2.append(99)
                    temp1.append(temp2)
                    temp2.append(99)
                    temp1.append(temp2)
                else:
                    dl = self.tree[-1][j][2]
                    dr = self.tree[-1][j][3]
                    # looking at best partition from split to left of relevant
                    # question on the above layer
                    if RF == 'yes':
                        self.new_col_vals() # features randomly re-selected
                    temp2.append(self.bestPartition(dl, RF)[0])
                    temp2.append(self.bestPartition(dl, RF)[1])
                    if temp2[-1] != 99:
                    # so long as the question is applicable, add the splits
                        temp2.append(self.bestPartition(dl, RF)[2])
                        temp2.append(self.bestPartition(dl, RF)[3])
                    temp1.append(temp2)
                    temp2 = []
                    # looking at best partition from split to right of relevant
                    # question on the above layer
                    if RF == 'yes':
                        self.new_col_vals() # features randomly re-selected
                    temp2.append(self.bestPartition(dr, RF)[0])
                    temp2.append(self.bestPartition(dr, RF)[1])
                    # so long as the question is applicable, add the splits
                    if temp2[-1] != 99:
                        temp2.append(self.bestPartition(dr, RF)[2])
                        temp2.append(self.bestPartition(dr, RF)[3])
                    temp1.append(temp2)
            self.tree.append(temp1)
    
    # this returns just the questions from a tree, without all of the
    # splits to left and right of each question
    def TreeQuestions(self):        
        self.tree_qs = []
        for i in self.tree:
            temp1 = []
            for j in i:
                temp2 = []
                temp2.append(j[0])
                temp2.append(j[1])
                temp1.append(temp2)
            self.tree_qs.append(temp1)
    
    def Tree_Classifier(self, data_item, RFtree='None'):
        """ This classifies a data item, using training data and layers as
        inputs too. A tree is built using the traininf data and layers inputs"""
        if RFtree == 'None':
            tree = self.tree # random forests not being used
        else:
            tree = RFtree # for use in random forets
        lay = 0 # initial layer value to look at
        entry = 0 # initial entry in a layer to look at
        if tree[lay][entry][0] == 99:
            classify_list = self.train_data
        else:
            if data_item[tree[lay][entry][0]] != tree[lay][entry][1]:
                # look at answer to question in the tree
                # does the data item include this variable in this column?
                pos = 2 # left (no)
            else:
                pos = 3 # right (yes)
            for i in range(0, self.layers-1):
                lay_temp = lay + 1 # temp in case question is N/A
                entry_temp = 2*entry + pos - 2 # temp in case question is N/A
                if tree[lay_temp][entry_temp][0] == 99:
                # if question is N/A, continue and look at the next question
                    continue
                else:
                    lay = lay_temp # can now be assigned, as question is not N/A
                    entry = entry_temp # can now be assigned, as q is not N/A
                    if data_item[tree[lay][entry][0]] != tree[lay][entry][1]:
                        pos = 2 # left (no)
                    else:
                        pos = 3 # right (yes)
             # list to look at for classification
            classify_list = tree[lay][entry][pos]
        # perform classification
        total = len(classify_list)
        count_1s = 0
        for i in classify_list:
            if i[0] == 1:
                count_1s += 1
        if (count_1s / total) < 0.5:
            return 0
        else:
            return 1
    
    def Item_Probability(self, data_item, RFtree='None'):
        """ Same as Tree_Classifier, but returns a probabiity rather
        than 1 or 0 """
        if RFtree == 'None':
            tree = self.tree # random forests not being used
        else:
            tree = RFtree # for use in random forets
        lay = 0 # initial layer value to look at
        entry = 0 # initial entry in a layer to look at
        if tree[lay][entry][0] == 99:
            classify_list = self.train_data
        else:
            if data_item[tree[lay][entry][0]] != tree[lay][entry][1]:
                # look at answer to question in the tree
                # does the data item include this variable in this column?
                pos = 2 # left (no)
            else:
                pos = 3 # right (yes)
            for i in range(0, self.layers-1):
                lay_temp = lay + 1 # temp in case question is N/A
                entry_temp = 2*entry + pos - 2 # temp in case question is N/A
                if tree[lay_temp][entry_temp][0] == 99:
                # if question is N/A, continue and look at the next question
                    continue
                else:
                    lay = lay_temp # can now be assigned, as q is not N/A
                    entry = entry_temp # can now be assigned, as q is not N/A
                    if data_item[tree[lay][entry][0]] != tree[lay][entry][1]:
                        pos = 2 # left (no)
                    else:
                        pos = 3 # right (yes)
            # list to look at for classification
            classify_list = tree[lay][entry][pos]
        # perform classification
        total = len(classify_list)
        count_1s = 0
        for i in classify_list:
            if i[0] == 1:
                count_1s += 1
        return (count_1s / total)
    
    # probabilities of multiple items
    def tree_probs(self, RFtree='None'):
        # RFtree is for use in random forests
        self.probs = []
        
        # takes a long time to run on lots of test items
        for item in self.test_data:
            self.probs.append(self.Item_Probability(item, RFtree))
    
    ## classification
    def tree_results(self, RFtree='None'):
        # RFtree is for use in random forests
        self.preds = []
        
        # takes a long time to run on lots of test items
        for item in self.test_data:
            self.preds.append(self.Tree_Classifier(item, RFtree))
    
    def metrics(self):
        """ This provides the metrics of the results,
        comparing the predicted classes vs the actual classes"""
        
        self.tp = 0 # initial setting for true positives count
        self.tn = 0 # initial setting for true negatives count
        self.fp = 0 # initial setting for false positives count
        self.fn = 0 # initial setting for false negatives count
        
        for i in range(len(self.preds)):
            if self.preds[i] == 1:
                if self.preds[i] == self.test_data[i][0]:
                    self.tp += 1
                else:
                    self.fp += 1
            else:
                if self.preds[i] == self.test_data[i][0]:
                    self.tn += 1
                else:
                    self.fn += 1
        
        self.accuracy = (self.tn + self.tp) / (self.tn + self.tp +\
                        self.fp + self.fn)
        
        # precision, recall and F1 score are in terms of the minority class
        
        if (self.tp + self.fn) > (self.tn + self.fp): 
            if (self.tn + self.fn) == 0:
                self.precision = 'N/A'
            else:
                self.precision = self.tn / (self.tn + self.fn)
        else:
            if (self.tp + self.fp) == 0:
                self.precision = 'N/A'
            else:
                self.precision = self.tp / (self.tp + self.fp)   
            
        if (self.tp + self.fn) > (self.tn + self.fp):   
            if (self.tn + self.fp) == 0:
                self.recall = 'N/A'
            else:
                self.recall = self.tn / (self.tn + self.fp)
        else:
            if (self.tp + self.fn) == 0:
                self.recall = 'N/A'
            else:
                self.recall = self.tp / (self.tp + self.fn)
        
        if self.precision == 'N/A' or self.recall == 'N/A':
            self.f1_score = 'N/A'
        else:
            self.f1_score = 2 * ((self.precision * self.recall)\
                                 / (self.precision + self.recall))
        
        
        
###################### Random Forests - Manual
        
class Man_RF(Man_DT):
    
    def __init__(self, data, number_of_forests, test_prop, layers, min_items,\
                 min_imp_dec, shuffle_state):
        import time
        self.start_time = time.time() # to start timing the method
        Man_DT.__init__(self, data, test_prop, layers,\
                        min_items, min_imp_dec, shuffle_state)
        # number of trees in the RF
        self.for_no = number_of_forests # new input that wasn't in Man_DT
        
    def new_col_vals(self):
        """ This gives the features to be looked at in the random
        forest trees"""
        import random
        # random feature reduction - using sqrt of number of features
        # rounded down - features in this case being a particular
        # entry in a column (so it is in line with one hot encoded data
        # used in sk learn) - taken from orig_col_vals
        # 1 taken off orig_col_vals[-1][-1][1] so that the class
        # isn't used
        no_feat_to_use = int((self.orig_col_vals[-1][-1][1] - 1) ** 0.5)
        orig_feat_numbers = list(range(2,self.orig_col_vals[-1][-1][1]))
        
        # numbers of features to be used from orig_col_vals
        new_feat_numbers = random.sample(orig_feat_numbers, no_feat_to_use)
        
        # numbers of other features to be used from orig_col_vals
        # which will be used if a tree in the RF needs to look beyong
        # the new_col_vals to find a valid split
        other_feat_numbers = [] # initial value
        for i in orig_feat_numbers:
            if i not in new_feat_numbers:
                other_feat_numbers.append(i)
        
        # creating new self.col_values to be used in the tree
        new_col_values = []
        new_col_values.append(self.orig_col_vals[0]) # the classes
        for i in self.orig_col_vals:
            for j in i:
                if j[1] in new_feat_numbers:
                    if j[2] == new_col_values[-1][-1][2]:
                        new_col_values[-1].append(j)
                    else:
                        new_col_values.append([])
                        new_col_values[-1].append(j)
            self.col_values = new_col_values
            
        # creating self.other_col_values to be used in the tree if needbe
        other_col_values = []
        other_col_values.append(self.orig_col_vals[0]) # the classes
        for i in self.orig_col_vals:
            for j in i:
                if j[1] in other_feat_numbers:
                    if j[2] == other_col_values[-1][-1][2]:
                        other_col_values[-1].append(j)
                    else:
                        other_col_values.append([])
                        other_col_values[-1].append(j)
            self.other_col_values = other_col_values
        
        
    def model_fit(self):
        """ This fits the random forest trees"""
        import copy
        import random
        self.rf = [] # array of trees to be used for classification
        # bagging to be taken from original test data
        self.orig_train_data = copy.deepcopy(self.train_data)

        # features to look at to be considered from the original col_vals
        self.orig_col_vals = copy.deepcopy(self.col_values)
        
        for i in range(0, self.for_no):
            
            # bootlegged data to use for a tree
            self.bag_data = []
            for j in range(0, len(self.orig_train_data)):
                self.bag_data.append(random.sample(self.orig_train_data, 1)[0])
            
            # training data now set to be the bagged data
            self.train_data = self.bag_data
                
            self.new_col_vals()
            
            self.tree_fit('yes')
            temp_tree = copy.deepcopy(self.tree)
            self.rf.append(temp_tree)
    
    def results(self, thresh=0.5):
        """ This calculates the predictions for the test
        items, based on the random forest trees"""
        
        import copy
       
        # to be listed with probability lists from all trees
        self.for_probs = []
        for tree in self.rf:
            self.tree_probs(tree)
            self.items_probs = copy.deepcopy(self.probs)
            self.for_probs.append(self.items_probs)
        
        # to be appended based on the mean of the probs from all trees
        self.preds = []
        self.probs = []
        for i in range(0, len(self.for_probs[0])):
            self.temp_probs = []
            for j in range(0, len(self.for_probs)):
                self.temp_probs.append(self.for_probs[j][i])
            self.temp_mean_prob = sum(self.temp_probs) / len(self.temp_probs)
            # if the mean prob is geq to 0.5, classify as 1, otherwise 0
            if self.temp_mean_prob > thresh:
                self.preds.append(1)
            else:
                self.preds.append(0)
            self.probs.append(self.temp_mean_prob)
