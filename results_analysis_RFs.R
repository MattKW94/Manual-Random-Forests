# Results Analysis
# Random Forests Manual vs SK Learn

###################### Loading in libraries and data

#setwd("H:/My Documents")
library(dplyr)
library(ggplot2)
library(ggthemes)

## Data for analysis 1 - Comparing consistency of methods

results <- read.csv('python_results_1.csv')
lib_res <- filter(results, (Test.Name == 'lib_res'))
man_res <- filter(results, (Test.Name == 'man_res'))

## Data for analysis 2 - Chanching number of trees in forests

results2 <- read.csv('python_results_2.csv')
lib_res2 <- filter(results2, (Test.Name == 'lib_res'))
man_res2 <- filter(results2, (Test.Name == 'man_res'))

## Data for analysis 3 - Changing tree depths

results3 <- read.csv('python_results_3.csv')
lib_res3 <- filter(results3, (Test.Name == 'lib_res'))
man_res3 <- filter(results3, (Test.Name == 'man_res'))

## Manual ROC Data

man_roc <- read.csv('python_ROC_Manual.csv') # for ROC curve 1
man_roc2 <- read.csv('python_ROC_Manual2.csv') # for ROC curve 2

## Library ROC Data

lib_roc <- read.csv('python_ROC_Library.csv') # for ROC curve 1
lib_roc2 <- read.csv('python_ROC_Library2.csv') # for ROC curve 2

###################### Analysis 1 - Comparing consistency of methods

lib_acc_greater = 0 # initial value
man_acc_greater = 0 # initial value
for (i in 1:length(lib_res$Accuracy)){
  if (lib_res$Accuracy[i] > man_res$Accuracy[i]){
    lib_acc_greater <- lib_acc_greater + 1}
  else{
    man_acc_greater <- man_acc_greater + 1
  }
}

print(lib_acc_greater)
print(man_acc_greater)

# Histogram showing number of times accuracy is greater for each classifier
acc <- data.frame(Test_Name = c('Library', 'Manual'), No_of_times_greater = c(lib_acc_greater, man_acc_greater) )
ggplot( data = acc, mapping = aes(x = Test_Name, y = No_of_times_greater) ) + 
  geom_bar(stat = 'identity', colour = 'black', fill = 'indianred1') + ggtitle('Accuracy Comparison of Methods') +
  xlab('Test Name') + ylab('Number of Times Greater') + theme_grey(base_size = 22)

# probability of imbalance being this high or greater
print(2*pbinom(min(lib_acc_greater, man_acc_greater), size=100, prob=0.5))

# T-test on accuracy mean differences using each classifier
print(t.test(lib_res$Accuracy, man_res$Accuracy, conf=0.95, paired=TRUE))

lib_prec_greater = 0 # initial value
man_prec_greater = 0 # initial value
for (i in 1:length(lib_res$Precision)){
  if (lib_res$Precision[i] > man_res$Precision[i]){
    lib_prec_greater <- lib_prec_greater + 1}
  else{
    man_prec_greater <- man_prec_greater + 1
  }
}

print(lib_prec_greater)
print(man_prec_greater)

# Histogram showing number of times precision is greater for each classifier
prec <- data.frame(Test_Name = c('Library', 'Manual'), No_of_times_greater = c(lib_prec_greater, man_prec_greater) )
ggplot( data = prec, mapping = aes(x = Test_Name, y = No_of_times_greater) ) + 
  geom_bar(stat = 'identity', colour = 'black', fill = 'indianred1') + ggtitle('Precision Comparison of Methods') +
  xlab('Test Name') + ylab('Number of Times Greater') + theme_grey(base_size = 22)

# probability of imbalance being this high or greater
print(2*pbinom(min(lib_prec_greater, man_prec_greater), size=100, prob=0.5))

# T-test on precision mean differences using each classifier
print(t.test(lib_res$Precision, man_res$Precision, conf=0.95, paired=TRUE))

lib_recall_greater = 0 # initial value
man_recall_greater = 0 # initial value
for (i in 1:length(lib_res$Recall)){
  if (lib_res$Recall[i] > man_res$Recall[i]){
    lib_recall_greater <- lib_recall_greater + 1}
  else{
    man_recall_greater <- man_recall_greater + 1
  }
}

print(lib_recall_greater)
print(man_recall_greater)

# Histogram showing number of times recall is greater for each classifier
recall <- data.frame(Test_Name = c('Library', 'Manual'), 
                     No_of_times_greater = c(lib_recall_greater, man_recall_greater) )
ggplot( data = recall, mapping = aes(x = Test_Name, y = No_of_times_greater) ) + 
  geom_bar(stat = 'identity', colour = 'black', fill = 'indianred1') + ggtitle('Recall Comparison of Methods') +
  xlab('Test Name') + ylab('Number of Times Greater') + theme_grey(base_size = 22)

# probability of imbalance being this high or greater
print(2*pbinom(min(lib_recall_greater, man_recall_greater), size=100, prob=0.5))

# T-test on recall mean differences using each classifier
print(t.test(lib_res$Recall, man_res$Recall, conf=0.95, paired=TRUE))

lib_f1_score_greater = 0 # initial value
man_f1_score_greater = 0 # initial value
for (i in 1:length(lib_res$F1.Score)){
  if (lib_res$F1.Score[i] > man_res$F1.Score[i]){
    lib_f1_score_greater <- lib_f1_score_greater + 1}
  else{
    man_f1_score_greater <- man_f1_score_greater + 1
  }
}

print(lib_f1_score_greater)
print(man_f1_score_greater)

# Histogram showing number of times F1 score is greater for each classifier
f1 <- data.frame(Test_Name = c('Library', 'Manual'),
                 No_of_times_greater = c(lib_f1_score_greater, man_f1_score_greater) )
ggplot( data = f1, mapping = aes(x = Test_Name, y = No_of_times_greater) ) + 
  geom_bar(stat = 'identity', colour = 'black', fill = 'indianred1') + ggtitle('F1 Score Comparison of Methods') +
  xlab('Test Name') + ylab('Number of Times Greater') + theme_grey(base_size = 22)

# probability of imbalance being this high or greater
print(2*pbinom(min(lib_f1_score_greater, man_f1_score_greater), size=100, prob=0.5))

# T-test on F1 Score mean differences using each classifier
print(t.test(lib_res$F1.Score, man_res$F1.Score, conf=0.95, paired=TRUE))


## Computational time boxplots

ggplot(data = lib_res, aes(x="", y=Computational.Time)) + geom_boxplot(colour="black", fill="cadetblue1") +
  ggtitle('Computational Time Comparison of Methods') + theme_grey(base_size = 22) +
  xlab('Library') + ylab('Computational Time')

ggplot(data = man_res, aes(x="", y=Computational.Time)) + geom_boxplot(colour="black", fill="cadetblue1") +
  ggtitle('Computational Time Comparison of Methods') + theme_grey(base_size = 22) +
  xlab('Manual') + ylab('Computational Time')

###################### Analysis 2 - Changing number of trees in forests

## Accuracy with tree increases - library

lib_1 <- filter(lib_res2, (Number.of.Forests == '1'))
lib_1_acc_mean <- mean(lib_1$Accuracy)
lib_1_acc_sd <- sd(lib_1$Accuracy)
lib_2 <- filter(lib_res2, (Number.of.Forests == '2'))
lib_2_acc_mean <- mean(lib_2$Accuracy)
lib_2_acc_sd <- sd(lib_2$Accuracy)
lib_3 <- filter(lib_res2, (Number.of.Forests == '3'))
lib_3_acc_mean <- mean(lib_3$Accuracy)
lib_3_acc_sd <- sd(lib_3$Accuracy)
lib_4 <- filter(lib_res2, (Number.of.Forests == '4'))
lib_4_acc_mean <- mean(lib_4$Accuracy)
lib_4_acc_sd <- sd(lib_4$Accuracy)
lib_5 <- filter(lib_res2, (Number.of.Forests == '5'))
lib_5_acc_mean <- mean(lib_5$Accuracy)
lib_5_acc_sd <- sd(lib_5$Accuracy)
lib_6 <- filter(lib_res2, (Number.of.Forests == '6'))
lib_6_acc_mean <- mean(lib_6$Accuracy)
lib_6_acc_sd <- sd(lib_6$Accuracy)
lib_7 <- filter(lib_res2, (Number.of.Forests == '7'))
lib_7_acc_mean <- mean(lib_7$Accuracy)
lib_7_acc_sd <- sd(lib_7$Accuracy)
lib_8 <- filter(lib_res2, (Number.of.Forests == '8'))
lib_8_acc_mean <- mean(lib_8$Accuracy)
lib_8_acc_sd <- sd(lib_8$Accuracy)
lib_9 <- filter(lib_res2, (Number.of.Forests == '9'))
lib_9_acc_mean <- mean(lib_9$Accuracy)
lib_9_acc_sd <- sd(lib_9$Accuracy)
lib_10 <- filter(lib_res2, (Number.of.Forests == '10'))
lib_10_acc_mean <- mean(lib_10$Accuracy)
lib_10_acc_sd <- sd(lib_10$Accuracy)
lib_11 <- filter(lib_res2, (Number.of.Forests == '11'))
lib_11_acc_mean <- mean(lib_11$Accuracy)
lib_11_acc_sd <- sd(lib_11$Accuracy)
lib_12 <- filter(lib_res2, (Number.of.Forests == '12'))
lib_12_acc_mean <- mean(lib_12$Accuracy)
lib_12_acc_sd <- sd(lib_12$Accuracy)
lib_13 <- filter(lib_res2, (Number.of.Forests == '13'))
lib_13_acc_mean <- mean(lib_13$Accuracy)
lib_13_acc_sd <- sd(lib_13$Accuracy)
lib_14 <- filter(lib_res2, (Number.of.Forests == '14'))
lib_14_acc_mean <- mean(lib_14$Accuracy)
lib_14_acc_sd <- sd(lib_14$Accuracy)
lib_15 <- filter(lib_res2, (Number.of.Forests == '15'))
lib_15_acc_mean <- mean(lib_15$Accuracy)
lib_15_acc_sd <- sd(lib_15$Accuracy)
lib_16 <- filter(lib_res2, (Number.of.Forests == '16'))
lib_16_acc_mean <- mean(lib_16$Accuracy)
lib_16_acc_sd <- sd(lib_16$Accuracy)
lib_17 <- filter(lib_res2, (Number.of.Forests == '17'))
lib_17_acc_mean <- mean(lib_17$Accuracy)
lib_17_acc_sd <- sd(lib_17$Accuracy)
lib_18 <- filter(lib_res2, (Number.of.Forests == '18'))
lib_18_acc_mean <- mean(lib_18$Accuracy)
lib_18_acc_sd <- sd(lib_18$Accuracy)
lib_19 <- filter(lib_res2, (Number.of.Forests == '19'))
lib_19_acc_mean <- mean(lib_19$Accuracy)
lib_19_acc_sd <- sd(lib_19$Accuracy)
lib_20 <- filter(lib_res2, (Number.of.Forests == '20'))
lib_20_acc_mean <- mean(lib_20$Accuracy)
lib_20_acc_sd <- sd(lib_20$Accuracy)

lib_for_acc <- data.frame("Number_of_Forests" = 1:20, "Mean_Accuracy" = 
                            c(lib_1_acc_mean, lib_2_acc_mean, lib_3_acc_mean, lib_4_acc_mean, lib_5_acc_mean
                              , lib_6_acc_mean, lib_7_acc_mean, lib_8_acc_mean, lib_9_acc_mean, lib_10_acc_mean,
                              lib_11_acc_mean, lib_12_acc_mean, lib_13_acc_mean, lib_14_acc_mean, lib_15_acc_mean,
                              lib_16_acc_mean, lib_17_acc_mean, lib_18_acc_mean, lib_19_acc_mean, lib_20_acc_mean),
                          "SD_Accuracy" = 
                            c(lib_1_acc_sd, lib_2_acc_sd, lib_3_acc_sd, lib_4_acc_sd, lib_5_acc_sd
                              , 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0,
                              lib_16_acc_sd, lib_17_acc_sd, lib_18_acc_sd, lib_19_acc_sd, lib_20_acc_sd))

ggplot( data = lib_for_acc, mapping = aes(x = Number_of_Forests, y = Mean_Accuracy) ) + 
  geom_point(color='red', size=2) + ggtitle('Library Accuracy Plot vs Number of Trees') +
  geom_errorbar(aes(ymin=Mean_Accuracy-SD_Accuracy, ymax=Mean_Accuracy+SD_Accuracy,), width=0.2) +
  theme_grey(base_size = 22) + xlab('Number of Trees in Random Forest') + ylab('Mean Accuracy') +
  geom_smooth(method='lm') # 95% level

## Accuracy with tree increases - manual

man_1 <- filter(man_res2, (Number.of.Forests == '1'))
man_1_acc_mean <- mean(man_1$Accuracy)
man_1_acc_sd <- sd(man_1$Accuracy)
man_1_time_sd <- sd(man_1$Computational.Time)
man_1_time_mean <- mean(man_1$Computational.Time)
man_2 <- filter(man_res2, (Number.of.Forests == '2'))
man_2_acc_mean <- mean(man_2$Accuracy)
man_2_acc_sd <- sd(man_2$Accuracy)
man_2_time_sd <- sd(man_2$Computational.Time)
man_2_time_mean <- mean(man_2$Computational.Time)
man_3 <- filter(man_res2, (Number.of.Forests == '3'))
man_3_acc_mean <- mean(man_3$Accuracy)
man_3_acc_sd <- sd(man_3$Accuracy)
man_3_time_sd <- sd(man_3$Computational.Time)
man_3_time_mean <- mean(man_3$Computational.Time)
man_4 <- filter(man_res2, (Number.of.Forests == '4'))
man_4_acc_mean <- mean(man_4$Accuracy)
man_4_acc_sd <- sd(man_4$Accuracy)
man_4_time_sd <- sd(man_4$Computational.Time)
man_4_time_mean <- mean(man_4$Computational.Time)
man_5 <- filter(man_res2, (Number.of.Forests == '5'))
man_5_acc_mean <- mean(man_5$Accuracy)
man_5_acc_sd <- sd(man_5$Accuracy)
man_5_time_sd <- sd(man_5$Computational.Time)
man_5_time_mean <- mean(man_5$Computational.Time)
man_6 <- filter(man_res2, (Number.of.Forests == '6'))
man_6_acc_mean <- mean(man_6$Accuracy)
man_6_time_mean <- mean(man_6$Computational.Time)
man_7 <- filter(man_res2, (Number.of.Forests == '7'))
man_7_acc_mean <- mean(man_7$Accuracy)
man_7_time_mean <- mean(man_7$Computational.Time)
man_8 <- filter(man_res2, (Number.of.Forests == '8'))
man_8_acc_mean <- mean(man_8$Accuracy)
man_8_time_mean <- mean(man_8$Computational.Time)
man_9 <- filter(man_res2, (Number.of.Forests == '9'))
man_9_acc_mean <- mean(man_9$Accuracy)
man_9_time_mean <- mean(man_9$Computational.Time)
man_10 <- filter(man_res2, (Number.of.Forests == '10'))
man_10_acc_mean <- mean(man_10$Accuracy)
man_10_time_mean <- mean(man_10$Computational.Time)
man_11 <- filter(man_res2, (Number.of.Forests == '11'))
man_11_acc_mean <- mean(man_11$Accuracy)
man_11_time_mean <- mean(man_11$Computational.Time)
man_12 <- filter(man_res2, (Number.of.Forests == '12'))
man_12_acc_mean <- mean(man_12$Accuracy)
man_12_time_mean <- mean(man_12$Computational.Time)
man_13 <- filter(man_res2, (Number.of.Forests == '13'))
man_13_acc_mean <- mean(man_13$Accuracy)
man_13_time_mean <- mean(man_13$Computational.Time)
man_14 <- filter(man_res2, (Number.of.Forests == '14'))
man_14_acc_mean <- mean(man_14$Accuracy)
man_14_time_mean <- mean(man_14$Computational.Time)
man_15 <- filter(man_res2, (Number.of.Forests == '15'))
man_15_acc_mean <- mean(man_15$Accuracy)
man_15_time_mean <- mean(man_15$Computational.Time)
man_16 <- filter(man_res2, (Number.of.Forests == '16'))
man_16_acc_mean <- mean(man_16$Accuracy)
man_16_acc_sd <- sd(man_16$Accuracy)
man_16_time_sd <- sd(man_16$Computational.Time)
man_16_time_mean <- mean(man_16$Computational.Time)
man_17 <- filter(man_res2, (Number.of.Forests == '17'))
man_17_acc_mean <- mean(man_17$Accuracy)
man_17_acc_sd <- sd(man_17$Accuracy)
man_17_time_sd <- sd(man_17$Computational.Time)
man_17_time_mean <- mean(man_17$Computational.Time)
man_18 <- filter(man_res2, (Number.of.Forests == '18'))
man_18_acc_mean <- mean(man_18$Accuracy)
man_18_acc_sd <- sd(man_18$Accuracy)
man_18_time_sd <- sd(man_18$Computational.Time)
man_18_time_mean <- mean(man_18$Computational.Time)
man_19 <- filter(man_res2, (Number.of.Forests == '19'))
man_19_acc_mean <- mean(man_19$Accuracy)
man_19_acc_sd <- sd(man_19$Accuracy)
man_19_time_sd <- sd(man_19$Computational.Time)
man_19_time_mean <- mean(man_19$Computational.Time)
man_20 <- filter(man_res2, (Number.of.Forests == '20'))
man_20_acc_mean <- mean(man_20$Accuracy)
man_20_acc_sd <- sd(man_20$Accuracy)
man_20_time_sd <- sd(man_20$Computational.Time)
man_20_time_mean <- mean(man_20$Computational.Time)

man_for <- data.frame("Number_of_Forests" = 1:20, "Mean_Accuracy" = 
                            c(man_1_acc_mean, man_2_acc_mean, man_3_acc_mean, man_4_acc_mean, man_5_acc_mean
                              , man_6_acc_mean, man_7_acc_mean, man_8_acc_mean, man_9_acc_mean, man_10_acc_mean,
                              man_11_acc_mean, man_12_acc_mean, man_13_acc_mean, man_14_acc_mean, man_15_acc_mean,
                              man_16_acc_mean, man_17_acc_mean, man_18_acc_mean, man_19_acc_mean, man_20_acc_mean),
                          "SD_Accuracy" = 
                            c(man_1_acc_sd, man_2_acc_sd, man_3_acc_sd, man_4_acc_sd, man_5_acc_sd
                              , 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0,
                              man_16_acc_sd, man_17_acc_sd, man_18_acc_sd, man_19_acc_sd, man_20_acc_sd),
                          "Mean_Time" = 
                            c(man_1_time_mean, man_2_time_mean, man_3_time_mean, man_4_time_mean, man_5_time_mean
                              , man_6_time_mean, man_7_time_mean, man_8_time_mean, man_9_time_mean, man_10_time_mean,
                              man_11_time_mean, man_12_time_mean, man_13_time_mean, man_14_time_mean,
                              man_15_time_mean, man_16_time_mean, man_17_time_mean, man_18_time_mean,
                              man_19_time_mean, man_20_time_mean),
                          "SD_Time" = 
                            c(man_1_time_sd, man_2_time_sd, man_3_time_sd, man_4_time_sd, man_5_time_sd
                            , 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0,
                            man_16_time_sd, man_17_time_sd, man_18_time_sd, man_19_time_sd, man_20_time_sd))

ggplot( data = man_for, mapping = aes(x = Number_of_Forests, y = Mean_Accuracy) ) + 
  geom_point(color='red', size=2) + ggtitle('Manual Accuracy Plot vs Number of Trees') +
  geom_errorbar(aes(ymin=Mean_Accuracy-SD_Accuracy, ymax=Mean_Accuracy+SD_Accuracy,), width=0.2) +
  theme_grey(base_size = 22) + xlab('Number of Trees in Random Forest') + ylab('Mean Accuracy') +
  geom_smooth(method='lm') # 95% level

ggplot( data = man_for, mapping = aes(x = Number_of_Forests, y = Mean_Time) ) + 
  geom_point(color='red', size=2) + ggtitle('Manual Time Plot vs Number of Trees') +
  geom_errorbar(aes(ymin=Mean_Time-SD_Time, ymax=Mean_Time+SD_Time,), width=0.2) +
  theme_grey(base_size = 22) + xlab('Number of Trees in Random Forest') + ylab('Mean Time') +
  geom_smooth(method='lm') # 95% level

###################### Analysis 3 - Changing tree depths

## Accuracy with layer increases - library

lib_1 <- filter(lib_res3, (Tree.Depth == '1'))
lib_1_acc_mean <- mean(lib_1$Accuracy)
lib_1_acc_sd <- sd(lib_1$Accuracy)
lib_2 <- filter(lib_res3, (Tree.Depth == '2'))
lib_2_acc_mean <- mean(lib_2$Accuracy)
lib_2_acc_sd <- sd(lib_2$Accuracy)
lib_3 <- filter(lib_res3, (Tree.Depth == '3'))
lib_3_acc_mean <- mean(lib_3$Accuracy)
lib_3_acc_sd <- sd(lib_3$Accuracy)
lib_4 <- filter(lib_res3, (Tree.Depth == '4'))
lib_4_acc_mean <- mean(lib_4$Accuracy)
lib_4_acc_sd <- sd(lib_4$Accuracy)
lib_5 <- filter(lib_res3, (Tree.Depth == '5'))
lib_5_acc_mean <- mean(lib_5$Accuracy)
lib_5_acc_sd <- sd(lib_5$Accuracy)
lib_6 <- filter(lib_res3, (Tree.Depth == '6'))
lib_6_acc_mean <- mean(lib_6$Accuracy)
lib_6_acc_sd <- sd(lib_6$Accuracy)
lib_7 <- filter(lib_res3, (Tree.Depth == '7'))
lib_7_acc_mean <- mean(lib_7$Accuracy)
lib_7_acc_sd <- sd(lib_7$Accuracy)

lib_depth_acc <- data.frame("Tree_Depth" = 1:7, "Mean_Accuracy" = 
                            c(lib_1_acc_mean, lib_2_acc_mean, lib_3_acc_mean, lib_4_acc_mean, lib_5_acc_mean
                              , lib_6_acc_mean, lib_7_acc_mean),
                          "SD_Accuracy" = 
                            c(lib_1_acc_sd, lib_2_acc_sd, lib_3_acc_sd, lib_4_acc_sd, lib_5_acc_sd
                              , lib_6_acc_sd, lib_7_acc_sd))

ggplot( data = lib_depth_acc, mapping = aes(x = Tree_Depth, y = Mean_Accuracy) ) + 
  geom_point(color='red', size=2) + ggtitle('Library Accuracy Plot vs Tree Depth') +
  geom_errorbar(aes(ymin=Mean_Accuracy-SD_Accuracy, ymax=Mean_Accuracy+SD_Accuracy,), width=0.2) +
  theme_grey(base_size = 22) + xlab('Depth of Trees in Random Forest') + ylab('Mean Accuracy') +
  geom_smooth(method='lm') # 95% level

## Accuracy with tree increases - manual

man_1 <- filter(man_res3, (Tree.Depth == '1'))
man_1_acc_mean <- mean(man_1$Accuracy)
man_1_acc_sd <- sd(man_1$Accuracy)
man_1_time_mean <- mean(man_1$Computational.Time)
man_1_time_sd <- sd(man_1$Computational.Time)
man_2 <- filter(man_res3, (Tree.Depth == '2'))
man_2_acc_mean <- mean(man_2$Accuracy)
man_2_acc_sd <- sd(man_2$Accuracy)
man_2_time_mean <- mean(man_2$Computational.Time)
man_2_time_sd <- sd(man_2$Computational.Time)
man_3 <- filter(man_res3, (Tree.Depth == '3'))
man_3_acc_mean <- mean(man_3$Accuracy)
man_3_acc_sd <- sd(man_3$Accuracy)
man_3_time_mean <- mean(man_3$Computational.Time)
man_3_time_sd <- sd(man_3$Computational.Time)
man_4 <- filter(man_res3, (Tree.Depth == '4'))
man_4_acc_mean <- mean(man_4$Accuracy)
man_4_acc_sd <- sd(man_4$Accuracy)
man_4_time_mean <- mean(man_4$Computational.Time)
man_4_time_sd <- sd(man_4$Computational.Time)
man_5 <- filter(man_res3, (Tree.Depth == '5'))
man_5_acc_mean <- mean(man_5$Accuracy)
man_5_acc_sd <- sd(man_5$Accuracy)
man_5_time_mean <- mean(man_5$Computational.Time)
man_5_time_sd <- sd(man_5$Computational.Time)
man_6 <- filter(man_res3, (Tree.Depth == '6'))
man_6_acc_mean <- mean(man_6$Accuracy)
man_6_acc_sd <- sd(man_6$Accuracy)
man_6_time_mean <- mean(man_6$Computational.Time)
man_6_time_sd <- sd(man_6$Computational.Time)
man_7 <- filter(man_res3, (Tree.Depth == '7'))
man_7_acc_mean <- mean(man_7$Accuracy)
man_7_acc_sd <- sd(man_7$Accuracy)
man_7_time_mean <- mean(man_7$Computational.Time)
man_7_time_sd <- sd(man_7$Computational.Time)

man_for2 <- data.frame("Tree_Depth" = 1:7, "Mean_Accuracy" = 
                              c(man_1_acc_mean, man_2_acc_mean, man_3_acc_mean, man_4_acc_mean, man_5_acc_mean
                                , man_6_acc_mean, man_7_acc_mean),
                            "SD_Accuracy" = 
                              c(man_1_acc_sd, man_2_acc_sd, man_3_acc_sd, man_4_acc_sd, man_5_acc_sd
                                , man_6_acc_sd, man_7_acc_sd),
                           "Mean_Time" = 
                             c(man_1_time_mean, man_2_time_mean, man_3_time_mean, man_4_time_mean, man_5_time_mean
                               , man_6_time_mean, man_7_time_mean),
                           "SD_Time" = 
                             c(man_1_time_sd, man_2_time_sd, man_3_time_sd, man_4_time_sd, man_5_time_sd
                               , man_6_time_sd, man_7_time_sd))

ggplot( data = man_for2, mapping = aes(x = Tree_Depth, y = Mean_Accuracy) ) + 
  geom_point(color='red', size=2) + ggtitle('Manual Accuracy Plot vs Tree Depth') +
  geom_errorbar(aes(ymin=Mean_Accuracy-SD_Accuracy, ymax=Mean_Accuracy+SD_Accuracy,), width=0.2) +
  theme_grey(base_size = 22) + xlab('Depth of Trees in Random Forest') + ylab('Mean Accuracy') +
  geom_smooth(method='lm') # 95% level

ggplot( data = man_for2, mapping = aes(x = Tree_Depth, y = Mean_Time) ) + 
  geom_point(color='red', size=2) + ggtitle('Manual Time Plot vs Tree Depth') +
  geom_errorbar(aes(ymin=Mean_Time-SD_Time, ymax=Mean_Time+SD_Time,), width=0.2) +
  theme_grey(base_size = 22) + xlab('Depth of Trees in Random Forest') + ylab('Mean Time') +
  geom_smooth(method='lm') # 95% level

###################### ROC plots

man_roc <-  man_roc[order(man_roc$TP.Rate),] # man_roc data re-ordered for use in ggplot for ROC curve

roc_data <- rbind(man_roc, lib_roc) # manual and library ROC dataframes combined

# Just manual

ggplot( data = man_roc) + 
  geom_point(aes(x = FP.Rate, y = TP.Rate, colour='Manual RF Method'), size=2) + ggtitle('Manual ROC Curve') +
  geom_step(aes(x = FP.Rate, y = TP.Rate), colour='blue', size=1, direction='vh') + xlab('False Positive Rate') +
  ylab('True Positive Rate') + geom_abline(aes(colour='Random', intercept=0, slope=1), size=1) +
  scale_color_manual(values = c("Manual RF Method" = "blue", "Random" = "red"),
  guide = guide_legend(override.aes = list(pch = c(NA, NA), linetype = c(1, 1))))

# Just Library

ggplot( data = lib_roc) + 
  geom_point(aes(x = FP.Rate, y = TP.Rate, colour='Library RF Method'), size=2) + ggtitle('Library ROC Curve') +
  geom_step(aes(x = FP.Rate, y = TP.Rate), colour='blue', size=1, direction='vh') + xlab('False Positive Rate') +
  ylab('True Positive Rate') + geom_abline(aes(colour='Random', intercept=0, slope=1), size=1) +
  scale_color_manual(values = c("Library RF Method" = "blue", "Random" = "red"),
                     guide = guide_legend(override.aes = list(pch = c(NA, NA), linetype = c(1, 1))))

# Combined ROC plot

ggplot(data = roc_data) + 
  geom_point(aes(x = FP.Rate, y = TP.Rate, group=Test.Name, colour=Test.Name, shape=Test.Name),
             size=2) +
  ggtitle('ROC Curves') +
  geom_step(aes(x = FP.Rate, y = TP.Rate, group=Test.Name, colour=Test.Name), size=1,
            direction='vh', show.legend=FALSE) +
  xlab('False Positive Rate') +
  ylab('True Positive Rate') + geom_abline(aes(colour='Random', intercept=0, slope=1), size=1) +
  scale_color_manual(values = c("blue", "green4", "red"), labels = c('Library', 'Manual', 'Random'),
                     guide = guide_legend(override.aes = list(pch = c(NA, NA, NA), linetype = c(1, 1, 1)))) +
  scale_shape_manual(values=c(16, 17), labels = c('Manual', 'Library')) +
  labs(colour = "Method", shape = "Method") + theme_grey(base_size = 22)

## ROC plot 2

man_roc2 <-  man_roc2[order(man_roc2$TP.Rate),] # man_roc data re-ordered for use in ggplot for ROC curve

roc_data2  <- rbind(man_roc2, lib_roc2) # manual and library ROC dataframes combined

# Combined ROC plot

ggplot(data = roc_data2) + 
  geom_point(aes(x = FP.Rate, y = TP.Rate, group=Test.Name, colour=Test.Name, shape=Test.Name),
             size=2) +
  ggtitle('ROC Curves') +
  geom_step(aes(x = FP.Rate, y = TP.Rate, group=Test.Name, colour=Test.Name), size=1,
            direction='vh', show.legend=FALSE) +
  xlab('False Positive Rate') +
  ylab('True Positive Rate') + geom_abline(aes(colour='Random', intercept=0, slope=1), size=1) +
  scale_color_manual(values = c("blue", "green4", "red"), labels = c('Library', 'Manual', 'Random'),
                     guide = guide_legend(override.aes = list(pch = c(NA, NA, NA), linetype = c(1, 1, 1)))) +
  scale_shape_manual(values=c(16, 17), labels = c('Manual', 'Library')) +
  labs(colour = "Method", shape = "Method") + theme_grey(base_size = 22)