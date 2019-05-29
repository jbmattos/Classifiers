clear all; clc
load('statistical_results.mat')

%% Write excel files

writetable(rates_table,'table_rates.xlsx')
writetable(statistical_results_classifiers,'table_statistics.xlsx')
writetable(rates_table,'table_rates.csv','Delimiter',';')

%% Generating plots 

labels = {'Bayes_{view1}','Bayes_{view2}','Bayes_{view3}','KNN_{view1}','KNN_{view2}','KNN_{view3}','Ensemble'};
bar(table2array(statistical_results_classifiers(:,1)),'w','BarWidth',0.3,'LineWidth',0.8)
xticklabels(labels)
hold on

x = 1:1:7;
neg = table2array(statistical_results_classifiers(:,3)) - table2array(statistical_results_classifiers(:,1));
sup = table2array(statistical_results_classifiers(:,3)) - table2array(statistical_results_classifiers(:,1));
errorbar(x,table2array(statistical_results_classifiers(:,1)),neg,sup,'r','LineStyle','none','LineWidth',0.5)
%title('Médias e Intervalos de Confiança dos Classificadores')
ylabel('Taxa de acerto')
