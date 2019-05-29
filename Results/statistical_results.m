%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% STRATIFIED N TIMES K-FOLD CROSS VALIDATION AND FRIEDMAN TEST IN 
% EVALUATING BAYESIAN, KNN AND ENSENBLE CLASSIFIERS 
%
% by: JULIANA BARCELLOS MATTOS
%
% "Image Segmentation" dataset from UCI machine learning repository
% Statistical parameters for the results
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc
load('all_datasets_rates_matrix.mat')

n = size(rates_all_datasets,1);
mean = mean(rates_all_datasets);
std = std(rates_all_datasets);
z_statistic_5 = norminv(0.975);
z_statistic_1 = norminv(0.995);

ic = zeros(size(rates_all_datasets,2),4);
for attr = 1:size(rates_all_datasets,2)
    % proportion confidence interval using t-student statistics
    ic(attr,1) = mean(attr) - z_statistic_5*(std(attr)/sqrt(n));
    ic(attr,2) = mean(attr) + z_statistic_5*(std(attr)/sqrt(n));
    ic(attr,3) = mean(attr) - z_statistic_1*(std(attr)/sqrt(n));
    ic(attr,4) = mean(attr) + z_statistic_1*(std(attr)/sqrt(n));
end

var_names = {'mean','std','ic_inf_5perc','ic_sup_5perc','ic_inf_1perc','ic_sup_1perc'};
row_names = {'bayes_view1','bayes_view2','bayes_view3','knn_view1','knn_view2','knn_view3','ensemble'};
statistical_results_classifiers = table(mean',std',ic(:,1),ic(:,2),ic(:,3),ic(:,4),'VariableNames',var_names,'RowNames',row_names);

var_names = {'bayes_view1','bayes_view2','bayes_view3','knn_view1','knn_view2','knn_view3','ensemble'};
rates_table = table(rates_all_datasets(:,1),rates_all_datasets(:,2),rates_all_datasets(:,3),rates_all_datasets(:,4),rates_all_datasets(:,5),rates_all_datasets(:,6),rates_all_datasets(:,7),'VariableNames',var_names);

[p_value_Friedman,Friedman_results,stats_F] = friedman(rates_all_datasets);
Friedman_ranks = stats_F.meanranks;
%%
save('statistical_results','rates_table','statistical_results_classifiers','p_value_Friedman','Friedman_results','Friedman_ranks')
