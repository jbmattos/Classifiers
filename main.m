%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% STRATIFIED N TIMES K-FOLD CROSS VALIDATION AND FRIEDMAN TEST IN 
% EVALUATING BAYESIAN, KNN AND ENSENBLE CLASSIFIERS 
%
% by: JULIANA BARCELLOS MATTOS
%
% "Image Segmentation" dataset from UCI machine learning repository
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% LOAD DATA AND INITIAL SETTINGS

clear all; close all; clc
load('data_input.mat')

k = 10;
n = 30;


%

for dataset = 1:n

rates_vector = dataset_classification_results(...
    segmentation_view, shape_view, rgb_view, data_labels, classes, k);



end


