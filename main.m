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

% Load data
clear all; close all; clc
load('data_input.mat')

% Global variables
k = 10;
n = 30;


% N-Times K-fold cross validation:
% generates 30 different datasets and return its hit rate

rates_matrix = zeros(n,7);
for dataset = 1:n

rates_vector = dataset_classification_results(...
    segmentation_view,shape_view,rgb_view,data_labels,classes,k,n);

rates_matrix(dataset,:) = rates_vector;

end


