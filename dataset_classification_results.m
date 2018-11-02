function [rates_vector] = dataset_classification_results(view1, view2, view3,...
                            data_labels, k)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


k_fold_idx = crossvalind('kFold',data_labels,k);
hits_vector = zeros(1,7);   % global hits array for classifiers: 
                            % 1-bayes_view1, 2-bayes_view2, 3-bayes_view3,
                            % 4-knn_view1,   5-knn_view2,   6-knn_view3, 
                            % 7-ensemble

for fold = 1:k
    
    % Generate the training and test datasets for the 3 views
    [training_view1, test_view1] = split_data(view1,k_fold_idx, fold);
    [training_view2, test_view2] = split_data(view2,k_fold_idx, fold);
    [training_view3, test_view3] = split_data(view3,k_fold_idx, fold);
    
    % Classifiers
    [posterior_bayes_view1,hits1] = bayes_classifier(training_view1,test_view1,classes);
    hits_vector(1) = hits_vector(1) + hits1;
    
    [posterior_bayes_view2,hits2] = bayes_classifier(training_view2,test_view2,classes);
    hits_vector(2) = hits_vector(2) + hits2;
    
    [posterior_bayes_view3,hits3] = bayes_classifier(training_view3,test_view3,classes);
    hits_vector(3) = hits_vector(3) + hits3;

    [posterior_knn_view1,hits4] = knn_classifier(training_view1,test_view1);
    hits_vector(4) = hits_vector(4) + hits4;
    
    [posterior_knn_view2,hits5] = knn_classifier(training_view2,test_view2);
    hits_vector(5) = hits_vector(5) + hits5;
    
    [posterior_knn_view3,hits6] = knn_classifier(training_view3,test_view3);
    hits_vector(6) = hits_vector(6) + hits6;

    [hits7] = ensemble_classifier(...
    posterior_bayes_view1,posterior_bayes_view2,posterior_bayes_view3,...
    posterior_knn_view1,posterior_knn_view2,posterior_knn_view3);
    hits_vector(7) = hits_vector(7) + hits7;

end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  FUNCTIONS

% SPLIT_DATA:
function [training_data,test_data] = split_data(data,kfold_idx, test_fold)
%split_data Splits data into training and test data
%   Uses a crossvalind vector for kFold as cvMethod to split data

test_data = data(find(kfold_idx==test_fold),:);
training_data = data(find(kfold_idx~=test_fold),:);

end
