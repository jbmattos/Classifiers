function [posterior_probability,hits] = knn_classifier(train_dataset,test_dataset,classes)
%knn_classifier: k - nearest neighbors classifier
%   Validation data: 20% of training data used to adjust k- value
%   Posterior prob.: (nº of neighbors of a given class)/(total nº of neighbors)
%   Return: posteriori probability and hit rate


    % Split training data into training and validation to adjust k-parameter
    n = size(train_dataset,1);
    p = 0.2;
    [training_idx, validation_idx] = crossvalind('HoldOut', n, p);
    [training,validation] = split_data(train_dataset,training_idx,validation_idx);
    
    % Global variables
    no_of_exemples = size(test_dataset,1);
    c = size(classes,2);
    posterior_probability = zeros(no_of_exemples,c);

    % k best value's search grid
    k_min = 1;
    k_max = 60;
    best_k = get_classifier_gs(training,validation,k_min,k_max);


end


%%  FUNCTIONS

% SPLIT_DATA:
function [training_data,validation_data] = split_data(data,training_idx,validation_idx)
%split_data: Splits data into training and validation data
%   Uses a crossvalind vector for kFold as cvMethod to split data

    training_data = data(find(training_idx==1),:);
    validation_data = data(find(validation_idx==1),:);

end

% GET_EUCLIDIAN_DISTANCES:
function [distances_vector] = get_euclidian_distances(exemple,training)
%get_euclidian_distances: Calculates euclidian distances between an exemple
% and a dataset of exemples.
%   Return: a vector of euclidian distances

    cases = size(training,1);
    distances_vector = zeros(1,cases);

    for case_idx = 1:cases

        train_case = training(case_idx,2:end);
        dist = sum((exemple - train_case).^2);
        distances_vector(case_idx) = dist;   

    end
end

% GET_CLASSIFIER_GS:
function [best_k] = get_classifier_gs(training,validation,k_min,k_max)
%get_classifier_gs: gets the best k value for knn classifier based on a
%grid search
%   Return: knn classifier parameters
        
    no_of_val_exemples = size(validation,1);
    
    for val_exemple = 1:no_of_val_exemples

        exemple = validation(val_exemple,2:end);
        distances_vector = get_euclidian_distances(exemple,training);
        
        classifiers = cell((k_max-k_min+1),3);
        hits = zeros(1,(k_max-k_min+1));
        
        for k = k_min:k_max
            idx = 1;
            [class,neighbors_classes] = classify(k,training,distances_vector);
            classifiers{idx,1} = k;
            classifiers{idx,2} = class;
            idx = idx + 1;
        end
         
        for k = k_min:k_max
            idx = 1;
            if classifiers{k,2} == exemple(1,1)
                hits(idx) = hits(idx) + 1;
            end
            classifiers{idx,3} = hits(idx);
            idx = idx + 1;
        end
    end
    
    [max_hits,maximum_idx] = max(classifiers{:,3});
    best_k = classifiers{maximum_idx,1};  

end

% CLASSIFY
function [class,neighbors_classes] = classify(k,training,distances_vector)
%classify: classify a new exemple based on knn classifier
%   Return: predicted class and a vector of the neighbors' classes



end

%% CLASSIFIER IDEA

        classifiers = cell(k_max,4);
        hits = zeros(1,(k_max-k_min+1));
        
        for k = k_min:k_max
            [predicted_class,neighbors_classes] = classify(k,training);
            classifiers{k,1} = k;
            classifiers{k,2} = predicted_class;
            classifiers{k,3} = neighbors_classes;