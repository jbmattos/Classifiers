function [posterior_probability_matrix,rate] = knn_classifier(train_dataset,test_dataset,classes)
%knn_classifier: k - nearest neighbors classifier
%   Validation data: 20% of training data used to adjust k- value
%   Posterior prob.: (nº of neighbors of a given class)/(total nº of neighbors)
%   Return: posteriori probability and number of hits achieved with test
%           data
    
    
    % Global variables
    no_of_exemples = size(test_dataset,1);
    c = size(classes,1);
    posterior_probability_matrix = zeros(no_of_exemples,c);

    % Split training data into training and validation to adjust k-parameter
    n = size(train_dataset,1);
    p = 0.2;
    [training_idx, validation_idx] = crossvalind('HoldOut', n, p);
    [training_data,validation_data] = split_data(train_dataset,training_idx,validation_idx);
    
    % k best value's search grid
    k_min = 5;
    k_max = 60;
    best_k = get_classifier_gs(training_data,validation_data,k_min,k_max,classes);
    
    % classification
    real_classes = table2array(test_dataset(:,1));
    count_hits = 0;
    for new_exemple = 1:no_of_exemples
        
        exemple = table2array(test_dataset(new_exemple,2:end));
        distances_vector = get_euclidian_distances(exemple,train_dataset);
        
        [class,classes_freq] = classify(best_k,train_dataset,distances_vector,classes);
        
        prob = get_posterior_probabilities(best_k,classes_freq,classes);
        posterior_probability_matrix(new_exemple,:) = prob;
        
        % count classifier hits
        if class == real_classes(new_exemple)
           count_hits = count_hits+1;
        end
    end
    
    rate = count_hits;
end


%%  FUNCTIONS

% SPLIT_DATA:
function [training_data,validation_data] = split_data(data,training_idx,validation_idx)
%split_data: Splits data into training and validation data
%   Uses crossvalind vectors for HoldOut as cvMethod to split data

    training_data = data(find(training_idx==1),:);
    validation_data = data(find(validation_idx==1),:);

end

% GET_CLASSIFIER_GS:
function [best_k] = get_classifier_gs(training,validation,k_min,k_max,classes)
%get_classifier_gs: gets the best k value for knn classifier based on a
%grid search
%   Return: knn classifier k- parameter
        

    no_of_val_exemples = size(validation,1);
    k_search_size = k_max - k_min + 1;
    
    classifiers = cell(k_search_size,2);
    hits_vector = zeros(1,k_search_size);
    
    % classification of each exemple on validation data
    for val_exemple = 1:no_of_val_exemples
        
        % compute distance from exemple to each case in training data
        exemple = table2array(validation(val_exemple,2:end));
        exemple_class = string(table2array(validation(val_exemple,1)));
        distances_vector = get_euclidian_distances(exemple,training);
                
        % k- parameter grid search: generate 'n' classifiers
        idx = 1;
        for k = k_min:k_max
            [class,classes_freq] = classify(k,training,distances_vector,classes);
            classifiers{idx,1} = k;
            classifiers{idx,2} = class;
            idx = idx + 1;
        end
        
        % compute the hits for each 'n'-classifier 
        for idx = 1:k_search_size
            if classifiers{idx,2} == exemple_class
                hits_vector(idx) = hits_vector(idx) + 1;
            end
            idx = idx + 1;
        end
    end
    
    [max_hits,maximum_idx] = max(hits_vector);
    best_k = classifiers{maximum_idx,1};  
end

% GET_EUCLIDIAN_DISTANCES:
function [distances_vector] = get_euclidian_distances(exemple,training)
%get_euclidian_distances: Calculates euclidian distances between an exemple
% and a dataset of exemples.
%   Return: a vector of euclidian distances

    cases = size(training,1);
    distances_vector = zeros(1,cases);

    for case_idx = 1:cases

        train_case = table2array(training(case_idx,2:end));
        dist = sum((exemple - train_case).^2);  % euclidian distance
        distances_vector(case_idx) = dist;   

    end
end

% CLASSIFY
function [predicted_class,classes_freq] = classify(k,training,distances_vector,classes)
%classify: classify a new exemple based on knn classifier
%   Return: predicted class and a vector of the neighbors' classes
    
    % get k-neighbors indexes
    [k_min_dist,k_min_idx] = mink(distances_vector,k);
    
    % get each neighbors' class
    neighbors_classes = strings(1,k);
    for neighbor = 1:k
        neighbor_idx = k_min_idx(neighbor);
        neighbor_class = table2array(training(neighbor_idx,1));
        neighbors_classes(neighbor) = neighbor_class;
    end
    
    c = size(classes,1);
    classes_freq = zeros(1,c);
    
    % get each class frequency among neighbors
    [frequencies,Categories] = histcounts(categorical(neighbors_classes),categorical(classes));
    categories = string(Categories);
    for class = 1:c
        class_name = classes(class);
        freq_idx = find(categories == class_name);
        freq = frequencies(freq_idx);
        classes_freq(class) = freq;    
    end
    
    % chooses predicted class as the class with maximum frequency
    [max_freq,max_class_idx] = max(classes_freq);
    predicted_class = classes(max_class_idx);
end

% GET_POSTERIOR_PROBABILITIES
function [posterior_probabilities] = get_posterior_probabilities(k,classes_freq,classes)
%get_posterior_probabilities: calculates the posterior probability of all
%classes to an exemple of test data
%   Return: vector of posterior probabilities
    
    c = size(classes,1);
    posterior_probabilities = zeros(1,c);
    
    for class = 1:c
       posterior_probabilities(class) = classes_freq(class)/k;
    end
end
 