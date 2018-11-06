function [rate] = ensemble_classifier(train,test,posterior_prob,no_of_views,classes)
%ensemble_classifier: classifier based on bayes and knn classifiers.
%   Decision rule: maximum objetive function value (between all classes)
%   Objetive function: (1-L).P(w) + L.max(posterior_probabilities)
%   Variables: 
%       L -> number of views
%       posterior_probabilities -> of bayes and knn classifiers
%                                  applied to all views.
%   Return: number of hits achieved with test data


    % global variables
    no_of_exemples = size(test,1);
    no_of_classifiers = size(posterior_prob,2);
    c = size(classes,1);
    count_hits = 0;
    
    % prior probabilities (on training data)
    prior_probabilities = get_prior_probabilities(train, classes);
    
    for exemple = 1:no_of_exemples
        
        % posterior probabilities matrix for one exemple
        exemple_probs_matrix = zeros(no_of_classifiers,c);  % dimentions: (classifier,classes)
        for classifier = 1:no_of_classifiers
            classifier_exemple_prob = posterior_prob{classifier}(exemple,:);
            exemple_probs_matrix(classifier) = classifier_exemple_prob;
        end
        
        % generates the objective function for each class given an new exemple
        objective_function = zeros(1,c);
        for class = 1:c
            max_prob = max(exemple_probs_matrix(:,class));
            obj_fnc = (1 - no_of_views) * prior_probabilities(class) ...
                      + no_of_views * max_prob;
            objective_function(class) = obj_fnc;       
        end
        
        [max_prob,max_idx] = max(objective_function);
        predicted_class = classes(max_idx);
        
        real_class = table2array(test(exemple,1));
        
        % count classifier total number of hits 
        if predicted_class == real_class
           count_hits = count_hits+1;
        end
    end
    
    rate = count_hits;
end


%% FUNCTIONS

% GET_PRIOR PROBABILITIES
function [prior_probabilities] = get_prior_probabilities(train, classes)
%get_prior_probabilities: calculates the prior probability of all classes
%   Return: vector of prior probabilities

    no_of_classes = size(classes,1);
    denominator = size(train,1);

    prior_probabilities = zeros(1,no_of_classes);

    for c = 1:no_of_classes

        class = classes(c);
        class_freq = size(find(table2array(train(:,1)) == class),1);

        prob = class_freq/denominator;
        prior_probabilities(c) = prob;    

    end

end
