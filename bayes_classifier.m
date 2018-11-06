function [posteriori_probability_matrix, rate] = ...
                                      bayes_classifier(train,test,classes)
%bayes_classifier: gaussian bayesian classifier
%   Decision rule: maximum posterior probability of all classes
%   Return: posteriori probability and hit rate

    exemples = size(test,1);
    c = size(classes,1);

    [class_means, class_cov] = fit_model(train, classes);
    prior_probabilities = get_prior_probabilities(train, classes);

    posteriori_probability_matrix = zeros(exemples,c);

    for ex = 1:exemples

        exemple = test(ex,:);
        prob = get_posterior_probabilities(prior_probabilities,class_means,...
                                             class_cov, exemple);
        posteriori_probability_matrix(ex,:) = prob;

    end

    predicted_classes = classify(posteriori_probability_matrix,classes);
    real_classes = table2array(test(:,1));

    count_hits = 0;
    for ex =1:exemples
       if predicted_classes(ex) == real_classes(ex)
           count_hits = count_hits+1;
       end
    end

    rate = count_hits;

end


%% FUNCTIONS

% FIT MODEL
function [class_means, class_cov] = fit_model(train, classes)
%fit_model: generates a model for the gaussian bayes classifier
%   Return: mean vector and covariance matrix of training data for each
%   class.
%       - mean vector: simple mean
%       - covariance matrix: diagonal matrix of variances

    no_of_classes = size(classes,1);
    train_data = table2array(train(:,2:end));

    class_means = cell(1,no_of_classes);
    class_cov = cell(1,no_of_classes);

    for c = 1: no_of_classes

        class = classes(c);
        train_class_c = train_data(find(table2array(train(:,1))== class),:);

        mi = mean(train_class_c);
        variances = var(train_class_c);
        SIGMA = diag(variances);

        class_means{c} = mi;
        class_cov{c} = SIGMA;
    end

end

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

% GET_POSTERIOR_PROBABILITIES
function [posterior_probabilities] = get_posterior_probabilities(...
                                prior, class_means,...
                                class_cov, exemple)
%get_posterior_probabilities: calculates the posterior probability of all
%classes to an exemple of test data
%   Return: vector of prior probabilities

    exemple_data = table2array(exemple(1,2:end));
    d = size(exemple_data,2);
    no_of_classes = size(class_means,2);

    likelihod_vector = zeros(1,no_of_classes);
    posterior_probabilities = zeros(1,no_of_classes);

    for c = 1:no_of_classes

        sigma = class_cov{c};
        mean = class_means{c};

        warning('') % Clear last warning message
        sigma_inv = inv(sigma);
        [warnMsg, warnId] = lastwarn;
        if ~isempty(warnMsg)
            sigma_inv = pinv(sigma);
        end

        likelihood = (2*pi)^(d/2) * det(sigma_inv)^(1/2) * exp( (-1/2) *...
                    (exemple_data - mean) * sigma_inv * (exemple_data - mean)');

        likelihod_vector(c) = likelihood;
    end

    denominator = sum(likelihod_vector .* prior);
    if denominator == 0 
        denominator = 1e-7;
    end

    for c = 1:no_of_classes

        posterior = prior(c) * likelihod_vector(c) / denominator;
        posterior_probabilities(c) = posterior;

    end

end

% CLASSIFY
function [predicted_classes] = classify(posteriori_probability_matrix,classes)
%classify: classify new exemples.
%   Return: vector of predicted classes

    exemples = size(posteriori_probability_matrix,1);
    predicted_classes = strings(exemples,1);

    for obs = 1:exemples
        
        [max_value, class_idx] = max(posteriori_probability_matrix(obs,:));
        class = classes(class_idx);
        predicted_classes(obs) = class;
    end

end
