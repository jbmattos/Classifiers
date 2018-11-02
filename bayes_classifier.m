function [posteriori_probability_matrix, rate] = bayes_classifier(train,test,classes)
%bayes_classifier: gaussian bayesian classifier
%   Decision rule: maximum posterior probability of all classes
%   Return: posteriori probability and hit rate

exemples = size(test,1);
c = size(classes);

[class_means, class_cov] = fit_model(train, classes);
prior_probabilities = get_prior_probabilities(train);

posteriori_probability_matrix = zeros(exemples,c);

for ex = 1:exemples

    prob = get_posteriori_probabilities(prior_probabilities,class_means,...
                                         class_cov, test(ex,:));
    posteriori_probability_matrix(ex,:) = prob;

end


predicted_classes = classification(posteriori_probability_matrix,classes);
real_classes = table2array(test(:,1));

hits = sum(find(predicted_classes == real_classes));
rate = hits/exemples;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTIONS


function [class_means, class_cov] = fit_model(train, classes)
%fit_model: generates a model for the gaussian bayes classifier
%   Return: mean vector and covariance matrix of training data
%       - mean vector: simple mean
%       - covariance matrix: diagonal matrix of variances

no_of_classes = size(classes);
train_data = table2array(train(:,2:end));

class_means = cell(no_of_classes);
class_cov = cell(no_of_classes);

for c = 1: no_of_classes

    class = classes(c);
    train_class_c = train_data(find(train(:,1)==class),:);

    mi = mean(train_class_c);
    variances = var(train_class_c);
    SIGMA = diag(variances);

    class_means{c} = mi;
    class_cov{c} = SIGMA;
end

end


function [prior_probabilities] = get_prior_probabilities(train)
%get_prior_probabilities: calculates the prior probability of all classes
%   Return: vector of prior probabilities

no_of_classes = size(classes);
denominator = size(train,1);

prior_probabilities = zeros(no_of_classes);

for class = 1:no_of_classes
    
    class_name = classes(class);
    class_freq = size(find(table2array(train(:,1)) == class_name),1);

    prob = class_freq/denominator;
    prior_probabilities(class) = prob;    

end

end


function [posterior_probabilities] = get_posteriori_probabilities(...
                                prior, class_means,...
                                class_cov, exemple)
%get_posteriori_probabilities: calculates the posterior probability of all
%classes to an exemple of test data
%   Return: vector of prior probabilities

obsevation = table2array(exemple(2:end));
d = size(observation,2);
c = size(class_means,2);

likelihod_vector = zeros(1,c);
posterior_probabilities = zeros(1,c);

for class = 1:c

    sigma = class_cov{class};
    mean = class_means{class};

    sigma_inv = inv(sigma);

    likelihood = (2*pi)^(d/2) * abs(sigma_inv)^(1/2) * exp( (-1/2) *...
                (obsevation-mean)' * sigma_inv * (obsevation-mean));

    likelihod_vector(class) = likelihood;
end

denominator = sum(likelihod_vector .* prior);

for class = 1:c
    
    posterior = prior(class) * likelihod_vector(class) / denominator;
    posterior_probabilities(class) = posterior;

end

end


function [predicted_classes] = classification(posteriori_probability_matrix,classes)

    exemples = size(posteriori_probability_matrix,1);
    predicted_classes = strings(exemples,1);

    for obs = 1:exemples
        
        class_ind = max(posteriori_probability_matrix(obs,:));
        class = classes(class_idx);
        predicted_classes(obs) = class;
    end


end
