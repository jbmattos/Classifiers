function [rate] = ensemble_classifier(posterior_prob,no_of_views)
%ensemble_classifier: classifier based on bayes and knn classifiers.
%   Decision rule: maximum between all classes:
%           (1-L).P(w) + L.max(posterior_probabilities)
%   Variables: 
%       L -> number of views
%       posterior_probabilities -> of bayes and knn classifiers
%                                  applied to all views.
%   Return: number of hits achieved with test data






end
