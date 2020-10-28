function [acc, mse] = cro_val( data,cls, kfold)
% cro_val uses cross validation to evaluate classification algorithms.
%   
%   data --- a matrix, each row is a sample, and each column represents a feature.
%   cls --- a column vector whose distinct values define the grouping of the rows in data.
%   kfold ---- kfold cross validation, default is 10
%   acc --- the average classification accuracy of cross validation.
%   mse --- the average mean square error of cross validation.

% load(datafile);

if nargin < 3
    kfold = 10;
end


[gindex,groups] = grp2idx(cls);
nans = find(isnan(gindex));
if ~isempty(nans)
    data(nans,:) = [];
    gindex(nans) = [];
end
ngroups = length(groups);
ndata = length(gindex);

indice = crossvalind('kfold',cls, kfold);



 acc=0;
 mse=0;
for i=1:kfold
    disp(['external fold: ', num2str(i)]);
    test = (indice==i);  % the index of test data
    train = ~test;       % the index of training data
    traindata=data(train,:);    
    testdata=data(test,:);
    [traindata, testdata]  = zscorestandardize(traindata,testdata);  % z-score normalization
%   [traindata, testdata]  = maxminstandardize(traindata,testdata);  % max-min normalization
    traincls=gindex(train);
    testcls=gindex(test);
    ntrain=length(traincls);
    ntest=length(testcls);
    
    [outclass,probability] = classify_1( testdata, traindata, traincls);
    
    comp = bsxfun(@eq,testcls,outclass);
    acc = sum(comp,1)/ntest+acc;
    
    mse = sum((double(comp(:,1:size(probability,2)))-probability).^2,1)/ntest+mse;

end

acc=acc/kfold;
mse = mse/kfold;
