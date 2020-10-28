function [outClass,probability] = bayesclassify( sample, TRAIN, group )
% Bayesian classifier  with different density estimation methods
%  


[gindex,groups] = grp2idx(group);
nans = find(isnan(gindex));
if ~isempty(nans)
    TRAIN(nans,:) = [];
    gindex(nans) = [];
end

ngroups = length(groups);
[n,d] = size(TRAIN);
if size(gindex,1) ~= n
    error('Bioinfo:knnclassify:BadGroupLength',...
        'The length of GROUP must equal the number of rows in TRAINING.');
elseif size(sample,2) ~= d
    error('Bioinfo:knnclassify:SampleTrainingSizeMismatch',...
        'SAMPLE and TRAINING must have the same number of columns.');
end
m = size(sample,1);

postP = zeros(m,ngroups,6);

for i=1:ngroups
    datai = TRAIN(gindex==i,:);
     ni=size(datai,1);
      
          P = prodensity(datai,sample,'P');
         postP(:,i,1) = ni*P;
          P = prodensity(datai,sample,'ks');
               postP(:,i,2) = ni*P;
          [P,fbw] = prodensity(datai,sample,'N',1);

      postP(:,i,3) = ni*P;
      postP(:,i,4:6) = ni*fbw;

end

[maxpost,outClass] = max(postP,[],2);
probability = squeeze(maxpost./sum(postP,2));
outClass = squeeze(outClass);


% Convert back to original grouping variable
if isa(group,'categorical')
    labels = getlabels(group);
    if isa(group,'nominal')
        groups = nominal(groups,[],labels);
    else
        groups = ordinal(groups,[],getlabels(group));
    end
    outClass = groups(outClass);
elseif isnumeric(group) || islogical(group)
    groups = str2num(char(groups)); %#ok
    outClass = groups(outClass);
elseif ischar(group)
    groups = char(groups);
    outClass = groups(outClass,:);
else %if iscellstr(group)
    outClass = groups(outClass);
end
