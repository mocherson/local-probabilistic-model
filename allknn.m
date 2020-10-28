function [outClass,probability] = allknn(TRAIN, group, sample, dSorted, dIndex)
% kNN classifier and its modifications
% TRAIN --- training data
% group --- the groups of training data
% sample --- test data
% dIndex --- a k column matrix, each row is the indices of the k nearest neighbors of the test sample in corresponding row of sample.
% dSorted --- a k column matrix, dSorted(i,j) is the distance between sample(i,:) and TRAIN(dIndex(i,j),:)

% grp2idx sorts a numeric grouping var ascending, and a string grouping var by order of first occurrence
[gindex,groups] = grp2idx(group);
nans = find(isnan(gindex));
if ~isempty(nans)
    TRAIN(nans,:) = [];
    gindex(nans) = [];
end
ngroups = length(groups);

if ~all(size(dSorted)==size(dIndex))
    error('the dSorted and dIndex must match!');
end
[m,K]=size(dSorted);
%[dIndex1,dSorted1] = knnsearch(TRAIN,sample,'K',K);

% find the K nearest

if K >1
    classes = gindex(dIndex);
    % special case when we have one input -- this gets turned into a
    % column vector, so we have to turn it back into a row vector.
    if size(classes,2) == 1
        classes = classes';
    end
    % count the occurrences of the classes
    % the voting kNN 
    counts1 = zeros(m,ngroups);
    for outer = 1:m
        for inner = 1:K
            counts1(outer,classes(outer,inner)) = counts1(outer,classes(outer,inner)) + 1;
        end
    end
    
    % the distance-weighted kNN 
    counts2 = zeros(m,ngroups);
    counts3 = ones(m,ngroups);
    for outer = 1:m           
        dk=dSorted(outer,K);
        d1=dSorted(outer,1);
        if(dk==d1)
            dk=d1+1;
        end
        for inner = 1:K
            counts2(outer,classes(outer,inner)) = counts2(outer,classes(outer,inner)) + (dk-dSorted(outer,inner))/(dk-d1);
            counts3(outer,classes(outer,inner)) = counts3(outer,classes(outer,inner)) + (dk-dSorted(outer,inner))/(dk-d1)*(1/inner);
        end
    end
    
    % local distribution based kNN
    counts7 = ones(m,ngroups);
    counts4 = zeros(m,ngroups);
    counts5 = zeros(m,ngroups);
    counts6 = zeros(m,ngroups);
    for i=1:m
        [center, sigma, ~ , n, r , rLPC] = classCharacter(TRAIN(dIndex(i,:),:), classes(i,:), ngroups , sample(i,:));
        cdis = sum(bsxfun(@minus,center,sample(i,:)).^2, 2);
        
        %  Monte Kalo 
        num = ones(ngroups,1);
        for g=1:ngroups
            if n(g)~=0&&n(g)~=1
            X=mvnrnd(center(g,:),sigma(g,:),1000);
            num(g)=sum(sum(bsxfun(@minus,sample(i,:),X).^2,2)<=dSorted(i,K).^2);       
            end
        end
        
        counts7(i,:)=n.*r(:,1);
        counts6(i,:)=n.*r(:,1)./num;
        counts5(i,:)=n.*r(:,2);
        counts4(i,:)=(n.*rLPC./cdis)';
    end
   
    counts4(isnan(counts4))=inf;
    
    [outClass1,probability1]=tiebreak(counts1,classes);
    [outClass2,probability2]=tiebreak(counts2,classes);
    [outClass3,probability3]=tiebreak(counts3,classes);
    [outClass4,probability4]=tiebreak(counts4,classes);
    [outClass5,probability5]=tiebreak(counts5,classes);
    [outClass6,probability6]=tiebreak(counts6,classes);
    [outClass7,probability7]=tiebreak(counts7,classes);
    outClass=[outClass1,outClass2,outClass3,outClass4,outClass5,outClass6,outClass7];
    probability=[probability1,probability2,probability3,probability4,probability5,probability6,probability7];
   
    
else
    outClass = repmat(gindex(dIndex(:,1)),1,7);
end

% Convert back to original grouping variable
if isnumeric(group)||islogical(group)
    groups = str2num(char(groups)); %#ok
    outClass = groups(outClass);
elseif iscellstr(group)
    outClass = groups(outClass);
else
    error('class name errors!');
end


function [dSorted,dIndex]  = distfun(Sample, Train, dist,K ,atttype)
%DISTFUN Calculate distances from training points to test points.
numSample = size(Sample,1);
dSorted = zeros(numSample,K);
dIndex = zeros(numSample,K);

oTrain = Train(:,logical(atttype));
oSample = Sample(:,logical(atttype));
cTrain = Train(:,~logical(atttype));
cSample = Sample(:,~logical(atttype));

switch dist
    
    case 'euclidean'  % we actually calculate the squared value
        for i = 1:numSample
            oDk = sum(bsxfun(@minus,oTrain,oSample(i,:)).^2, 2);
            cDk = sum(logical(bsxfun(@minus,cTrain,cSample(i,:))).^2, 2);
            Dk = oDk+cDk;
            [dSorted(i,:),dIndex(i,:)] = getBestK(Dk,K);
        end
        
    case 'cityblock'
        for i = 1:numSample
            Dk = sum(abs(bsxfun(@minus,Train,Sample(i,:))), 2);
            [dSorted(i,:),dIndex(i,:)] = getBestK(Dk,K);
        end
        
    case {'cosine'}
        % Normalize both the training and test data.
        normSample = sqrt(sum(Sample.^2, 2));
        normTrain = sqrt(sum(Train.^2, 2));
        if any(min(normTrain) <= eps(max(normTrain))) || any(min(normSample) <= eps(max(normSample)))
            warning('Bioinfo:knnclassify:ConstantDataForCos', ...
                ['Some points have small relative magnitudes, making them ', ...
                'effectively zero.\nEither remove those points, or choose a ', ...
                'distance other than ''cosine''.']);
        end
        Train = Train ./ normTrain(:,ones(1,size(Train,2)));
        for i = 1:numSample
            Dk = 1 - (Train * Sample(i,:)') ./ normSample(i);
            [dSorted(i,:),dIndex(i,:)] = getBestK(Dk,K);
        end
    case {'correlation'}
        % Normalize both the training and test data.
        Sample = bsxfun(@minus,Sample,mean(Sample,2));
        Train = bsxfun(@minus,Train,mean(Train,2));
        normSample = sqrt(sum(Sample.^2, 2));
        normTrain = sqrt(sum(Train.^2, 2));
        if any(min(normTrain) <= eps(max(normTrain))) || any(min(normSample) <= eps(max(normSample)))
            warning('Bioinfo:knnclassify:ConstantDataForCorr', ...
                ['Some points have small relative standard deviations, making them ', ...
                'effectively constant.\nEither remove those points, or choose a ', ...
                'distance other than ''correlation''.']);
        end
        
        Train = Train ./ normTrain(:,ones(1,size(Train,2)));
        
        for i = 1:numSample
            Dk = 1 - (Train * Sample(i,:)') ./ normSample(i);
            [dSorted(i,:),dIndex(i,:)] = getBestK(Dk,K);
        end
        
        
    case 'hamming'
        if ~all(ismember(Sample(:),[0 1]))||~all(ismember(Train(:),[0 1]))
            error('Bioinfo:knnclassify:HammingNonBinary',...
                'Non-binary data cannot be classified using Hamming distance.');
        end
        p = size(Sample,2);
        for i = 1:numSample
            Dk = sum(abs(bsxfun(@minus,Train,Sample(i,:))), 2) / p;
            [dSorted(i,:),dIndex(i,:)] = getBestK(Dk,K);
        end
        
end


% utility function to get the best K values from a vector
function [sorted,index] = getBestK(Dk,K)
% sort if needed
if K>1
    [sorted,index] = sort(Dk);
    sorted = sorted(1:K);
    index = index(1:K);
else
    [sorted,index] = min(Dk);
end


function [outClass,probability] = tiebreak(counts, classes,rule)

if nargin<3
    rule='nearest';
end

[L,outClass] = max(counts,[],2);
probability = L./sum(counts,2);
 m=size(counts,1);
 K=size(classes,2);
        for i = 1:m
            ties = counts(i,:) == L(i);
            numTies = sum(ties);
            if numTies > 1
                choice = find(ties);
                switch rule
                    case 'random'
                        % random tie break
                        
                        tb = randsample(numTies,1);
                        outClass(i) = choice(tb);
                    case 'nearest'
                        % find the use the closest element of the equal groups
                        % to break the tie
                        for inner = 1:K
                            if ismember(classes(i,inner),choice)
                                outClass(i) = classes(i,inner);
                                break
                            end
                        end
                    case 'farthest'
                        % find the use the closest element of the equal groups
                        % to break the tie
                        for inner = K:-1:1
                            if ismember(classes(i,inner),choice)
                                outClass(i) = classes(i,inner);
                                break
                            end
                        end
                end
            end
        end
