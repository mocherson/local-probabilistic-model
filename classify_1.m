function [outclass,probability] = classify_1( testdata, traindata, stdcls )
% an integration of a number of classification algorithms
% testdata --- test set
% traindata --- training set
% stdcls --- the groups of samples in training set
% outclass --- the predicted groups of samples in test set
% probability --- the predicted probability that a sample in test set belongs to the group outclass
% each column of outclass and probability indicates the classification results of a classifier.

ntest = size(testdata,1);  % number of test samples 
ntrain = size(traindata,1); % number of train samples

[gindex,groups] = grp2idx(stdcls);
nans = find(isnan(gindex));
if ~isempty(nans)
    data(nans,:) = [];
    gindex(nans) = [];
end
ngroups = length(groups);


%% local classification methods
 testclassknn=[]; probabilityknn=[];  
 
pp = computepp(traindata,stdcls);   % compute the probability that a training sample belongs to its observed class
ta = tabulate(stdcls);
k=min(ta(:,2));
 [dIndex,dSorted] = knnsearch(traindata,testdata,'K',ntrain); 
 for kpc=[1,5:5:k]
     disp(['kNN: kpc=', num2str(kpc), ' k=', num2str(k)]);
     
    % kNN classifiers, the local region is selected as a neighborhood with kpc*ngroups neighbors for each class  
     [testclass1,probability1]=allknn(traindata,stdcls,testdata,dSorted(:,1:kpc*ngroups),dIndex(:,1:kpc*ngroups));  % 7 classification results

    % LPC classifiers, the local region is selected for each class that it has kpc neighbors in the corresponding class
    [testclass2,probability2]=lpcclassify(testdata,traindata,stdcls,pp,kpc);  % 7 classification results
    
    % local SVM
    testclass_local=zeros(ntest,6);  %  6 classification results
    probability_local=zeros(ntest,6);
    for i=1:ntest
        localtest = testdata(i,:);
   
        %%% search kpc*ngroups nearest neighbors for SVM
        localtrain = traindata(dIndex(i,1:kpc*ngroups),:);
        localtraincls = stdcls(dIndex(i,1:kpc*ngroups));
        
        if all(localtraincls==localtraincls(1))
            testclass_local(i,:) = localtraincls(1);
            probability_local(i,:)=1;
            continue;
        end
              
        svmmodel_rbf = svmtrain(localtraincls,localtrain,'-c 10 -h 0 -q -b 1');
        [testclass_local(i,1),~,dec]=svmpredict(localtraincls(1),localtest,svmmodel_rbf,'-q -b 1');
        probability_local(i,1) = max(dec,[],2)./sum(dec,2);
        
        svmmodel_poly = svmtrain(localtraincls,localtrain,'-t 1 -c 10 -h 0 -q');
        [testclass_local(i,2),~,dec]=svmpredict(localtraincls(1),localtest,svmmodel_poly,'-q');
        probability_local(i,2) = max(dec,[],2)./sum(dec,2);
        
        svmmodel_lin = svmtrain(localtraincls,localtrain,'-t 0 -c 10 -h 0 -q');
        [testclass_local(i,3),~,dec]=svmpredict(localtraincls(1),localtest,svmmodel_lin,'-q');
        probability_local(i,3) = max(dec,[],2)./sum(dec,2);
        
        %%%  search kpc nearest neighbors in each class for SVM
        idx = kpergroup(dIndex(i,:),stdcls,ngroups,kpc); 
        localtrain = traindata(dIndex(i,idx),:);
        localtraincls = stdcls(dIndex(i,idx));
        
        svmmodel_rbf = svmtrain(localtraincls,localtrain,'-c 10 -h 0 -q -b 1');
        [testclass_local(i,4),~,dec]=svmpredict(localtraincls(1),localtest,svmmodel_rbf,'-q -b 1');
        probability_local(i,4) = max(dec,[],2)./sum(dec,2);
        
        svmmodel_poly = svmtrain(localtraincls,localtrain,'-t 1 -c 10 -h 0 -q');
        [testclass_local(i,5),~,dec]=svmpredict(localtraincls(1),localtest,svmmodel_poly,'-q');
        probability_local(i,5) = max(dec,[],2)./sum(dec,2);
        
        svmmodel_lin = svmtrain(localtraincls,localtrain,'-t 0 -c 10 -h 0 -q');
        [testclass_local(i,6),~,dec]=svmpredict(localtraincls(1),localtest,svmmodel_lin,'-q');
        probability_local(i,6) = max(dec,[],2)./sum(dec,2);
    end   

    testclassknn = [testclassknn,testclass1,testclass2,testclass_local];
    probabilityknn = [probabilityknn,probability1,probability2,probability_local];
 end

%% SVM classifier
disp('SVM:');
    svmmodel_rbf = svmtrain(stdcls,traindata,['-h 0 -q -b 1']);  %RBF_SVM
    [testclass_SVMR(:,1),~,dec_SVM]=svmpredict(rand(ntest,1),testdata,svmmodel_rbf,'-b 1');    
    probability_SVMR(:,1) = max(dec_SVM,[],2)./sum(dec_SVM,2);
   
    testclass_SVMP=[]; probability_SVMP=[];
     svmmodel_poly = svmtrain(stdcls,traindata,['-t 1 -h 0 -q -b 1']);  %Poly_SVM
     [testclass_SVMP(:,1),~,dec_SVM]=svmpredict(rand(ntest,1),testdata,svmmodel_poly,'-q -b 1');
     probability_SVMP(:,1) = max(dec_SVM,[],2)./sum(dec_SVM,2);
     
     svmmodel_lin = svmtrain(stdcls,traindata,['-t 0 -h 0 -q -b 1']);  %Line_SVM
     [testclass_SVML(:,1),~,dec_SVM]=svmpredict(rand(ntest,1),testdata,svmmodel_lin,'-q -b 1');
     probability_SVML(:,1) = max(dec_SVM,[],2)./sum(dec_SVM,2);
 
%% naive Bayes
    disp('naive Bayes:');
     [testclass_NB,probability_NB] = bayesclassify(testdata,traindata,stdcls);  % Bayesian classifiers
     
%% C4.5
   disp('C4.5: ')
testclass_C45=[];
   testclass_C45 = C4_5(traindata',stdcls,testdata',5);
     
%% Linear Discriminant Analysis
     disp('LDA:');
     M = cov(traindata);
     E = eig(M);
     if (min(E)/max(E) < 0.0001)
         testclass_LDA=NaN(ntest,1);
     else
     testclass_LDA = classify(testdata,traindata,stdcls);  % LDA
     end
%      
%% % neural network
disp('Neural Network:'); 

  T=zeros(length(stdcls),ngroups);
  for i=1:length(stdcls)
      T(i,stdcls(i))=1;
  end

  testclassbpnn = [];
  for s=3:10
      disp(['BPNN: hidden unit number ', num2str(s)]);
    net = newff(traindata',T',s);
    net.trainParam.goal = 0.001;
    net.trainParam.epochs = 2000;
    net.trainParam.lr=0.05;
    net.trainParam.mc =0.9;
    net.trainParam.show = 50;   
    net.trainParam.showWindow = false; 
    net.trainParam.showCommandLine = false; 
    
    net = train(net,traindata',T');
    y = sim(net,testdata');
    [~,I] = min(abs(y-1));
    testclassbpnn = [testclassbpnn, I'];
  end
  
  %%
   outclass = [testclassknn,testclass_SVMR,testclass_SVMP,testclass_SVML,testclass_NB,testclass_C45',testclass_LDA,testclassbpnn];
   probability = [probabilityknn,probability_SVMR,probability_SVMP,probability_SVML,probability_NB];
%  outclass = [testclassknn,testclass_NB];

