function [ center,sigma, LPC, n, r ,rLPC] = classCharacter( traindata, group, nclass, X)
%  classCharacter computes some distribution characters of each class from traindata
%   
%   traindata --- the training data
%   group --- the corresponding class of training data
%   nclass--- the number of classes?
%   center--- the center of each class
%   n---- the number of samples in each class
%   r---- the density around point X
%   rLPC --- a discription of class variation

d=size(traindata,2);
center=Inf(nclass,d);
sigma=zeros(nclass,d);
LPC=Inf(nclass,d);
n=zeros(nclass,1);
r=zeros(nclass,4);
rLPC=zeros(nclass,1);
for i=1:nclass
    datai = traindata(group==i,:);
    n(i) = size(datai,1);
    if n(i)>=1
    center(i,:)=mean(datai,1);
    sigma(i,:)=std(datai,1,1);
    dist = sum(bsxfun(@minus,datai,center(i,:)).^2, 2);


    rLPC(i) = sum(dist)/n(i);
    if rLPC(i) ==0
        rLPC(i) = eps(0);
    end
%    r(i) = evaluate(kde(datai','LCV'),X');
    r(i,1) = prodensity(datai,X);
    f=ones(d,1);
    u=ones(d,1);
    for j=1:d
        [f(j),~,u(j)]=ksdensity(datai(:,j),X(j));
    end
    r(i,2) = prodensity(datai,X,'N',u);
    r(i,3) = prod(f);
     

    end

end

