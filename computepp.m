function pp= computepp(train, group, k0, tao, q)
%  compute the probablity that a sample belongs to its observed class

if nargin<5
    q=1;
end
if nargin<4
    tao=0.1;
end

[gindex,groups] = grp2idx(group);

ntrain = size(train,1);
ngroup = length(groups);

if nargin<3
    k0=max(10,floor(0.01*ntrain));
end

pdf = zeros(ntrain,ngroup);
pw = zeros(ngroup,1);    %  the prior probability
for i=1:ngroup
    dataclassi = train(gindex==i,:);
    pw(i) = size(dataclassi,1)/ntrain;
    idx = knnsearch(dataclassi,train,'K',k0,'NSMethod','exhaustive');
    for j=1:ntrain
        pdf(j,i) = sum(exp(-sum(bsxfun(@minus,dataclassi(idx(j,:),:),train(j,:)).^2,2)/(2*tao^q)));   %  the probability density estimation in each class for each sample.
    end
end

pp = (diag(pdf(:,gindex)).*pw(gindex))./(pdf*pw);  % the probability for each sample that it belongs to its observed class