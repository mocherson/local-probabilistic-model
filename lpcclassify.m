function [outClass,probability] = lpcclassify(sample, TRAIN, group, pp, K)
% Local probability center classifier and its modifications. 
 
[gindex,groups] = grp2idx(group);
 nans = find(isnan(gindex));
 if ~isempty(nans)
     TRAIN(nans,:) = [];
     gindex(nans) = [];
 end
 
 
 ngroups = length(groups);
 [N,d]=size(TRAIN);
 
 dataclass=cell(1,ngroups);
 n=zeros(ngroups,1);
 pro=cell(1,ngroups);
 for i=1:ngroups
     dataclass{i} = TRAIN(gindex==i,:); 
     n(i) = sum(gindex==i);
     pro{i} = pp(gindex==i);
 end
% 

nsamples = size(sample,1);
probability=zeros(nsamples,7);
CAPdist = zeros(nsamples,ngroups);
capindex1 = zeros(nsamples,ngroups);
capindex2 = zeros(nsamples,ngroups);
capindex3 = zeros(nsamples,ngroups);
capindex4 = zeros(nsamples,ngroups);
capindex5 = zeros(nsamples,ngroups);
LPCdist = zeros(nsamples,ngroups);

for i=1:ngroups
    [idx,dSorted] = knnsearch(dataclass{i},sample,'K',K,'Distance','euclidean');
    nc = size(dataclass{i},1);

    for j=1:nsamples 
       
        %%%%  local center
        datai = dataclass{i}(idx(j,:),:);
        wi = pro{i}(idx(j,:));
        center = mean(datai,1);
        sigma = std(datai,1,1);
        
        % CAP
        CAPdist(j,i) = sum((center-sample(j,:)).^2);
        
        % LUA
        capindex3(j,i) = dSorted(j,end); 

        %  LGA
          X=mvnrnd(center,sigma,1000);
          num=sum(sum(bsxfun(@minus,sample(j,:),X).^2,2)<=dSorted(j,end).^2); 
          capindex1(j,i) = prodensity(datai,sample(j,:));
          capindex2(j,i) = capindex1(j,i)/num;

        %  LCA
          capindex4(j,i) = prodensity(datai,sample(j,:),'N',1.06/K^0.2);

          capindex5(j,i) = prodensity(datai,sample(j,:),'ks');
        
         
        
        %%%% Local probability center
         a=sum(bsxfun(@times,datai,wi),1);
         b=sum(wi);
         LPcenter = a/b;
         LPCdist(j,i) = sum((LPcenter-sample(j,:)).^2);
  
    end
end

[CAPv,CAPI] = min(CAPdist,[],2);            probability(:,1) = exp(-CAPv/2)./sum(exp(-CAPdist/2),2);
[CAPv1,CAPplus1] = max(capindex1,[],2);     probability(:,2) = CAPv1./sum(capindex1,2);
[CAPv2,CAPplus2] = max(capindex2,[],2);     probability(:,3) = CAPv2./sum(capindex2,2);
[CAPv3,CAPplus3] = min(capindex3,[],2);    
probability(CAPv3==0,4)=1;  probability(CAPv3~=0,4) = CAPv3(CAPv3~=0).^(-d)./sum(capindex3(CAPv3~=0,:).^(-d),2); 
[CAPv4,CAPplus4] = max(capindex4,[],2);     probability(:,5) = CAPv4./sum(capindex4,2);
[CAPv5,CAPplus5] = max(capindex5,[],2);     probability(:,6) = CAPv5./sum(capindex5,2);
[LPCv,LPCI] = min(LPCdist,[],2);            probability(:,7) = exp(-LPCv)./sum(exp(-LPCdist),2);



if isnumeric(group)||islogical(group)
    groups = str2num(char(groups)); %#ok
    outClass = groups([CAPI,CAPplus1,CAPplus2,CAPplus3,CAPplus4,CAPplus5,LPCI]);
elseif iscellstr(group)
    outClass = groups([CAPI,CAPplus1,CAPplus2,CAPplus3,CAPplus4,CAPplus5,LPCI]);
else
    error('class name errors!');
end
end

