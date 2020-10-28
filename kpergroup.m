function idx = kpergroup(dindex,stdcls,ngroups,k)
% search k nearest neighbors in each class

sidx = stdcls(dindex);
for i=1:ngroups
    ind(i,:) = find(sidx==i,k);
end
idx=ind(:);