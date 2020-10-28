function test_targets = C4_5(train_patterns, train_targets, test_patterns, inc_node)
% Classify using Quinlan's C4.5 algorithm
% Inputs:
% 	training_patterns   - Train patterns
%	training_targets	- Train targets
%   test_patterns       - Test  patterns
%	inc_node            - Percentage of incorrectly assigned samples at a node
%
% Outputs
%	test_targets        - Predicted targets
%NOTE: In this implementation it is assumed that a pattern vector with fewer than 10 unique values (the parameter Nu)
%is discrete, and will be treated as such. Other vectors will be treated as continuous
[Ni, M]		= size(train_patterns);
inc_node    = inc_node*M/100;
Nu          = 10;
%Find which of the input patterns are discrete, and discretisize the corresponding
%dimension on the test patterns
discrete_dim = zeros(1,Ni);

% for i = 1:Ni,
%     Ub = unique(train_patterns(i,:));
%     Nb = length(Ub);
%     if (Nb <= Nu),
%         %This is a discrete pattern
%         discrete_dim(i)	= Nb;
%         dist            = abs(ones(Nb ,1)*test_patterns(i,:) - Ub'*ones(1, size(test_patterns,2)));
%         [m, in]         = min(dist);
%         test_patterns(i,:)  = Ub(in);
%     end
% end

%Build the tree recursively
disp('Building tree')
tree            = make_tree(train_patterns, train_targets, inc_node, discrete_dim, max(discrete_dim), inf, 0);
%Classify test samples
disp('Classify test samples using the tree')
test_targets    = use_tree(test_patterns, 1:size(test_patterns,2), tree, discrete_dim, unique(train_targets));
%END
function targets = use_tree(patterns, indices, tree, discrete_dim, Uc)
%Classify recursively using a tree
targets = zeros(1, size(patterns,2));
if (tree.dim == 0)
    %Reached the end of the tree
    targets(indices) = tree.child;
    return
end
%This is not the last level of the tree, so:
%First, find the dimension we are to work on
dim = tree.dim;
dims= 1:size(patterns,1);
%And classify according to it
if (discrete_dim(dim) == 0),
    %Continuous pattern
    in				= indices(patterns(dim, indices) <= tree.split_loc);
    targets		= targets + use_tree(patterns(dims, :), in, tree.child(1), discrete_dim(dims), Uc);
    in				= indices(patterns(dim, indices) >  tree.split_loc);
    targets		= targets + use_tree(patterns(dims, :), in, tree.child(2), discrete_dim(dims), Uc);
else
    %Discrete pattern
    Uf				= unique(patterns(dim,:));
    for i = 1:length(Uf),
        if any(Uf(i) == tree.Nf) %Has this sort of data appeared before? If not, do nothing
            in   	= indices(patterns(dim, indices) == Uf(i));
            targets	= targets + use_tree(patterns(dims, :), in, tree.child(Uf(i)==tree.Nf), discrete_dim(dims), Uc);
        end
    end
end
%END use_tree



function tree = make_tree(patterns, targets, inc_node, discrete_dim, maxNbin, lastdim, base)
%Build a tree recursively
if size(targets,1)>1
    targets=targets';
end

[Ni, L]    					= size(patterns);
Uc         					= unique(targets);
 tree.level      = 1;
 tree.dim	    = 0;
%tree.level                  = 0;
%tree.child(1:maxNbin)	= zeros(1,maxNbin);
tree.split_loc				= inf;
% if isempty(patterns),
%     return
% end
%When to stop: If the dimension is one or the number of examples is almost
%in the same class 
[most,m] = mode(targets);
if ((inc_node >= L)|| m/length(targets)>=0.9 || (length(Uc) == 1) || Ni == 0),
    tree.Nf         = [];
    tree.split_loc  = [];
    tree.child	 	= most;
    tree.level      = 1;
    tree.dim	    = 0;
    return
end
%Compute the node's I
% for i = 1:length(Uc),
%     Pnode(i) = length(find(targets == Uc(i))) / L;
% end

Pnode = sum(bsxfun(@eq,targets',Uc))./L;
Inode = -sum(Pnode.*log(Pnode)/log(2));
%For each dimension, compute the gain ratio impurity
%This is done separately for discrete and continuous patterns
delta_Ib    = zeros(1, Ni);
split_loc	= inf(1, Ni);
for i = 1:Ni,
    data	= patterns(i,:);
    Ud      = unique(data);
    Nbins	= length(Ud);
    if Nbins == 1
        continue;
    end
    if (discrete_dim(i)),
        %This is a discrete pattern
        P	= zeros(length(Uc), Nbins);
        for j = 1:length(Uc),
            for k = 1:Nbins,
                P(j,k) 	= sum((targets == Uc(j)) & (data == Ud(k)));
            end
        end
        Pk          = sum(P);
        P           = P./repmat(Pk,length(Uc),1);
        Pk          = Pk/L;
        info        = sum(-P.*log(eps+P)/log(2));
        Pk = Pk+eps*((Pk==0)-(Pk==1));
        delta_Ib(i) = (Inode-sum(Pk.*info))/-sum(Pk.*log(Pk)/log(2));
    else
        %This is a continuous pattern
        Ud  = sort(Ud);
        P	= zeros(length(Uc), 2);
          %Sort the patterns
%           [sorted_data, indices] = sort(data);
%           sorted_targets = targets(indices);          %Calculate the information for each possible split
          I	= zeros(1, Nbins-1);
          for j = 1:Nbins-1,
              %for k =1:length(Uc),
              %    P(k,1) = sum(sorted_targets(1:j)        == Uc(k));
              %    P(k,2) = sum(sorted_targets(j+1:end)    == Uc(k));
              %end
              P(:, 1) = hist(targets(data <= Ud(j)) , Uc);
              P(:, 2) = hist(targets(data > Ud(j)) , Uc);
              Pk      = sum(P);   
              Ps	  = Pk/L;                   
              P1      = repmat(Pk, length(Uc), 1);
              P       = P./P1;
              P       = P + eps*(P==0);              
              info	= sum(-P.*log(P)/log(2));
              Ps = Ps+eps*((Ps==0)-(Ps==1));
              I(j)	= (Inode - sum(info.*Ps))/-sum(Ps.*log(Ps)/log(2));
          end
          [delta_Ib(i), s] = max(I);
          split_loc(i) = mean([Ud(s),Ud(s+1)]);
    end
end%Find the dimension minimizing delta_Ib
[m, dim]    = max(delta_Ib);
% dims        = 1:Ni; 
dims        = [1:dim-1,dim+1:Ni];   % delete the dim that has been selected.
if lastdim<=dim
    tree.dim = dim+1;
else
tree.dim    = dim;
end
%Split along the 'dim' dimension
Nf		= unique(patterns(dim,:));
Nbins	= length(Nf);
tree.Nf = Nf;
tree.split_loc      = split_loc(dim);
%If only one value remains for this pattern, one cannot split it.
if (Nbins == 1)
    H				= hist(targets, length(Uc));
    [m, largest] 	= max(H);
    tree.dim        =0;
    tree.Nf         = [];
    tree.split_loc  = [];
    tree.child	 	= Uc(largest);
    tree.level    =1;
    return
end
if (discrete_dim(dim)),
    %Discrete pattern
    for i = 1:Nbins,
        indices         = patterns(dim, :) == Nf(i);
        tree.child(i)	= make_tree(patterns(dims, indices), targets(indices), inc_node, discrete_dim(dims), maxNbin, dim, base);
    end   
else
    %Continuous pattern
    indices1		   	= patterns(dim,:) <= split_loc(dim);
    indices2	   		= patterns(dim,:) > split_loc(dim);
    if any(indices1) && any(indices2)
        tree.child(1)	= make_tree(patterns(dims, indices1), targets(indices1), inc_node, discrete_dim(dims), maxNbin, dim, base+1);
        tree.child(2)	= make_tree(patterns(dims, indices2), targets(indices2), inc_node, discrete_dim(dims), maxNbin, dim, base+1);
    else
        H				= hist(targets, length(Uc));
        [m, largest] 	= max(H);
        tree.child	 	= Uc(largest);
        tree.dim                = 0;
        tree.level      =1;
    end
end
tree.level = max([tree.child.level])+1;
