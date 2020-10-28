function [fX,fbw] = prodensity(data, X, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  prodensity estimates probability density around point X from data
%  data--- a matrix, each row is a sample, each column is a feature
%  type--- parametric, nonparametric or ksdensity. default is parametric
%  fX --- the density
%  fbw --- different bandwith selection for nonparametric KDE.
narg=numel(varargin);
if narg==0
    type = 'Parameter';
elseif narg==1
    type = varargin{1};
    h=1;   %  bandwith
else
    type = varargin{1};
    h = varargin{2};  %  bandwith
end

[n,d] = size(data);

if size(X,2)~=d
    error('the sample dimensions must agree!');
end

fX = ones(size(X,1),1);
fbw = ones(size(X,1),3);
if n==1
    fX = all(bsxfun(@eq,X,data),2)./(1-all(bsxfun(@eq,X,data),2));
    return;
end
    
miu = mean(data,1);
sigma = var(data);
S=cov(data);

switch type
    % parametric estimation---Gaussian model
    case {'parameter','Parameter','p','P'}
        sigma(sigma/max(sigma)<0.0001)=0;
        rc = (sigma==0);
        if ~any(rc)
            sigma = diag(sigma);
            fX = (2*pi)^(-d/2)*det(sigma)^(-0.5)*exp(-0.5*diag(bsxfun(@minus,X,miu)*sigma^(-1)*bsxfun(@minus,X,miu)'));
            return;
        end
        flag = all(bsxfun(@le,X(:,rc),max(data(:,rc),[],1)),2) & all(bsxfun(@ge,X(:,rc),min(data(:,rc),[],1)),2);
        sigma = diag(sigma(~rc));
        for i=1:length(flag);
            if flag(i)    
                if isempty(sigma)
                    fX(i)=inf;
                else                
                    fX(i) = (2*pi)^((-d+sum(rc))/2)*det(sigma)^(-0.5)*exp(-0.5*diag((X(i,~rc)-miu(~rc))*sigma^(-1)*(X(i,~rc)-miu(~rc))'));
                           
                end
            else fX(i)=0;
            end
        end
        
   %    fbw(:,1) = (2*pi)^(-d/2)*det(S)^(-0.5)*exp(-0.5*diag(bsxfun(@minus,X,miu)*S^(-1)*bsxfun(@minus,X,miu)'));
        
   % non-parametric estimation---Gaussian kernel     
    case {'nonparameter','N','n'}
        sigma(sigma/max(sigma)<0.0001)=0.0001*max(sigma);
        [row,col]=size(h);
        if row==1&&col==1
            h=h*sqrt(sigma);
        elseif row~=1&&col~=1
            error('the bandwidth does not match the dimension: must be a scalor or vector!');
        elseif length(h)~=d
            error('the bandwidth length does not match the dimension!');
        elseif col==1
            h=h';
        end
        
            
        for i=1:size(X,1)
            fX(i) = sum(kernel(bsxfun(@rdivide,bsxfun(@minus,X(i,:),data),h)))/(n*prod(h));
            h = 1/sqrt(n)*ones(1,d);    
            fbw(i,1) = sum(kernel(bsxfun(@rdivide,bsxfun(@minus,X(i,:),data),h)))/(n*prod(h));
            h = 1.06*n^(-0.2)*sqrt(sigma);
            fbw(i,2) = sum(kernel(bsxfun(@rdivide,bsxfun(@minus,X(i,:),data),h)))/(n*prod(h));
            h = ones(1,d)*(4*d*prod(sqrt(sigma))/(n*(2*sum(sigma.^(-2))+sum(sigma.^(-1))^2)))^(1/(d+4));
            fbw(i,3) = sum(kernel(bsxfun(@rdivide,bsxfun(@minus,X(i,:),data),h)))/(n*prod(h));
        end
        
    case {'ksdensity','ks'}
        for i=1:d
            fX=fX.*ksdensity(data(:,i),X(:,i));
        end
        
end
               
end



function  p = kernel(X)
d = size(X,2);
  p = (2*pi)^(-d/2)*exp(-0.5*(sum(X.^2,2)));
end
