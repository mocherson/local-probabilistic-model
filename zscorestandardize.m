function [trainoutput,testoutput] = zscorestandardize( traininput,testinput )
% z-score nomalization

 sigma = std(traininput);
 t = sigma==0;
 sigma(t)=[];
 traininput(:,t)=[];
 miu = mean(traininput);

trainoutput = bsxfun(@rdivide,bsxfun(@minus,traininput,miu),sigma);

if nargin>=2
    testinput(:,t)=[];
    testoutput = bsxfun(@rdivide,bsxfun(@minus,testinput,miu),sigma);
end