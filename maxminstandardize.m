function [trainoutput,testoutput] = maxminstandardize( traininput,testinput )
%maxmin normalization

mi=min(traininput);
ma=max(traininput);


trainoutput=bsxfun(@rdivide,bsxfun(@minus,traininput,mi),(ma-mi));
testoutput=bsxfun(@rdivide,bsxfun(@minus,testinput,mi),(ma-mi));