# local probabilistic model
This is the implementation of our paper:

Chengsheng Mao, Lijuan Lu, and Bin Hu. "Local probabilistic model for Bayesian classification: A generalized local classification model." Applied Soft Computing (2020): 106379.

## file description
`cro_val.m`  --    	uses cross validation to evaluate classification algorithms.

`classify_1.m`  --		an integration of a number of classification algorithms.

`allknn.m`   --     	kNN classifier and its modifications

`lpcclassify.m`	--	Local probability center classifier and its modifications

`bayesclassify.m` -- 	Bayesian classifiers with different density estimation methods

`C4_5.m`    --		C4.5 algorithm for classification

`zscorestandardize.m`--	z-score nomalization

`maxminstandardize.m` --  	maxmin normalization

`classCharacter.m`--	computes distribution characters for allkNN

`computepp.m`	--	computes the probablity that a sample belongs to its observed class for lpcclassify

`prodensity.m`	--	estimates probability density for bayesclassify

`kpergroup.m`	--	searches k nearest neighbors in each class for local learning

`bupaliverdata.mat`	-- a UCI dataset to test algorithms`


## run 
```
load('bupaliverdata.mat')  % load dataset
[acc, mse] = cro_val( data,cls, 10) % 10-fold cross-validation classification
```
