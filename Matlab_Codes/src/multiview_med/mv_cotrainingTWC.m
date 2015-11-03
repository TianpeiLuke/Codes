function [accuracy, errorlist, prob_tst, Model,history, history_tst] = ...
    mv_cotrainingTWC(Traindata, Testdata, param)
%% implementation of co-training in Avrim Blum and Tom Mitchell, 1998
%    requires the Statistical Toolbox installed for Matlab 2013b or above
%    The base model is Naive Bayes classifier
% Input: 
%   Traindata:  a struct for training set
%            .nV:  no. of views
%            .nU:  no. of unlabeled samples
%            .nL:  no. of labeleed samples
%            .d:   1 x nV, each for dimension of features in one view
%            .X_U: 1 x nV cell structure for unlabeled data
%                  each cell contains a nU x d(i) data 
%            .X_L: 1 x nV cell structure for labeled data
%                  each cell contains a nL x d(i)data set 
%            .y_L: nL x 1 labels 
%            .y_U: nU x 1 the ground truth for unlabeled data (not used in learning)
%   Testdata:  a struct for test set
%            .nTst:  no. of unlabeled samples
%            .d:   dimension of features
%            .X_Tst: 1 x nV cell structure for test data
%                    each cell contains a nTst x d(i) data set
%            .y_Tst: nTst x 1 the ground truth for test data (for error estimate)
%     param:  a struct for parameters
%            .maxIterOut:  maximum iteration for MAP computing
%            .psel:     the number of most confidence positive sample
%                       selected
%            .nsel:     the number of most confidence negative samples
%                       selected
%            .rpool:    the ratio of unlabeled samples for pooling
%            .Distribution: 1x nV cell that define the base distribution of fit in NaiveBayes 
%                      = 'normal' by default;
%                      = 'mn' for multinominal distribution 
%            .mode:    =1; for normal mode
%                      =0; no accuracy computed
%
% Output:
%   accuracy:  1x nV a array of accuracy of  prediction on test samples for
%              one view
%   prob_tst:  nTst x nV+1, probability on test samples; the last column is 
%              consensus probability on test samples
%   prob_trn:  nU x 1, (consensus) probability values on unlabeled
%              samples
%      Model:  final model for two-views 
%
%   history:  a struct for tracking the training procedure
%   history_tst: a struct for tracking the testing procedure
%   
%
%
%   Written by Tianpei Xie, 09/21/2014
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


nU = Traindata.nU;
nL = Traindata.nL;
nV = Traindata.nV;
nTst = Testdata.nTst;
d = Traindata.d;

Npool = ceil(nU*param.rpool);

maxIter = param.maxIterOut;
Dist    = param.Distribution;
%thresh = param.thre; % for definition of confidence filter


X_U = Traindata.X_U;
y_U = Traindata.y_U;
X_L = Traindata.X_L;
y_L = Traindata.y_L;

X_Tst = Testdata.X_Tst;
y_Tst = Testdata.y_Tst;

prob_tst = zeros(nTst, 1);
dev_tst = zeros(nTst, 1);

chanSel = @(s)( mod(s,nV)+1 );
psel = param.psel;
nsel = param.nsel;

Model = cell(1, nV);

errorlist = []; %cell(1,nV);
accruacy = 0; %zeros(1,nV);

history.OutlierXind = cell(1,maxIter);
history.InlierXind = cell(1, maxIter);
history.post = cell(nV, maxIter);
history.cpre = cell(nV, maxIter);
history.Model = cell(nV,maxIter);
history.OutPool = cell(1,maxIter);


history_tst.cpre = cell(nV,1);
history_tst.post = cell(nV,1);

X = cell(1,nV);
OutlierX = cell(1,nV);
InlierX = cell(1,nV);
for iv=1:nV
 X{iv} = [X_L{iv}; X_U{iv}];
end
y = [y_L; zeros(nU,1)];
y_org= y;
%OutlierY = [];
%% Initialization
display(sprintf('Initialization'));
for iv=1:nV
    % define the classifier object
    obj = struct('NClasses', 0, 'NDims',0, 'ClassLevels', cell(1), ...
         'CIsNonEmpty', [], 'Params',[], 'Prior',[], 'Dist', '', ...
         'ClassNames',cell(1), 'ClassSize', [], 'LUsedClasses', 0, ...
         'GaussianFS',[], 'MVMNFS',[], 'KernelFS',[], 'KernelWidth',[],...
         'KernelSuport', 'unbounded', 'KernelType','normal', 'UniqVal', cell(1),...
         'NonEmptyClasses',[], 'alpha', []);

     alpha = ones(1,size(X_L{iv},2))./size(X_L{iv},2); % parameter for Dirichlet prior
     obj.alpha = alpha;
     
   % fit Naive Bayes classifiers on labeled set
   Model{iv} = TWC_NaiveBayesfit(obj, X_L{iv},y_L, 'Distribution', Dist{iv}); 
   % use Normal distribution to fit (== Gaussian RBF kernel)
   history.Model{iv, 1} = Model{iv};   
   % the outlier samples are uniformly selected subset of unlabeled set 
end

   history.OutlierXind{1} = randsample([nL+1:nL+nU], Npool);
   history.OutPool{1} = setdiff([nL+1:nL+nU], history.OutlierXind{1});
   history.InlierYind{1} = [1:nL];
  
   for iv=1:nV
   OutlierX{iv} = X{iv}(history.OutlierXind{1},:);
   InlierX{iv} = X{iv}(history.InlierYind{1},:);
   end
   InlierY = y(history.InlierYind{1});
%% Co-training loop 
iter = 1;
outflag = 1;
while( outflag && iter < maxIter)   
    display(sprintf('Outer loop: iter = %d', iter));
    iter = iter + 1;
    %% prediction using Naive Bayes
    for iv=1:nV
       %iSel = chanSel(iv); % find the other view
       % prediction and find the posterior samples on the outlier set
       display(sprintf('Predicting via classifer %d...', iv)); 
       [post,cpre] = TWCNB_posterior(Model{iv},OutlierX{iv});
       history.post{iv, iter-1} = post;
       history.cpre{iv, iter-1} = cpre;
    end
    inlierInd= [];
    for iv= 1:nV
       [pcond, ind_pcond] = sort(history.post{iv, iter-1}(:,1), 'descend');
       [ncond, ind_ncond] = sort(history.post{iv, iter-1}(:,2), 'descend'); 
       if length(ind_pcond) >  psel && length(ind_ncond) >  nsel;
       % select high confidence positive and negative samples as inlier\\
         inlierInd =  union(inlierInd, union(ind_pcond(1:psel), ind_ncond(1:nsel))); 
         tempyind = history.OutlierXind{iter-1}(union(ind_pcond(1:psel), ind_ncond(1:nsel)));
         y(tempyind) =  history.cpre{iv, iter-1}(union(ind_pcond(1:psel), ind_ncond(1:nsel))); 
         % fill in psudo-labels
         ind_pcond = [];
         ind_ncond = [];
       else length(ind_pcond) >= 1;
         inlierInd =  union(inlierInd, union(ind_pcond(1), ind_ncond(1))); 
         tempyind = history.OutlierXind{iter-1}(union(ind_pcond(1), ind_ncond(1)));
         y(tempyind) =  history.cpre{iv, iter-1}(union(ind_pcond(1), ind_ncond(1))); 
         % fill in psudo-labels
         ind_pcond = [];
         ind_ncond = [];
       end
    end
       display(sprintf('# of Selected Inclier index: %d.',length(inlierInd)));
       outlierInd = setdiff([1:size(OutlierX{1},1)], inlierInd);     
       % the index of unlabeled samples selected
       OutlierOutind = history.OutlierXind{iter-1}(inlierInd);
       history.OutlierXind{iter} = setdiff(history.OutlierXind{iter-1}, OutlierOutind);
       history.InlierYind{iter} = union(history.InlierYind{iter-1}, OutlierOutind);
       
       % new Inlier set
       for iv=1:nV
         InlierX{iv} = X{iv}(history.InlierYind{iter},:); 
       end
       InlierY = y(history.InlierYind{iter});
       % new Outlier set, replenish the pool U 
      if ~isempty(history.OutlierXind{iter});
         if ~isempty(history.OutPool{iter-1}) && length(history.OutPool{iter-1})>=length(inlierInd);
          itemp = randperm(length(history.OutPool{iter-1}));
          newOutlierind = history.OutPool{iter-1}(itemp(1:length(inlierInd)));
          display(sprintf('# of Added Pooling index: %d.',length(newOutlierind)));
          history.OutlierXind{iter} = union(history.OutlierXind{iter}, newOutlierind);
          history.OutPool{iter} = setdiff(history.OutPool{iter-1},newOutlierind);    
         elseif ~isempty(history.OutPool{iter-1})
          newOutlierind = history.OutPool{iter-1};
          display(sprintf('# of Added Pooling index: %d.',length(newOutlierind)));
          history.OutlierXind{iter} = union(history.OutlierXind{iter}, newOutlierind);
          history.OutPool{iter} = setdiff(history.OutPool{iter-1},newOutlierind);     
         end
         for iv=1:nV
           OutlierX{iv} = X{iv}(history.OutlierXind{iter},:);
         end
      else % break if the unlabeled set is empty
          display(sprintf('The unlabeled set is empty. End loop...'))
          outflag = 0;
          history.OutlierXind(iter+1:end) = [];
          history.InlierXind(iter+1:end) = [];
          history.OutPool(iter+1:end) = [];
          history.post(:,iter+1:end) = [];
          history.cpre(:,iter+1:end) = [];
          history.Model(iter+1:end) = [];
          continue; 
      end
    % re-train the classifer with Inlier as training set 
    for iv=1:nV      
       display(sprintf('Retraining classifer %d...', iv)); 
        obj = struct('NClasses', 0, 'NDims',0, 'ClassLevels', cell(1), ...
         'CIsNonEmpty', [], 'Params',[], 'Prior',[], 'Dist', '', ...
         'ClassNames',cell(1), 'ClassSize', [], 'LUsedClasses', 0, ...
         'GaussianFS',[], 'MVMNFS',[], 'KernelFS',[], 'KernelWidth',[],...
         'KernelSuport', 'unbounded', 'KernelType','normal', 'UniqVal', cell(1),...
         'NonEmptyClasses',[], 'alpha', []);

     alpha = ones(1,size(X_L{iv},2))./size(X_L{iv},2); % parameter for Dirichlet prior
     obj.alpha = alpha;
       Model{iv} = TWC_NaiveBayesfit(obj, X_L{iv},y_L, 'Distribution', Dist{iv}); 
       history.Model{iv, iter} = Model{iv};  
    end
    
    
end

%% make prediction 
    for iv=1:nV
       % prediction and find the posterior samples on the other view
       display(sprintf('Predicting on test data via classifer %d...', iv)); 
       [post_tst,cpre_tst] = TWCNB_posterior(Model{iv},X_Tst{iv});
       history_tst.post{iv} = post_tst;
       history_tst.cpre{iv} = cpre_tst;
       if iv==1
         prob_tst = post_tst(:,1);
         %dev_tst = cpre_tst;
         dev_tst = dev_tst + 0.5*cpre_tst;
       end
       %0.5*sum(post_tst.*[ones(nTst,1), -ones(nTst,1)],2);
    end
    dev_tst = sign(dev_tst);
   if param.mode == 1
%        for iv =1:nV
%         errorlist{iv} = (history_tst.cpre{iv}~= y_Tst) ; 
%         accuracy(iv) = 1 - sum(errorlist{iv})/nTst;
%         display(sprintf('Accuracy for classifier %d: %.2f %%', iv, accuracy*100));
%        end
       errorlist = (dev_tst~= y_Tst) ; 
        accuracy = 1 - sum(errorlist)/nTst;
        display(sprintf('Accuracy for classifier: %.2f %%',  accuracy*100));
   elseif param.mode == 0
    errorlist = [];
    accuracy  = -1;
    display('Prediction Ends')
   end
end






