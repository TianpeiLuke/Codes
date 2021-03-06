function   logCondPDF=TWCNB_getlogCondPDF(obj,test, handleNaNs)
            nTest= size(test,1);

            %log of conditional class density (P(x_i| theta))
            %Initialize to NaNs
            logCondPDF = NaN(nTest, obj.NClasses);
            
            if  isscalar(obj.Dist) && strcmp(obj.Dist,'mn')
                %The fitted probabilities are guaranteed to be non-zero.
                logpw = log(cell2mat(obj.Params));
                %cell2mat discards empty rows corresponding to empty classes
                if strcmp(handleNaNs,'on')
                    test(isnan(test)) = 0;
                end
                len = sum(test,2);
                lnCoe = gammaln(len+sum(obj.alpha))- gammaln(sum(obj.alpha))...
                    - sum(gammaln(test+ones(size(test),1)*obj.alpha),2)...
                    + sum(gammaln(ones(size(test),1)*obj.alpha),2);
                logCondPDF(:,obj.NonEmptyClasses) = bsxfun(@plus,test * logpw', lnCoe);
                
            else % 'normal', 'kernel' or 'mvmn'
                if any(obj.MVMNFS)
                    mvmnfsidx = find(obj.MVMNFS);
                    tempIdx = zeros(nTest,length(mvmnfsidx));
                    if strcmp(handleNaNs,'on')
                        for j = 1: length(mvmnfsidx)
                            [tf,tempIdx(:,j)]=ismember(test(:,mvmnfsidx(j)),obj.UniqVal{j});
                            isNaNs = isnan(test(:,mvmnfsidx(j)));
                            tempIdx(isNaNs,j) = length(obj.UniqVal{j})+1;
                        end
                    else % handleNaNs is 'off',
                        for j = 1: length(mvmnfsidx)
                            [tf,tempIdx(:,j)]=ismember(test(:,mvmnfsidx(j)),obj.UniqVal{j});
                        end
                    end
                    
                    testUnseen = any(tempIdx==0,2); % rows with unseen values
                    if any(testUnseen)
                        %remove rows with invalid input
                        warning(message('stats:NaiveBayes:BadDataforMVMN'));
                        test(testUnseen,:)=[];
                        tempIdx (testUnseen,:)=[];
                    end
                else
                    testUnseen = false(nTest,1);
                end
                
                ntestValid = size(test,1);
                
                for k = obj.NonEmptyClasses
                    logPdf =zeros(ntestValid,1);
                    if any(obj.GaussianFS)
                        param_k=cell2mat(obj.Params(k,obj.GaussianFS));
                        templogPdf = bsxfun(@plus, -0.5* (bsxfun(@rdivide,...
                            bsxfun(@minus,test(:,obj.GaussianFS),param_k(1,:)),param_k(2,:))) .^2,...
                            -log(param_k(2,:))) -0.5 *log(2*pi);
                        if strcmp(handleNaNs,'off')
                            logPdf = logPdf + sum(templogPdf,2);
                        else
                            logPdf = logPdf + nansum(templogPdf,2);
                        end
                    end%
                    
                    if any(obj.KernelFS)
                        kdfsIdx = find(obj.KernelFS);
                        for j = 1:length(kdfsIdx);
                            tempLogPdf = log(obj.Params{k,kdfsIdx(j)}.pdf(test(:,kdfsIdx(j))));
                            if strcmp(handleNaNs,'on')
                                tempLogPdf(isnan(tempLogPdf)) = 0;
                            end
                            logPdf = logPdf + tempLogPdf;
                            
                        end
                    end
                    
                    if any(obj.MVMNFS)
                        for j = 1: length(mvmnfsidx)
                            curParams = [obj.Params{k,mvmnfsidx(j)}; 1];
                            %log(1)=0;
                            tempP = curParams(tempIdx(:,j));
                            logPdf = logPdf + log(tempP);
                        end
                    end
                    
                    if any(testUnseen)
                        % saves the log of class conditional PDF for
                        % the kth class
                        logCondPDF(~testUnseen,k)= logPdf;
                        %set to -inf for unseen test value.
                        logCondPDF(testUnseen,k)=-inf;
                    else
                        logCondPDF(:,k)= logPdf;
                    end
                    
                end %loop for k
                
            end
            
        end