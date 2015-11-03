        function [cidx, postP, logPdf] = TWCNB_getClassIdx(obj,log_condPdf)
          
            log_condPdf =bsxfun(@plus,log_condPdf, log(obj.Prior));
            [maxll, cidx] = max(log_condPdf,[],2);
            %set cidx to NaN if it is outlier
            cidx(maxll == -inf |isnan(maxll)) = NaN;
            %minus maxll to avoid underflow
            if nargout >= 2
                postP = exp(bsxfun(@minus, log_condPdf, maxll));
                %density(i) is \sum_j \alpha_j P(x_i| \theta_j)/ exp(maxll(i))
                density = nansum(postP,2); %ignore the empty classes
                %normalize posteriors
                postP = bsxfun(@rdivide, postP, density);
                if nargout >= 3
                    logPdf = log(density) + maxll;
                end
                
            end
            
        end %function getClassIdx