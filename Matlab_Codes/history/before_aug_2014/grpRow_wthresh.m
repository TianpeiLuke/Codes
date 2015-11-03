function v = grpRow_wthresh(b, lambda)

[n, C] = size(b);
for i = 1:n    
        tmp = 1 - lambda/norm(b(i,:)); %  tmp = 1 - lambda/\|b\|_{2}
        tmp = (tmp + abs(tmp))/2; %if  tmp = 1 - lambda/\|b\|_{2} <0, tmp = 0; otherwise tmp = 1 - lambda/\|b\|_{2}
        v(i,:) = tmp*b(i,:);    % row sparse solution 
end
