function [ x_out ] = filt( x )
 
  ind = find(isnan(x));
  x(ind) = zeros(length(ind),1);

  x_out = x;
end
