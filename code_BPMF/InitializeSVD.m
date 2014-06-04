function [U,V] = InitializeSVD( X, nil, k)
X_pred = X;
num_users = size(X_pred,1);
num_movies = size(X_pred, 2);
avg =zeros(num_movies,1);
offset = zeros(num_users,1);
br = 10; %blending ratio
observed = zeros(num_movies,1);
rated = zeros(num_users,1);

for i= 1:size(X_pred,2)
    curr_col = X_pred(:, i);
    avg(i) = mean(curr_col(curr_col~=nil));
    len = size(curr_col(curr_col~=nil),1);
    observed(i) = len;
end

global_average = mean(avg)
%Calculate better average
for i = 1:size(avg)
    r = (br*global_average + avg(i) *observed(i))/ (br + observed(i));
    avg(i) = r;
end   
for i= 1:size(X_pred,1)
    counter = 0;
    o =0;
    for j = 1:num_movies
          if X_pred(i,j)~= nil
              counter = counter + 1;
              o = o + X_pred(i, j) - avg(j);    
          end   
    end
    rated(i) = counter;
    offset(i) = o/counter;
end

offset_average = mean(offset);
for i = 1:size(offset)
    r = (br*offset_average + offset(i) *rated(i))/ (br + rated(i));
    offset(i) = r;
end  
%Calculate better offset averages


for i = 1:size(X_pred,1)
    for j= 1:size(X_pred,2)
        if(X(i,j) == nil)
            X_pred(i,j) = avg(j) + offset(i);
        end
    end
end



lambda = 125;
[U,S,V] = svd(X_pred,0);
I = eye(size(S,1));
S = S + I * lambda;
U = U * sqrt(S);
U = U(:,1:k);
V =  sqrt(S)* V ;
V = V(:,1:k);
V = V';

end