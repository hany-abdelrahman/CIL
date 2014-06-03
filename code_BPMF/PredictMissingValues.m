function X_pred = PredictMissingValues(X, nil)
% Predict missing entries in matrix X based on known entries. Missing
% values in X are denoted by the special constant value nil.

% your collaborative filtering code here!
X_pred = X;
for i = 1:size(X_pred,2)
   cur_col = X_pred(:,i);
   mis_val = mean(cur_col(cur_col ~= nil));
   X_pred((X_pred(:,i) == nil),i) = mis_val;
end

k = 5;
lambda = 350;
[U,D,V] = svd(X_pred,0);
D = D + lambda*eye(size(D,1));
D_sqrt = sqrt(D);
U_p = U*D_sqrt;
U_p = U_p(:,1:k);

V_p = D_sqrt*V;
V_p = V_p(:,1:k);
V_p = V_p';
for i = 1:size(X_pred,1)
    for j = 1:size(X_pred,2)
        if(X(i,j) == nil)
            X_pred(i,j) = U_p(i,:)*V_p(:,j);            
        end
    end
end
