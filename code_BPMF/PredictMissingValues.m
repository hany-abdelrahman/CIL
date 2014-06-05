function [res] = PredictMissingValues(train_data, val_data, nil)
    %val_data = train_data;
    scaling_coeff = 5;
    ind = train_data==nil;
    train_data = (train_data + 10)./scaling_coeff;
    train_data(ind) = nil;
    ind = val_data==nil;
    val_data = (val_data + 10)./scaling_coeff;
    val_data(ind) = nil;
    
    [a,b] = find(train_data~=nil);
    ind = sub2ind(size(train_data), a,b);
    train_vec = [a b train_data(ind)];

    [a,b] = find(val_data~=nil);
    ind = sub2ind(size(val_data), a,b);
    probe_vec = [a b val_data(ind)];
    restart = 1;
    if restart==1 
      restart=0;
      epsilon=50; % Learning rate 
      lambda  = 0.01; % Regularization parameter 
      momentum=0.0; 

      epoch=1; 
      maxepoch=10; 

      mean_rating = mean(train_vec(:,3)); 

      pairs_tr = length(train_vec); % training data 
      pairs_pr = length(probe_vec); % validation data 

      numbatches= 9; % Number of batches  
      num_m = size(train_data,2);   % Number of movies 
      num_p = size(train_data,1);  % Number of users 
      num_feat = 3; % Rank 10 decomposition 

      [U_init, V_init] = InitializeSVD(train_data, nil, num_feat);
      w1_M1     = V_init';%0.1*randn(num_m, num_feat); % Movie feature vectors
      w1_P1     = U_init; %0.1*randn(num_p, num_feat); % User feature vecators
      w1_M1_inc = zeros(num_m, num_feat);
      w1_P1_inc = zeros(num_p, num_feat);

    end

    for epoch = epoch:maxepoch
      rr = randperm(pairs_tr);
      train_vec = train_vec(rr,:);
      clear rr 

      for batch = 1:numbatches
        N = int32(pairs_tr/numbatches)-1;
        N = double(N);

        aa_p   = double(train_vec((batch-1)*N+1:batch*N,1));
        aa_m   = double(train_vec((batch-1)*N+1:batch*N,2));
        rating = double(train_vec((batch-1)*N+1:batch*N,3));

        rating = rating-mean_rating; % Default prediction is the mean rating. 

        %%%%%%%%%%%%%% Compute Predictions %%%%%%%%%%%%%%%%%
        %pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);
        pred_out = Kernel(w1_M1(aa_m,:),w1_P1(aa_p,:));
        f = sum( (pred_out - rating).^2 + 0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));

        %%%%%%%%%%%%%% Compute Gradients %%%%%%%%%%%%%%%%%%%
        IO = repmat(2*(pred_out - rating),1,num_feat);
        Ix_m=IO.*w1_P1(aa_p,:) + lambda*w1_M1(aa_m,:);
        Ix_p=IO.*w1_M1(aa_m,:) + lambda*w1_P1(aa_p,:);

        dw1_M1 = zeros(num_m,num_feat);
        dw1_P1 = zeros(num_p,num_feat);

        for ii=1:N
          dw1_M1(aa_m(ii),:) =  dw1_M1(aa_m(ii),:) +  Ix_m(ii,:);
          dw1_P1(aa_p(ii),:) =  dw1_P1(aa_p(ii),:) +  Ix_p(ii,:);
        end

        %%%% Update movie and user features %%%%%%%%%%%

        w1_M1_inc = momentum*w1_M1_inc + epsilon*dw1_M1/N;
        w1_M1 =  w1_M1 - w1_M1_inc;

        w1_P1_inc = momentum*w1_P1_inc + epsilon*dw1_P1/N;
        w1_P1 =  w1_P1 - w1_P1_inc;
      end 

      %%%%%%%%%%%%%% Compute Predictions after Paramete Updates %%%%%%%%%%%%%%%%%
      pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);
      f_s = sum( (pred_out - rating).^2 + 0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));
      err_train(epoch) = sqrt(f_s/N);

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%% Compute predictions on the validation set %%%%%%%%%%%%%%%%%%%%%% 
      NN=pairs_pr;

      aa_p = double(probe_vec(:,1));
      aa_m = double(probe_vec(:,2));
      rating = double(probe_vec(:,3));

      pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2) + mean_rating;
      ff = find(pred_out>scaling_coeff); pred_out(ff)=scaling_coeff; % Clip predictions 
      ff = find(pred_out<0); pred_out(ff)=0;

      err_valid(epoch) = sqrt(mean((scaling_coeff*(pred_out- rating)).^2));
      fprintf(1, 'epoch %4i batch %4i Training RMSE %6.4f  Test RMSE %6.4f  \n', epoch, batch, err_train(epoch), err_valid(epoch));
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    end 
    res = scaling_coeff*((w1_P1*w1_M1') + mean_rating)-10;
    res(find(res>9.5))=9.5;
    res(find(res<-9.5))=-9.5;
end