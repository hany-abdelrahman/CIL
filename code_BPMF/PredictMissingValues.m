function [res] = PredictMissingValues(train_data, nil)
    [a,b] = find(train_data~=nil);
    ind = sub2ind(size(train_data), a,b);
    train_vec = [a b train_data(ind)]';
    n_samples = size(train_vec,2);
    n_users = size(train_data,1);
    n_movies = size(train_data,2);

    N_ITERATIONS = 2;
    
    n_features = 11;
    GAMMA = 0.005;
    LAMBDA = 0.1;
    n_batches = int32(n_samples/1000);
    GLOBAL_MEAN = mean(train_vec(:,3));
    REDUCER = 0.45;
    user_bias = zeros(n_users,1);
    movie_bias = zeros(n_movies,1);
    Q = randn(n_features, n_movies);
    P = randn(n_features, n_users);
    
    [X_pred, P, Q] = StandardSVD(train_data, train_data, nil);
    P = P';
    %Q = Q';
    for i =1:N_ITERATIONS
        %pick a point at random 
        
        for cur_sample_index = 1:n_samples
            %cur_sample_index = randi(n_samples, 1,1);
            cur_sample = train_vec(:,cur_sample_index);
            cur_user = cur_sample(1);
            cur_movie = cur_sample(2);
            cur_rating = cur_sample(3);

            pred = GLOBAL_MEAN + movie_bias(cur_movie) + user_bias(cur_user) + Q(:,cur_movie)'*P(:,cur_user);
            e = cur_rating - pred;
            cur_user_vector = P(:,cur_user);
            cur_movie_vector = Q(:,cur_movie);
            Q(:,cur_movie) = cur_movie_vector + GAMMA*(e*cur_user_vector  - LAMBDA*cur_movie_vector);
            P(:,cur_user)  = cur_user_vector  + GAMMA*(e*cur_movie_vector - LAMBDA*cur_user_vector);
            user_bias(cur_user) = user_bias(cur_user) + GAMMA*(e - LAMBDA*user_bias(cur_user));
            movie_bias(cur_movie) = movie_bias(cur_movie) + GAMMA*(e - LAMBDA*movie_bias(cur_movie));
        end
        GAMMA = GAMMA*REDUCER;
    end
res = P'*Q + repmat(user_bias, 1, n_movies) + repmat(movie_bias', n_users, 1) + GLOBAL_MEAN*ones(n_users, n_movies);    
end
