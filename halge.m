%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File:     halge.m
% Usage:    the utility function to learn matrix W by HALGE algorithm
% Input:    feature matrix X and graph matrix S
% Output:   weight matrix W
% Author:   Fei Han
% Email:    fhan@mines.edu
% Date:     12/15/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function W = rlpp(X, S, p, r)
%% Solution of the optimization problem in Eq.(5) by Algorithm 3
% X = [x_1, x_2, ..., x_n] \in \Re^{d*n},
%   encoding the features for all nodes in the graph
% W = \in \Re^{d*r}, is the weight matrix to be learned
% p and r, hyper parameters
    
    %% Guarantee normalized features
    X = normc(X);
    D = diag(sum(S, 2));
    
    %% Iterations    
    W_k_1 = initW(X, D, r);
    W_k = updateW(W_k_1, X, D, S, p, r);
    obj_old = Inf;
    obj = calcObj(W_k, X, S, p);
    i = 0;
    while i < 15
        i = i + 1;
        obj_iter = ['Iteration #',num2str(i),', obj=',num2str(obj)];
        obj_old = obj;
        W_k_1 = W_k;
        W_k = updateW(W_k_1, X, D, S, p, r);
        obj = calcObj(W_k, X, S, p);
        if (abs(obj - obj_old) < 1e-4)
            break;
        end
    end
end

%% Calculates lambda by Eq. (5)
function obj = calcObj(W, X, S, p)
    n = size(X, 2);
    obj = 0;
    for i = 1 : n
        for j = 1 : n
            if S(i, j) == 0
                continue;
            end
            xx = W'*X(:,i)-W'*X(:,j);
            obj = obj + S(i, j) * (sqrt(sum(xx.*xx)))^p;
        end
    end
end

%% Updates W_k by Step 2 of Algorithm 2
function W_k = updateW(W_k_1, X, D, S, p, r)
    n = size(X, 2);
    if p == 2
        tildaS = S;
    else
        tildaS = zeros(n);
        for i = 1 : n
            for j = 1 : n
                xx = W_k_1'*X(:,i)-W_k_1'*X(:,j);
                tildaS(i, j) = 0.5 * p * S(i, j) * (sqrt(sum(xx.*xx)))^(p-2);
            end
        end
    end
    tildaD = diag(sum(tildaS, 2));
    tildaL = tildaD - tildaS;

    WW = X*tildaL*X';
    opts.tol = 1e-4;
    [W_k, ~] = eigs(WW, r, 'sm', opts);
end

%% Inits W matrix
function W = initW(X, D, r)
    [d, n] = size(X);
    assert(d>=r, 'r should be less than d');
    AA = sqrt(D) * X';
    BB = eye(n, r);
	W = (AA'*AA) \ (AA'*BB);
end
