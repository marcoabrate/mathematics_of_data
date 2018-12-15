
%% load MovieLens Data

clear

%% load 100k data
load('./dataset/ml-100k/ub_base');

% sharp
Z = accumarray([MovID,UserID],Rating);
Z = sparse(Z);
kappa = 5000;

tic
sharpZ = sharpNuc(Z,kappa);
time100k = toc;

%% load 1M data
load('./dataset/ml-1m/ml1m_base');

% sharp
Z = accumarray([MovID,UserID],Rating);
Z = sparse(Z);
kappa = 5000;

tic
sharpZ = lmoNuc(Z,kappa);
time1M = toc;

display(['sharp of 100k data takes ', num2str(time100k), ' sec'])
display(['sharp of 1M data takes ', num2str(time1M), ' sec'])