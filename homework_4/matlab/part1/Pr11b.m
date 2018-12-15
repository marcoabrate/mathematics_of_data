
%% load MovieLens Data

clear

display(['This can take few minutes.'])

%% load 100k data
load('./dataset/ml-100k/ub_base');

% proxNuc
Z = accumarray([MovID,UserID],Rating);
kappa = 5000;

tic
projZ = projNuc(Z,kappa);
time100k = toc;

%% load 1M data
load('./dataset/ml-1m/ml1m_base');
% proxNuc
Z = accumarray([MovID,UserID],Rating);
kappa = 5000;

tic
projZ = projNuc(Z,kappa);
time1M = toc;

display(['proj for 100k data takes ', num2str(time100k), ' sec'])
display(['proj for 1M data takes ', num2str(time1M), ' sec'])