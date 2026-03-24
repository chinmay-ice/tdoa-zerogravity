% ============================================================
%   RSSI Trilateration Monte-Carlo (Stable lsqnonlin version)
% ============================================================

clear; clc; close all;

% --------- SIMULATION PARAMETERS ---------
N = 5000;

Pt = -30;                 % RSSI at 1 m (dBm)
n = 2.4;                  % path loss exponent
sigmaNoise = 2.5;         % RSSI noise std-dev (dB)

% --------- ANCHOR POSITIONS (meters, ENU frame) ---------
A = [0 0];
B = [4000 0];
C = [0 4000];
anchors = [A; B; C];

errors = zeros(N,1);

for k = 1:N

    % --------- RANDOM DRONE POSITION (INSIDE TRIANGLE) ---------
    u = rand; v = rand;
    if u + v > 1
        u = 1-u; v = 1-v;
    end
    droneTrue = A + u*(B-A) + v*(C-A);

    % --------- TRUE DISTANCES ---------
    dTrue = vecnorm(anchors - droneTrue, 2, 2);

    % --------- GENERATE RSSI ---------
    rssi = Pt - 10*n.*log10(dTrue) + sigmaNoise*randn(3,1);

    % --------- INVERT RSSI → ESTIMATED DISTANCE ---------
    dEst = 10.^((Pt - rssi) ./ (10*n));

    % --------- NONLINEAR MULTILATERATION ---------
    p0 = mean(anchors);          % initial guess = centroid
    droneEst = solve_nls_ls(anchors, dEst, p0);

    % --------- LOCALIZATION ERROR ---------
    errors(k) = norm(droneEst - droneTrue);

end


% --------- RESULTS ---------
fprintf('\n==== RESULTS (lsqnonlin) ====\n');
fprintf('Mean Error = %.1f m\n',   mean(errors));
fprintf('Median Error = %.1f m\n', median(errors));
fprintf('95th Percentile = %.1f m\n', prctile(errors,95));

figure;
histogram(errors,60);
xlabel('Localization Error (m)');
ylabel('Count');
title('RSSI Trilateration Error Distribution');
grid on;


% ============================================================
%   Nonlinear Least Squares Trilateration Solver
%   — dimension-safe, no weird syntax —
% ============================================================
function p = solve_nls_ls(anchors, dEst, p0)

    anchors = double(anchors);
    dEst    = double(dEst(:));       % column vector
    p0      = double(p0(:)');        % row [x y]

    % residuals = predicted_range − estimated_range
    residual = @(x) ...
        vecnorm(anchors - repmat(x(:).', size(anchors,1), 1), 2, 2) - dEst;

    opts = optimoptions('lsqnonlin', ...
        'Display','off', ...
        'TolFun',1e-8, ...
        'TolX',1e-8);

    p = lsqnonlin(residual, p0, [], [], opts);

end
