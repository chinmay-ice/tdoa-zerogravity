% ============================================================
%   TDoA Simulation – Long Flight with Sharp U-Turns
%   Area: 5 x 5 km
%   Gateways: (0,0), (4,0), (0,4) km
% ============================================================

clear; clc; close all;

% ---------------- CONSTANTS ----------------
c = 3e8;                 % speed of light (m/s)
sigma_t = 10e-9;         % 10 ns timestamp noise (SX1302)

% ---------------- GATEWAYS (meters) ----------------
G1 = [0, 0] * 1000;
G2 = [4, 0] * 1000;
G3 = [0, 4] * 1000;
gateways = [G1; G2; G3];

% ---------------- TIME ----------------
T  = 3000;               % 10x longer flight (3000 s)
dt = 1;                  % 1 Hz update
t  = 0:dt:T;
N  = length(t);

% ---------------- DRONE MOTION ----------------
rng(7);                  % reproducible randomness

x_true = zeros(N,1);
y_true = zeros(N,1);

x_true(1) = 1200;
y_true(1) = 1200;

speed = 12;              % m/s
heading = pi/3;

for k = 2:N

    % ----- Sharp U-turn every ~300 seconds -----
    if mod(k,300) == 0
        heading = heading + pi + 0.3*randn;   % near 180° turn
    else
        heading = heading + 0.03*randn;        % smooth curvature
    end

    % ----- Propagate position -----
    x_true(k) = x_true(k-1) + speed*cos(heading)*dt;
    y_true(k) = y_true(k-1) + speed*sin(heading)*dt;

    % ----- Reflect from boundaries -----
    if x_true(k) < 500 || x_true(k) > 4500
        heading = pi - heading;
    end
    if y_true(k) < 500 || y_true(k) > 4500
        heading = -heading;
    end

    x_true(k) = min(max(x_true(k),500),4500);
    y_true(k) = min(max(y_true(k),500),4500);
end

% ---------------- STORAGE ----------------
errors = zeros(N,1);
estPos = zeros(N,2);

for k = 1:N

    droneTrue = [x_true(k), y_true(k)];

    % -------- True distances --------
    d1 = norm(droneTrue - G1);
    d2 = norm(droneTrue - G2);
    d3 = norm(droneTrue - G3);

    % -------- Arrival times --------
    t1 = d1 / c;
    t2 = d2 / c;
    t3 = d3 / c;

    % -------- Add timestamp noise --------
    t1m = t1 + sigma_t*randn;
    t2m = t2 + sigma_t*randn;
    t3m = t3 + sigma_t*randn;

    % -------- TDoA --------
    dt21 = t2m - t1m;
    dt31 = t3m - t1m;

    % -------- Solve TDoA --------
    if k == 1
        p0 = mean(gateways);
    else
        p0 = estPos(k-1,:);
    end

    droneEst = solve_tdoa(gateways, dt21, dt31, p0, c);
    estPos(k,:) = droneEst;

    % -------- Error --------
    errors(k) = norm(droneEst - droneTrue);
end

% ---------------- PLOTS ----------------

figure;
plot(t, errors, 'LineWidth', 1.2);
xlabel('Time (s)');
ylabel('Position Error (m)');
title('TDoA Error vs Time – Long Flight with Sharp U-Turns');
grid on;

figure;
plot(x_true, y_true, 'g', 'LineWidth', 1.5); hold on;
plot(estPos(:,1), estPos(:,2), 'r--', 'LineWidth', 1);
scatter(gateways(:,1), gateways(:,2), 100, 'b', 'filled');
legend('True Path','Estimated Path','Gateways');
xlabel('X (m)');
ylabel('Y (m)');
title('Aggressive Curvy Flight: True vs Estimated');
grid on;
axis equal;

fprintf('\n==== LONG FLIGHT ERROR STATS ====\n');
fprintf('Mean Error = %.1f m\n', mean(errors));
fprintf('Median Error = %.1f m\n', median(errors));
fprintf('95th Percentile = %.1f m\n', prctile(errors,95));
fprintf('Max Error = %.1f m\n', max(errors));

% ============================================================
%   TDoA Solver
% ============================================================
function p = solve_tdoa(gateways, dt21, dt31, p0, c)

    cost = @(x) ...
        ( (norm(x - gateways(2,:)) - norm(x - gateways(1,:))) / c - dt21 )^2 + ...
        ( (norm(x - gateways(3,:)) - norm(x - gateways(1,:))) / c - dt31 )^2;

    p = fminsearch(cost, p0, optimset('Display','off'));
end
