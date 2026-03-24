% ============================================================
%   TDoA Simulation – 20 km Corridor (45° Tilted)
%   6 Gateways parallel to flight path (±500 m offset)
% ============================================================

clear; clc; close all;

% ---------------- CONSTANTS ----------------
c = 3e8;                     % speed of light (m/s)

% ---------------- ROTATION ----------------
theta = 45 * pi/180;         % 45 degrees
R = [cos(theta) -sin(theta);
     sin(theta)  cos(theta)];

% ---------------- BASE CORRIDOR (UNROTATED) ----------------
% Drone path: (0,0) → (20000,0)
% Gateways parallel with ±500 m offset

gateways_base = [ ...
     0     500;
  4000    -500;
  8000     500;
 12000    -500;
 16000     500;
 20000    -500 ];

numGW = size(gateways_base,1);

% Rotate gateways
gateways = (R * gateways_base')';

% ---------------- TIME ----------------
T  = 4000;                   % seconds
dt = 1;
t  = 0:dt:T;
N  = length(t);

% ---------------- DRONE PATH (45°) ----------------
x_base = linspace(0, 20000, N);
y_base = 200 * sin(2*pi*x_base/6000);   % small lateral wiggle

path_base = [x_base' y_base'];
path_rot  = (R * path_base')';

x_true = path_rot(:,1);
y_true = path_rot(:,2);

% ---------------- STORAGE ----------------
errors = zeros(N,1);
estPos = zeros(N,2);

% ---------------- MAIN LOOP ----------------
for k = 1:N

    droneTrue = [x_true(k), y_true(k)];

    % -------- True arrival times --------
    d = vecnorm(gateways - droneTrue, 2, 2);
    t_true = d / c;

    % -------- Variable timestamp noise --------
    sigma = 5e-9 + 25e-9*rand(numGW,1);

    % Occasional bad packets
    if rand < 0.05
        sigma = sigma * (4 + 4*rand);
    end

    t_meas = t_true + sigma .* randn(numGW,1);

    % -------- TDoA (gateway 1 as reference) --------
    dt_tdoa = t_meas(2:end) - t_meas(1);

    % -------- Solve TDoA --------
    if k == 1
        p0 = droneTrue;
    else
        p0 = estPos(k-1,:);
    end

    est = solve_tdoa_multi(gateways, dt_tdoa, p0, c);
    estPos(k,:) = est;

    % -------- Error --------
    errors(k) = norm(est - droneTrue);
end

% ---------------- ERROR VS TIME ----------------
figure;
plot(t, errors, 'LineWidth', 1.2);
xlabel('Time (s)');
ylabel('Position Error (m)');
title('TDoA Error vs Time – 20 km Corridor (45°)');
grid on;

% ---------------- TRAJECTORY ----------------
figure;
plot(x_true, y_true, 'g', 'LineWidth', 2); hold on;
plot(estPos(:,1), estPos(:,2), 'r--', 'LineWidth', 1.2);
scatter(gateways(:,1), gateways(:,2), 140, 'b', 'filled');
legend('True Path','Estimated Path','Gateways');
xlabel('X (m)');
ylabel('Y (m)');
title('20 km Corridor Navigation (45° Tilted)');
grid on;
axis equal;

fprintf('\n==== 45° CORRIDOR NAVIGATION STATS ====\n');
fprintf('Mean Error = %.1f m\n', mean(errors));
fprintf('Median Error = %.1f m\n', median(errors));
fprintf('95th Percentile = %.1f m\n', prctile(errors,95));
fprintf('Max Error = %.1f m\n', max(errors));

% ============================================================
%   MULTI-GATEWAY TDoA SOLVER (N >= 4)
% ============================================================
function p = solve_tdoa_multi(gateways, dt, p0, c)

    ref = gateways(1,:);
    others = gateways(2:end,:);

    cost = @(x) sum( ...
        ((vecnorm(others - x,2,2) - norm(x - ref)) / c - dt).^2 );

    p = fminsearch(cost, p0, optimset('Display','off'));
end
