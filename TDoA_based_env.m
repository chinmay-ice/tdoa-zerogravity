% ============================================================
%   TDoA Simulation – Variable & Heavy Noise Stress Test
%   Drone Area: 10 x 10 km
%   Gateways: (0,0), (4,0), (0,4) km
% ============================================================

clear; clc; close all;

% ---------------- CONSTANTS ----------------
c = 3e8;                 % speed of light (m/s)

AREA_MIN = 500;          % meters
AREA_MAX = 4500;         % meters

% ---------------- GATEWAYS (meters) ----------------
G1 = [0, 0] * 1000;
G2 = [4, 0] * 1000;
G3 = [0, 4] * 1000;
gateways = [G1; G2; G3];

% ---------------- TIME ----------------
T  = 3000;               % long flight
dt = 1;
t  = 0:dt:T;
N  = length(t);

% ---------------- DRONE MOTION (10x10 km) ----------------
rng(11);

x_true = zeros(N,1);
y_true = zeros(N,1);

x_true(1) = 2000;
y_true(1) = 2000;

speed = 14;              % m/s
heading = pi/4;

for k = 2:N

    % Sharp U-turns
    if mod(k,250) == 0
        heading = heading + pi + 0.5*randn;
    else
        heading = heading + 0.05*randn;
    end

    x_true(k) = x_true(k-1) + speed*cos(heading)*dt;
    y_true(k) = y_true(k-1) + speed*sin(heading)*dt;

    % Reflect at boundaries
    if x_true(k) < AREA_MIN || x_true(k) > AREA_MAX
        heading = pi - heading;
    end
    if y_true(k) < AREA_MIN || y_true(k) > AREA_MAX
        heading = -heading;
    end

    x_true(k) = min(max(x_true(k),AREA_MIN),AREA_MAX);
    y_true(k) = min(max(y_true(k),AREA_MIN),AREA_MAX);
end

% ---------------- STORAGE ----------------
errors = zeros(N,1);
estPos = zeros(N,2);

% ---------------- MAIN LOOP ----------------
for k = 1:N

    droneTrue = [x_true(k), y_true(k)];

    % -------- True distances --------
    d1 = norm(droneTrue - G1);
    d2 = norm(droneTrue - G2);
    d3 = norm(droneTrue - G3);

    % -------- True arrival times --------
    t1 = d1 / c;
    t2 = d2 / c;
    t3 = d3 / c;

    % -------- Variable timestamp noise --------
    sigma = [ ...
        5e-9 + 25e-9*rand;   % Gateway 1
        8e-9 + 30e-9*rand;   % Gateway 2
        6e-9 + 20e-9*rand ]; % Gateway 3

    % Occasional bad packets
    if rand < 0.07
        sigma = sigma * (5 + 5*rand);
    end

    t1m = t1 + sigma(1)*randn;
    t2m = t2 + sigma(2)*randn;
    t3m = t3 + sigma(3)*randn;

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

% ---------------- TIME ERROR PLOT ----------------
figure;
plot(t, errors, 'LineWidth', 1.2);
xlabel('Time (s)');
ylabel('Position Error (m)');
title('TDoA Error vs Time (10×10 km Area)');
grid on;

% ---------------- TRAJECTORY ----------------
figure;
plot(x_true, y_true, 'g', 'LineWidth', 1.5); hold on;
plot(estPos(:,1), estPos(:,2), 'r--', 'LineWidth', 1);
scatter(gateways(:,1), gateways(:,2), 120, 'b', 'filled');
legend('True Path','Estimated Path','Gateways');
xlabel('X (m)');
ylabel('Y (m)');
title('True vs Estimated Trajectory (10×10 km)');
grid on;
axis equal;

fprintf('\n==== VARIABLE NOISE ERROR STATS (10×10 km) ====\n');
fprintf('Mean Error = %.1f m\n', mean(errors));
fprintf('Median Error = %.1f m\n', median(errors));
fprintf('95th Percentile = %.1f m\n', prctile(errors,95));
fprintf('Max Error = %.1f m\n', max(errors));

% ============================================================
%   TDoA ERROR HEATMAP (10 x 10 km)
% ============================================================

fprintf('\nGenerating TDoA heatmap...\n');

grid_step = 200;
xg = AREA_MIN:grid_step:AREA_MAX;
yg = AREA_MIN:grid_step:AREA_MAX;

[X, Y] = meshgrid(xg, yg);
ErrMap = zeros(size(X));

MC = 150;

for ix = 1:length(xg)
    for iy = 1:length(yg)

        droneTrue = [xg(ix), yg(iy)];
        err_mc = zeros(MC,1);

        for m = 1:MC
            d1 = norm(droneTrue - G1);
            d2 = norm(droneTrue - G2);
            d3 = norm(droneTrue - G3);

            t1 = d1 / c;
            t2 = d2 / c;
            t3 = d3 / c;

            sigma = [ ...
                5e-9 + 25e-9*rand;
                8e-9 + 30e-9*rand;
                6e-9 + 20e-9*rand ];

            if rand < 0.07
                sigma = sigma * (5 + 5*rand);
            end

            t1m = t1 + sigma(1)*randn;
            t2m = t2 + sigma(2)*randn;
            t3m = t3 + sigma(3)*randn;

            dt21 = t2m - t1m;
            dt31 = t3m - t1m;

            est = solve_tdoa(gateways, dt21, dt31, mean(gateways), c);
            err_mc(m) = norm(est - droneTrue);
        end

        ErrMap(iy,ix) = mean(err_mc);
    end
end

figure;
imagesc(xg, yg, ErrMap);
set(gca,'YDir','normal');
colorbar;
colormap(jet);
xlabel('X (m)');
ylabel('Y (m)');
title('TDoA Mean Localization Error Heatmap (10×10 km)');
hold on;
scatter(gateways(:,1), gateways(:,2), 150, 'w', 'filled');
axis equal;

% ============================================================
%   TDoA SOLVER
% ============================================================
function p = solve_tdoa(gateways, dt21, dt31, p0, c)

    cost = @(x) ...
        ((norm(x - gateways(2,:)) - norm(x - gateways(1,:))) / c - dt21)^2 + ...
        ((norm(x - gateways(3,:)) - norm(x - gateways(1,:))) / c - dt31)^2;

    p = fminsearch(cost, p0, optimset('Display','off'));
end
