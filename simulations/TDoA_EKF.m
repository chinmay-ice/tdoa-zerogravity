% ============================================================
%   TDoA + EKF + IMU Acceleration Simulation
% ============================================================

clear; clc; close all;

% ---------------- CONSTANTS ----------------
c = 3e8;                 % speed of light (m/s)
AREA_MIN = 500;
AREA_MAX = 4500;

% ---------------- GATEWAYS ----------------
G1 = [0, 0] * 1000;
G2 = [4, 0] * 1000;
G3 = [0, 4] * 1000;
gateways = [G1; G2; G3];

% ---------------- TIME ----------------
T = 3000;
dt = 1;
t = 0:dt:T;
N = length(t);

% ---------------- DRONE MOTION ----------------
rng(11);

x_true = zeros(N,1);
y_true = zeros(N,1);
vx_true = zeros(N,1);
vy_true = zeros(N,1);

x_true(1) = 2000;
y_true(1) = 2000;

speed = 14;
heading = pi/4;

for k = 2:N
    if mod(k,250) == 0
        heading = heading + pi + 0.5*randn;
    else
        heading = heading + 0.05*randn;
    end

    vx_true(k) = speed*cos(heading);
    vy_true(k) = speed*sin(heading);

    x_true(k) = x_true(k-1) + vx_true(k)*dt;
    y_true(k) = y_true(k-1) + vy_true(k)*dt;

    if x_true(k) < AREA_MIN || x_true(k) > AREA_MAX
        heading = pi - heading;
    end
    if y_true(k) < AREA_MIN || y_true(k) > AREA_MAX
        heading = -heading;
    end
end

% ---------------- IMU ACCELERATION ----------------
ax_true = [0; diff(vx_true)/dt];
ay_true = [0; diff(vy_true)/dt];

imuNoiseStd = 0.15;     % m/s^2
ax_meas = ax_true + imuNoiseStd*randn(N,1);
ay_meas = ay_true + imuNoiseStd*randn(N,1);

% ---------------- STORAGE ----------------
estPos = zeros(N,2);
errors = zeros(N,1);

ekfPos = zeros(N,2);
ekfErr = zeros(N,1);

% ---------------- EKF INITIALIZATION ----------------
xEKF = [x_true(1); y_true(1); 0; 0];   % [x y vx vy]
PEKF = diag([100^2 100^2 10^2 10^2]);

Q = diag([0.2^2 0.2^2 0.4^2 0.4^2]);   % process noise
R = diag([25^2 25^2]);                 % TDoA measurement noise

% ---------------- MAIN LOOP ----------------
for k = 1:N

    droneTrue = [x_true(k), y_true(k)];

    % -------- TDoA MEASUREMENT --------
    d = vecnorm(gateways - droneTrue,2,2);
    t_true = d/c;

    sigma = [5e-9+25e-9*rand;
             8e-9+30e-9*rand;
             6e-9+20e-9*rand];

    if rand < 0.07
        sigma = sigma * (5 + 5*rand);
    end

    t_meas = t_true + sigma.*randn(3,1);
    dt21 = t_meas(2) - t_meas(1);
    dt31 = t_meas(3) - t_meas(1);

    if k == 1
        p0 = mean(gateways);
    else
        p0 = estPos(k-1,:);
    end

    zPos = solve_tdoa(gateways, dt21, dt31, p0, c);
    estPos(k,:) = zPos;
    errors(k) = norm(zPos - droneTrue);

    % ================= EKF =================

    % ---- Prediction (IMU driven) ----
    F = [1 0 dt 0;
         0 1 0 dt;
         0 0 1  0;
         0 0 0  1];

    B = [0.5*dt^2  0;
         0  0.5*dt^2;
         dt  0;
         0  dt];

    u = [ax_meas(k); ay_meas(k)];

    xEKF = F*xEKF + B*u;
    PEKF = F*PEKF*F' + Q;

    % ---- Measurement update (TDoA) ----
    H = [1 0 0 0;
         0 1 0 0];

    y = zPos(:) - H*xEKF;
    S = H*PEKF*H' + R;
    K = PEKF*H'/S;

    xEKF = xEKF + K*y;
    PEKF = (eye(4) - K*H)*PEKF;

    ekfPos(k,:) = xEKF(1:2)';
    ekfErr(k) = norm(xEKF(1:2)' - droneTrue);
end

% ---------------- ERROR VS TIME ----------------
figure;
plot(t, errors, 'r', 'LineWidth', 1.2); hold on;
plot(t, ekfErr, 'b', 'LineWidth', 1.2);
legend('Raw TDoA','EKF + IMU');
xlabel('Time (s)');
ylabel('Position Error (m)');
title('Error Comparison');
grid on;

% ---------------- TRAJECTORY ----------------
figure;
plot(x_true,y_true,'g','LineWidth',1.5); hold on;
plot(estPos(:,1),estPos(:,2),'r--','LineWidth',1);
plot(ekfPos(:,1),ekfPos(:,2),'b','LineWidth',1.5);
scatter(gateways(:,1),gateways(:,2),120,'k','filled');
legend('True Path','TDoA','EKF + IMU','Gateways');
axis equal; grid on;
title('Trajectory Comparison');

fprintf('\n==== RAW TDoA ERROR STATS ====\n');
fprintf('Mean %.1f | Median %.1f | 95%% %.1f | Max %.1f\n', ...
    mean(errors), median(errors), prctile(errors,95), max(errors));

fprintf('\n==== EKF + IMU ERROR STATS ====\n');
fprintf('Mean %.1f | Median %.1f | 95%% %.1f | Max %.1f\n', ...
    mean(ekfErr), median(ekfErr), prctile(ekfErr,95), max(ekfErr));

% ============================================================
%   TDoA SOLVER
% ============================================================
function p = solve_tdoa(gateways, dt21, dt31, p0, c)
    cost = @(x) ...
        ((norm(x - gateways(2,:)) - norm(x - gateways(1,:))) / c - dt21)^2 + ...
        ((norm(x - gateways(3,:)) - norm(x - gateways(1,:))) / c - dt31)^2;
    p = fminsearch(cost, p0, optimset('Display','off'));
end
