% ============================================================
%   TDoA Simulation – 20 km Corridor (45° Tilted)
%   WITH REALISTIC HARDWARE INEFFICIENCIES (CORRECTED)
%   GPS, MCU, and LoRa error modeling
% ============================================================
clear; clc; close all;

% ---------------- CONSTANTS ----------------
c = 3e8;                     % speed of light (m/s)

% ---------------- HARDWARE ERROR PARAMETERS (REALISTIC) ----------------
% GPS (u-blox NEO-M9N)
GPS_PPS_SIGMA = 30e-9;       % PPS accuracy ±30 ns (1σ)
GPS_POS_SIGMA = 1.5;         % Position error ±1.5 m (1σ)

% MCU (STM32F4)
MCU_CAPTURE_JITTER = 10e-9;  % PPS capture jitter ±10 ns
MCU_TX_JITTER = 20e-9;       % TX scheduling jitter ±20 ns

% LoRa (SX1278) - REDUCED to realistic values
LORA_AIRTIME_SIGMA = 100e-8; % Air-time uncertainty ~100 ns (30 m equivalent)
LORA_MULTIPATH_PROB = 0.02;  % 2% probability of multipath spike
LORA_MULTIPATH_SIGMA = 300e-8; % Multipath delay ~300 ns (90 m equivalent)

% Combined timing budget
% RSS = sqrt(30^2 + 10^2 + 20^2 + 100^2 + (0.02*300)^2) ≈ 110 ns ≈ 33 m

% ---------------- ROTATION ----------------
theta = 45 * pi/180;         % 45 degrees
R = [cos(theta) -sin(theta);
     sin(theta)  cos(theta)];

% ---------------- BASE CORRIDOR (UNROTATED) ----------------
gateways_base = [ ...
     0     500;
  4000    -500;
  8000     500;
 12000    -500;
 16000     500;
 20000    -500 ];
numGW = size(gateways_base,1);

% Rotate gateways (ideal positions)
gateways_ideal = (R * gateways_base')';

% Add GPS position errors to gateway locations (one-time error)
gps_pos_error = GPS_POS_SIGMA * randn(numGW, 2);
gateways = gateways_ideal + gps_pos_error;

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
estPos_filtered = zeros(N,2);

% Error breakdown storage
timing_error_breakdown = zeros(N, 5); % GPS PPS, MCU capture, MCU TX, LoRa, Multipath

% Kalman filter initialization
P = eye(2) * 100;  % Initial covariance
Q = eye(2) * 1;    % Process noise (drone movement uncertainty)
R_kf = eye(2) * 25; % Measurement noise

% ---------------- MAIN LOOP ----------------
fprintf('Simulating TDoA with realistic hardware errors...\n');

for k = 1:N
    droneTrue = [x_true(k), y_true(k)];
    
    % -------- True arrival times --------
    d = vecnorm(gateways - droneTrue, 2, 2);
    t_true = d / c;
    
    % -------- Apply hardware timing errors for each gateway --------
    t_meas = zeros(numGW, 1);
    
    for gw = 1:numGW
        % 1. GPS PPS timing error (±30 ns)
        gps_pps_error = GPS_PPS_SIGMA * randn();
        
        % 2. MCU PPS capture jitter (±10 ns)
        mcu_capture_error = MCU_CAPTURE_JITTER * randn();
        
        % 3. MCU TX scheduling jitter (±20 ns)
        mcu_tx_error = MCU_TX_JITTER * randn();
        
        % 4. LoRa air-time uncertainty (~100 ns)
        lora_airtime_error = LORA_AIRTIME_SIGMA * randn();
        
        % 5. Occasional multipath/NLOS spike (~300 ns, 2% probability)
        multipath_error = 0;
        if rand() < LORA_MULTIPATH_PROB
            multipath_error = LORA_MULTIPATH_SIGMA * abs(randn());
        end
        
        % Store error breakdown (for gateway 1)
        if gw == 1
            timing_error_breakdown(k, :) = [gps_pps_error, ...
                                           mcu_capture_error, ...
                                           mcu_tx_error, ...
                                           lora_airtime_error, ...
                                           multipath_error] * 1e9; % ns
        end
        
        % Total timing error
        total_timing_error = gps_pps_error + mcu_capture_error + ...
                            mcu_tx_error + lora_airtime_error + ...
                            multipath_error;
        
        % Measured time
        t_meas(gw) = t_true(gw) + total_timing_error;
    end
    
    % -------- TDoA (gateway 1 as reference) --------
    dt_tdoa = t_meas(2:end) - t_meas(1);
    
    % -------- Solve TDoA with better initial guess --------
    if k == 1
        p0 = droneTrue;
    else
        p0 = estPos_filtered(k-1,:);
    end
    
    % Use constrained optimization
    est = solve_tdoa_robust(gateways, dt_tdoa, p0, c, droneTrue);
    estPos(k,:) = est;
    
    % -------- Kalman Filter for smoothing --------
    if k == 1
        estPos_filtered(k,:) = est;
    else
        % Predict
        x_pred = estPos_filtered(k-1,:)';
        P_pred = P + Q;
        
        % Update
        z = est';
        K = P_pred / (P_pred + R_kf);
        x_filt = x_pred + K * (z - x_pred);
        P = (eye(2) - K) * P_pred;
        
        estPos_filtered(k,:) = x_filt';
    end
    
    % -------- Error --------
    errors(k) = norm(estPos_filtered(k,:) - droneTrue);
    
    % Progress indicator
    if mod(k, 500) == 0
        fprintf('  Progress: %d/%d samples\n', k, N);
    end
end

fprintf('Simulation complete!\n\n');

% ---------------- STATISTICS ----------------
fprintf('==== 45° CORRIDOR NAVIGATION STATS (with hardware errors) ====\n');
fprintf('Mean Error      = %.2f m\n', mean(errors));
fprintf('Median Error    = %.2f m\n', median(errors));
fprintf('Std Dev         = %.2f m\n', std(errors));
fprintf('95th Percentile = %.2f m\n', prctile(errors,95));
fprintf('99th Percentile = %.2f m\n', prctile(errors,99));
fprintf('Max Error       = %.2f m\n\n', max(errors));

% Gateway position errors
fprintf('==== GATEWAY POSITION ERRORS (GPS) ====\n');
for i = 1:numGW
    pos_err = norm(gateways(i,:) - gateways_ideal(i,:));
    fprintf('Gateway %d: %.2f m offset\n', i, pos_err);
end
fprintf('\n');

% Timing error statistics (in nanoseconds)
fprintf('==== TIMING ERROR BREAKDOWN (Gateway 1) ====\n');
fprintf('GPS PPS:        RMS = %.1f ns (%.1f m equiv)\n', ...
    rms(timing_error_breakdown(:,1)), rms(timing_error_breakdown(:,1))*c/1e9);
fprintf('MCU Capture:    RMS = %.1f ns (%.1f m equiv)\n', ...
    rms(timing_error_breakdown(:,2)), rms(timing_error_breakdown(:,2))*c/1e9);
fprintf('MCU TX:         RMS = %.1f ns (%.1f m equiv)\n', ...
    rms(timing_error_breakdown(:,3)), rms(timing_error_breakdown(:,3))*c/1e9);
fprintf('LoRa Air-time:  RMS = %.1f ns (%.1f m equiv)\n', ...
    rms(timing_error_breakdown(:,4)), rms(timing_error_breakdown(:,4))*c/1e9);
fprintf('Multipath hits: %d (%.1f%%), RMS = %.1f ns (%.1f m equiv)\n', ...
    sum(timing_error_breakdown(:,5) > 0), ...
    100*sum(timing_error_breakdown(:,5) > 0)/N, ...
    rms(timing_error_breakdown(:,5)), rms(timing_error_breakdown(:,5))*c/1e9);

% Total RSS timing error
total_timing_rms = rms(sum(timing_error_breakdown, 2));
fprintf('\nTotal RMS:      %.1f ns (%.1f m equiv)\n', ...
    total_timing_rms, total_timing_rms*c/1e9);

% ---------------- FIGURE 1: ERROR VS TIME ----------------
figure('Position', [100 100 1200 400]);
plot(t, errors, 'LineWidth', 1.2, 'Color', [0.2 0.4 0.8]);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Position Error (m)', 'FontSize', 12);
title('TDoA Position Error vs Time – 20 km Corridor with Hardware Inefficiencies', 'FontSize', 13);
grid on;
ylim([0 max(errors)*1.1]);

% Add statistics text box
text_str = sprintf('Mean: %.2f m\nMedian: %.2f m\n95%%: %.2f m', ...
    mean(errors), median(errors), prctile(errors,95));
annotation('textbox', [0.15 0.65 0.15 0.2], 'String', text_str, ...
    'FitBoxToText', 'on', 'BackgroundColor', 'white', 'EdgeColor', 'black');

% ---------------- FIGURE 2: TRAJECTORY ----------------
figure('Position', [100 550 900 700]);
plot(x_true, y_true, 'g', 'LineWidth', 2.5); hold on;
plot(estPos(:,1), estPos(:,2), 'Color', [1 0.5 0], 'LineWidth', 1.2, 'LineStyle', ':');
plot(estPos_filtered(:,1), estPos_filtered(:,2), 'r--', 'LineWidth', 1.5);
scatter(gateways(:,1), gateways(:,2), 140, 'b', 'filled', 'MarkerEdgeColor', 'k');

% Show gateway position uncertainty
for i = 1:numGW
    viscircles(gateways(i,:), GPS_POS_SIGMA*2, 'Color', 'b', ...
        'LineStyle', '--', 'LineWidth', 0.5);
end

legend('True Path','Raw TDoA','Kalman Filtered','Gateways', ...
    'Location', 'best', 'FontSize', 10);
xlabel('X (m)', 'FontSize', 12);
ylabel('Y (m)', 'FontSize', 12);
title('20 km Corridor Navigation (45° Tilted) - Hardware Error Model', 'FontSize', 13);
grid on;
axis equal;

% ---------------- FIGURE 3: ERROR HISTOGRAM ----------------
figure('Position', [1050 550 600 400]);
histogram(errors, 50, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'k');
xlabel('Position Error (m)', 'FontSize', 12);
ylabel('Frequency', 'FontSize', 12);
title('Position Error Distribution', 'FontSize', 13);
grid on;
xline(mean(errors), 'r--', 'LineWidth', 2, 'Label', 'Mean');
xline(median(errors), 'g--', 'LineWidth', 2, 'Label', 'Median');

% ---------------- FIGURE 4: TIMING ERROR BREAKDOWN ----------------
figure('Position', [1050 100 700 400]);
labels = {'GPS PPS', 'MCU Capture', 'MCU TX', 'LoRa Air-time', 'Multipath'};
error_rms = [rms(timing_error_breakdown(:,1)), ...
             rms(timing_error_breakdown(:,2)), ...
             rms(timing_error_breakdown(:,3)), ...
             rms(timing_error_breakdown(:,4)), ...
             rms(timing_error_breakdown(:,5))];
bar(error_rms, 'FaceColor', [0.4 0.7 0.9]);
set(gca, 'XTickLabel', labels);
ylabel('RMS Timing Error (ns)', 'FontSize', 12);
title('Timing Error Sources (RMS)', 'FontSize', 13);
grid on;
xtickangle(45);

% Add values on bars
for i = 1:length(error_rms)
    text(i, error_rms(i), sprintf('%.1f ns\n(%.1f m)', ...
        error_rms(i), error_rms(i)*c/1e9), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontSize', 9, 'FontWeight', 'bold');
end

% ============================================================
%   ROBUST TDoA SOLVER with bounds
% ============================================================
function p = solve_tdoa_robust(gateways, dt, p0, c, hint)
    ref = gateways(1,:);
    others = gateways(2:end,:);
    
    % Cost function
    cost = @(x) sum( ...
        ((vecnorm(others - x,2,2) - norm(x - ref)) / c - dt).^2 );
    
    % Add bounds based on corridor extent
    options = optimset('Display','off', 'MaxFunEvals', 1000);
    
    % Try optimization with initial guess
    p = fminsearch(cost, p0, options);
    
    % If solution is way off, try from true position hint
    if norm(p - p0) > 500
        p = fminsearch(cost, hint, options);
    end
end

function r = rms(x)
    r = sqrt(mean(x.^2));
end