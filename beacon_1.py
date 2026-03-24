import numpy as np
import matplotlib.pyplot as plt

def get_user_input():
    """Get simulation parameters from user"""
    print("=" * 50)
    print("TDoA BEACON OPTIMIZATION SIMULATOR")
    print("=" * 50)
    
    while True:
        try:
            corridor_length = float(input("\nEnter corridor length (km): "))
            if corridor_length > 0:
                break
            print("Please enter a positive value")
        except ValueError:
            print("Please enter a valid number")
    
    corridor_length_m = corridor_length * 1000
    
    print(f"\nDefault parameters: Angle: 45°, Connectivity: 4 km")
    print("The simulator will find optimal beacon counts for your corridor.\n")
    
    return {
        'corridor_length_m': corridor_length_m,
        'connectivity_radius': 4000,
        'angle': 45
    }

def place_gateways(corridor_length_m, num_gateways, angle):
    """Place gateways along the corridor with alternating offsets"""
    if num_gateways == 1:
        x_positions = np.array([corridor_length_m / 2])
    else:
        x_positions = np.linspace(0, corridor_length_m, num_gateways)
    
    y_offsets = np.array([500 if i % 2 == 0 else -500 for i in range(num_gateways)])
    gateways_base = np.column_stack((x_positions, y_offsets))
    
    if angle != 0:
        theta = np.radians(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        gateways = (R @ gateways_base.T).T
    else:
        gateways = gateways_base
    
    return gateways

def generate_drone_path(corridor_length_m, angle, sample_interval=50):
    """Generate sample points along the corridor with sinusoidal wiggle"""
    # Sample every 'sample_interval' meters
    N = int(corridor_length_m / sample_interval) + 1
    x_base = np.linspace(0, corridor_length_m, N)
    
    # Add sinusoidal wiggle
    amplitude = min(200, corridor_length_m * 0.01)
    wavelength = 6000
    y_base = amplitude * np.sin(2 * np.pi * x_base / wavelength)
    path_base = np.column_stack((x_base, y_base))
    
    if angle != 0:
        theta = np.radians(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        path_rot = (R @ path_base.T).T
        x_true, y_true = path_rot[:, 0], path_rot[:, 1]
    else:
        x_true, y_true = path_base[:, 0], path_base[:, 1]
    
    return x_true, y_true, N

def calculate_connectivity_vectorized(drone_positions, gateways, connectivity_radius):
    """Calculate connectivity for all drone positions at once (vectorized)"""
    # drone_positions: (N, 2), gateways: (G, 2)
    # Calculate all distances at once: (N, G)
    diff = drone_positions[:, np.newaxis, :] - gateways[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    connected_mask = distances <= connectivity_radius
    num_connected = np.sum(connected_mask, axis=1)
    
    return connected_mask, num_connected

def run_single_simulation(num_gateways, params, verbose=False):
    """Run a single simulation with specified gateway count"""
    corridor_length_m = params['corridor_length_m']
    connectivity_radius = params['connectivity_radius']
    angle = params['angle']
    
    gateways = place_gateways(corridor_length_m, num_gateways, angle)
    x_true, y_true, N = generate_drone_path(corridor_length_m, angle)
    
    # Vectorized connectivity calculation
    drone_positions = np.column_stack((x_true, y_true))
    connected_mask, num_connected_array = calculate_connectivity_vectorized(
        drone_positions, gateways, connectivity_radius
    )
    
    avg_connected = np.mean(num_connected_array)
    
    if verbose:
        print(f"  {num_gateways} gateways: Avg connected = {avg_connected:.2f}")
    
    return {
        'num_gateways': num_gateways,
        'avg_connected': avg_connected,
        'min_connected': np.min(num_connected_array),
        'max_connected': np.max(num_connected_array),
        'connectivity': num_connected_array,
        'gateways': gateways,
        'x_true': x_true,
        'y_true': y_true
    }

def find_optimal_configurations(params):
    """Find recommended and minimum beacon configurations"""
    corridor_length_m = params['corridor_length_m']
    corridor_length_km = corridor_length_m / 1000
    
    # Start with high density: 1 gateway per km
    start_gateways = max(4, int(corridor_length_km))
    
    print(f"\nOptimizing beacon count for {corridor_length_km:.1f} km corridor...")
    print(f"Starting with {start_gateways} gateways, reducing to find optimal configurations.\n")
    
    results = []
    recommended_config = None
    minimum_config = None
    
    # Test decreasing gateway counts
    for num_gateways in range(start_gateways, 3, -1):
        result = run_single_simulation(num_gateways, params, verbose=True)
        results.append(result)
        
        # Find recommended configuration (avg connected ≈ 4)
        if recommended_config is None and result['avg_connected'] <= 4.0:
            recommended_config = result
            print(f"  → RECOMMENDED: {num_gateways} gateways (avg {result['avg_connected']:.2f} connected)")
        
        # Find minimum configuration (avg connected ≈ 3)
        if minimum_config is None and result['avg_connected'] <= 3.0:
            minimum_config = result
            print(f"  → MINIMUM: {num_gateways} gateways (avg {result['avg_connected']:.2f} connected)")
            break
    
    # If we didn't find configurations, use closest
    if recommended_config is None:
        recommended_config = min(results, key=lambda x: abs(x['avg_connected'] - 4.0))
        print(f"\n  Note: Closest to recommended is {recommended_config['num_gateways']} gateways")
    
    if minimum_config is None:
        minimum_config = min(results, key=lambda x: abs(x['avg_connected'] - 3.0))
        print(f"  Note: Closest to minimum is {minimum_config['num_gateways']} gateways")
    
    return results, recommended_config, minimum_config

def plot_optimization_results(results, recommended_config, minimum_config, params):
    """Plot optimization results and detailed analysis"""
    corridor_length_km = params['corridor_length_m'] / 1000
    
    # Extract data for plotting
    gateway_counts = [r['num_gateways'] for r in results]
    avg_connected = [r['avg_connected'] for r in results]
    min_connected = [r['min_connected'] for r in results]
    max_connected = [r['max_connected'] for r in results]
    
    fig = plt.figure(figsize=(18, 10))
    
    # Plot 1: Average Connected Gateways vs Gateway Count
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(gateway_counts, avg_connected, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.axhline(y=4, color='green', linestyle='--', linewidth=2, label='Recommended (4 avg)')
    ax1.axhline(y=3, color='orange', linestyle='--', linewidth=2, label='Minimum (3 avg)')
    if recommended_config:
        ax1.plot(recommended_config['num_gateways'], recommended_config['avg_connected'], 
                'gs', markersize=15, label=f"Recommended: {recommended_config['num_gateways']} beacons")
    if minimum_config:
        ax1.plot(minimum_config['num_gateways'], minimum_config['avg_connected'], 
                'rs', markersize=15, label=f"Minimum: {minimum_config['num_gateways']} beacons")
    ax1.set_xlabel('Number of Gateways')
    ax1.set_ylabel('Average Connected Gateways')
    ax1.set_title('Connectivity vs Gateway Count')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.invert_xaxis()
    
    # Plot 2: Connectivity Range
    ax2 = plt.subplot(2, 3, 2)
    ax2.fill_between(gateway_counts, min_connected, max_connected, alpha=0.3, color='blue')
    ax2.plot(gateway_counts, avg_connected, 'o-', linewidth=2, markersize=6, color='blue', label='Average')
    ax2.plot(gateway_counts, min_connected, 's--', linewidth=1, markersize=4, color='red', label='Minimum')
    ax2.plot(gateway_counts, max_connected, '^--', linewidth=1, markersize=4, color='green', label='Maximum')
    ax2.set_xlabel('Number of Gateways')
    ax2.set_ylabel('Connected Gateways')
    ax2.set_title('Connectivity Range (Min/Avg/Max)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.invert_xaxis()
    
    # Plot 3: Recommended Configuration Trajectory
    if recommended_config:
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(recommended_config['x_true'], recommended_config['y_true'], 
                'g', linewidth=2, label='Drone Path')
        for i, (x, y) in enumerate(recommended_config['gateways']):
            circle = plt.Circle((x, y), params['connectivity_radius'], 
                               color='blue', alpha=0.1, linestyle='--', linewidth=1)
            ax3.add_patch(circle)
            ax3.scatter(x, y, s=100, c='b', marker='^', zorder=5)
            ax3.text(x, y, f'G{i+1}', fontsize=8, ha='center', va='bottom', fontweight='bold')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_title(f'RECOMMENDED: {recommended_config["num_gateways"]} Beacons')
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        ax3.legend()
    
    # Plot 4: Recommended Connectivity over time
    if recommended_config:
        ax4 = plt.subplot(2, 3, 4)
        sample_points = np.arange(len(recommended_config['connectivity']))
        ax4.plot(sample_points, recommended_config['connectivity'], linewidth=2, color='green')
        ax4.fill_between(sample_points, 0, recommended_config['connectivity'], alpha=0.3, color='green')
        ax4.axhline(y=4, color='red', linestyle='--', linewidth=1, label='Target avg (4)')
        ax4.set_xlabel('Sample Point')
        ax4.set_ylabel('Connected Gateways')
        ax4.set_title(f'Connectivity - Recommended (Avg: {recommended_config["avg_connected"]:.2f})')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    # Plot 5: Minimum Configuration Trajectory
    if minimum_config:
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(minimum_config['x_true'], minimum_config['y_true'], 
                'g', linewidth=2, label='Drone Path')
        for i, (x, y) in enumerate(minimum_config['gateways']):
            circle = plt.Circle((x, y), params['connectivity_radius'], 
                               color='blue', alpha=0.1, linestyle='--', linewidth=1)
            ax5.add_patch(circle)
            ax5.scatter(x, y, s=100, c='orange', marker='^', zorder=5)
            ax5.text(x, y, f'G{i+1}', fontsize=8, ha='center', va='bottom', fontweight='bold')
        ax5.set_xlabel('X (m)')
        ax5.set_ylabel('Y (m)')
        ax5.set_title(f'MINIMUM: {minimum_config["num_gateways"]} Beacons')
        ax5.grid(True, alpha=0.3)
        ax5.axis('equal')
        ax5.legend()
    
    # Plot 6: Minimum Connectivity over time
    if minimum_config:
        ax6 = plt.subplot(2, 3, 6)
        sample_points = np.arange(len(minimum_config['connectivity']))
        ax6.plot(sample_points, minimum_config['connectivity'], linewidth=2, color='orange')
        ax6.fill_between(sample_points, 0, minimum_config['connectivity'], alpha=0.3, color='orange')
        ax6.axhline(y=3, color='red', linestyle='--', linewidth=1, label='Target avg (3)')
        ax6.set_xlabel('Sample Point')
        ax6.set_ylabel('Connected Gateways')
        ax6.set_title(f'Connectivity - Minimum (Avg: {minimum_config["avg_connected"]:.2f})')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    
    plt.suptitle(f'Beacon Optimization for {corridor_length_km:.1f} km Corridor', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    
    # Summary Report
    print("\n" + "=" * 70)
    print("BEACON OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"Corridor Length: {corridor_length_km:.1f} km")
    print(f"Connectivity Radius: {params['connectivity_radius']/1000:.1f} km")
    print()
    if recommended_config:
        print(f"✓ RECOMMENDED CONFIGURATION: {recommended_config['num_gateways']} beacons")
        print(f"  - Average connected: {recommended_config['avg_connected']:.2f}")
        print(f"  - Connectivity range: {recommended_config['min_connected']} - {recommended_config['max_connected']}")
        print(f"  - Spacing: ~{corridor_length_km/recommended_config['num_gateways']:.2f} km between beacons")
    print()
    if minimum_config:
        print(f"⚠ MINIMUM CONFIGURATION: {minimum_config['num_gateways']} beacons")
        print(f"  - Average connected: {minimum_config['avg_connected']:.2f}")
        print(f"  - Connectivity range: {minimum_config['min_connected']} - {minimum_config['max_connected']}")
        print(f"  - Spacing: ~{corridor_length_km/minimum_config['num_gateways']:.2f} km between beacons")
    print("=" * 70)

def main():
    """Main function to run the optimizer"""
    try:
        params = get_user_input()
        results, recommended_config, minimum_config = find_optimal_configurations(params)
        plot_optimization_results(results, recommended_config, minimum_config, params)
        
        save_choice = input("\nSave optimization results? (y/n): ").lower()
        if save_choice == 'y':
            filename = f"beacon_optimization_{params['corridor_length_m']/1000:.0f}km.csv"
            data = np.array([[r['num_gateways'], r['avg_connected'], 
                            r['min_connected'], r['max_connected']] for r in results])
            header = "Num_Gateways,Avg_Connected,Min_Connected,Max_Connected"
            np.savetxt(filename, data, delimiter=',', header=header, fmt='%.2f', comments='')
            print(f"Results saved to '{filename}'")
            
    except KeyboardInterrupt:
        print("\nOptimization cancelled.")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()