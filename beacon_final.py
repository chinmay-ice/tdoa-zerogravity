import numpy as np
from pyproj import Proj, Transformer
from geopy.distance import geodesic
import json
import folium
from folium import plugins
import webbrowser
import os
import matplotlib.pyplot as plt

# ============================================================
#   INTEGRATED GPS-BASED TDoA BEACON SYSTEM
# ============================================================

class GPSCoordinateConverter:
    """Handles conversion between GPS and local Cartesian coordinates"""
    
    def __init__(self, origin_lat, origin_lon):
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        
        # Determine UTM zone and hemisphere
        utm_zone = int((origin_lon + 180) / 6) + 1
        hemisphere = 'north' if origin_lat >= 0 else 'south'
        
        # Create projections
        self.proj_utm = Proj(proj='utm', zone=utm_zone, ellps='WGS84', 
                             south=(hemisphere == 'south'))
        self.proj_wgs84 = Proj(proj='latlong', datum='WGS84')
        
        # Create transformers
        self.transformer_to_utm = Transformer.from_proj(self.proj_wgs84, self.proj_utm)
        self.transformer_to_gps = Transformer.from_proj(self.proj_utm, self.proj_wgs84)
        
        # Calculate origin in UTM
        self.origin_x, self.origin_y = self.transformer_to_utm.transform(origin_lon, origin_lat)
    
    def gps_to_local(self, lat, lon):
        """Convert GPS coordinates to local Cartesian (meters from origin)"""
        x_utm, y_utm = self.transformer_to_utm.transform(lon, lat)
        return x_utm - self.origin_x, y_utm - self.origin_y
    
    def local_to_gps(self, x_local, y_local):
        """Convert local Cartesian coordinates back to GPS"""
        x_utm = x_local + self.origin_x
        y_utm = y_local + self.origin_y
        lon, lat = self.transformer_to_gps.transform(x_utm, y_utm)
        return lat, lon
    
    def batch_local_to_gps(self, xy_array):
        """Convert array of local coordinates to GPS"""
        return np.array([self.local_to_gps(x, y) for x, y in xy_array])


def calculate_corridor_parameters(start_lat, start_lon, end_lat, end_lon):
    """Calculate corridor parameters from GPS coordinates"""
    converter = GPSCoordinateConverter(start_lat, start_lon)
    end_x, end_y = converter.gps_to_local(end_lat, end_lon)
    
    corridor_length_m = np.sqrt(end_x**2 + end_y**2)
    angle_degrees = np.degrees(np.arctan2(end_x, end_y))
    
    if angle_degrees < 0:
        angle_degrees += 360
    
    return corridor_length_m, angle_degrees, converter, (end_x, end_y)


def place_beacons_gps(corridor_length_m, num_beacons, offset_m, end_local):
    """Place beacons along corridor with alternating offsets"""
    num_beacons = max(4, num_beacons)
    
    end_x, end_y = end_local
    corridor_vector = np.array([end_x, end_y])
    corridor_unit = corridor_vector / corridor_length_m
    perp_unit = np.array([-corridor_unit[1], corridor_unit[0]])
    
    t_values = np.linspace(0, 1, num_beacons) if num_beacons > 1 else np.array([0.5])
    
    beacons_local = []
    beacons_base = []
    
    for i, t in enumerate(t_values):
        pos_along = t * corridor_vector
        offset_sign = 1 if i % 2 == 0 else -1
        offset_vector = offset_sign * offset_m * perp_unit
        
        beacons_local.append(pos_along + offset_vector)
        beacons_base.append([t * corridor_length_m, offset_sign * offset_m])
    
    return np.array(beacons_local), np.array(beacons_base)


def get_gps_input():
    """Get GPS coordinates from user"""
    print("=" * 60)
    print("GPS-BASED TDoA BEACON PLACEMENT SYSTEM")
    print("=" * 60)
    print("\nEnter GPS coordinates (Latitude, Longitude)")
    print("Example: 26.9124, 75.7873")
    
    # Get start point
    while True:
        try:
            start_input = input("\nStart Point (lat, lon): ")
            start_lat, start_lon = map(float, start_input.split(','))
            if not (-90 <= start_lat <= 90 and -180 <= start_lon <= 180):
                print("Invalid coordinates. Lat: -90 to 90, Lon: -180 to 180")
                continue
            break
        except ValueError:
            print("Invalid format. Please enter: latitude, longitude")
    
    # Get end point
    while True:
        try:
            end_input = input("End Point (lat, lon): ")
            end_lat, end_lon = map(float, end_input.split(','))
            if not (-90 <= end_lat <= 90 and -180 <= end_lon <= 180):
                print("Invalid coordinates. Lat: -90 to 90, Lon: -180 to 180")
                continue
            break
        except ValueError:
            print("Invalid format. Please enter: latitude, longitude")
    
    distance_km = geodesic((start_lat, start_lon), (end_lat, end_lon)).kilometers
    print(f"\nCorridor distance: {distance_km:.2f} km")
    
    return {
        'start_lat': start_lat, 'start_lon': start_lon,
        'end_lat': end_lat, 'end_lon': end_lon,
        'distance_km': distance_km
    }


# ============================================================
#   BEACON OPTIMIZATION FUNCTIONS
# ============================================================

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
    N = int(corridor_length_m / sample_interval) + 1
    x_base = np.linspace(0, corridor_length_m, N)
    
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
    
    start_gateways = max(4, int(corridor_length_km))
    
    print(f"\nOptimizing beacon count for {corridor_length_km:.1f} km corridor...")
    print(f"Starting with {start_gateways} gateways, reducing to find optimal configurations.\n")
    
    results = []
    recommended_config = None
    minimum_config = None
    
    for num_gateways in range(start_gateways, 3, -1):
        result = run_single_simulation(num_gateways, params, verbose=True)
        results.append(result)
        
        if recommended_config is None and result['avg_connected'] <= 4.0:
            recommended_config = result
            print(f"  → RECOMMENDED: {num_gateways} gateways (avg {result['avg_connected']:.2f} connected)")
        
        if minimum_config is None and result['avg_connected'] <= 3.0:
            minimum_config = result
            print(f"  → MINIMUM: {num_gateways} gateways (avg {result['avg_connected']:.2f} connected)")
            break
    
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


# ============================================================
#   MAP CREATION AND EXPORT
# ============================================================

def create_interactive_map(start_gps, end_gps, beacons_gps, corridor_params):
    """Create interactive folium map"""
    corridor_length_m, angle_degrees, _, _ = corridor_params
    
    center_lat = (start_gps[0] + end_gps[0]) / 2
    center_lon = (start_gps[1] + end_gps[1]) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.PolyLine(
        locations=[[start_gps[0], start_gps[1]], [end_gps[0], end_gps[1]]],
        color='green', weight=4, opacity=0.8,
        popup=f'Corridor: {corridor_length_m/1000:.2f} km'
    ).add_to(m)
    
    folium.Marker(
        [start_gps[0], start_gps[1]],
        popup=f"START<br>Lat: {start_gps[0]:.6f}<br>Lon: {start_gps[1]:.6f}",
        icon=folium.Icon(color='blue', icon='play', prefix='fa')
    ).add_to(m)
    
    folium.Marker(
        [end_gps[0], end_gps[1]],
        popup=f"END<br>Lat: {end_gps[0]:.6f}<br>Lon: {end_gps[1]:.6f}",
        icon=folium.Icon(color='purple', icon='stop', prefix='fa')
    ).add_to(m)
    
    colors = ['red', 'blue', 'green', 'orange', 'darkred', 'darkblue']
    
    for i, gps in enumerate(beacons_gps):
        color = colors[i % len(colors)]
        
        popup_html = f"""
            <b>Beacon B{i+1}</b><br>
            Lat: {gps[0]:.8f}°<br>
            Lon: {gps[1]:.8f}°<br>
            Coverage: 4 km radius
        """
        
        folium.Marker(
            [gps[0], gps[1]],
            popup=popup_html,
            tooltip=f'Beacon B{i+1}',
            icon=folium.Icon(color=color, icon='broadcast-tower', prefix='fa')
        ).add_to(m)
        
        folium.Circle(
            [gps[0], gps[1]], radius=4000,
            color=color, fill=True, fillOpacity=0.1, weight=1
        ).add_to(m)
    
    plugins.MeasureControl(position='topright').add_to(m)
    plugins.Fullscreen(position='topleft').add_to(m)
    folium.LayerControl().add_to(m)
    
    map_file = 'beacon_map.html'
    m.save(map_file)
    print(f"\n✅ Map created: {map_file}")
    
    try:
        webbrowser.open('file://' + os.path.realpath(map_file))
        print("🌐 Opening in browser...")
    except:
        print(f"⚠️  Please open '{map_file}' manually")
    
    return map_file


def export_results(beacons_gps, params, corridor_params):
    """Export beacon coordinates"""
    export_choice = input("\nExport coordinates? (y/n): ").lower()
    
    if export_choice != 'y':
        return
    
    export_data = {
        'corridor': {
            'start': {'latitude': params['start_lat'], 'longitude': params['start_lon']},
            'end': {'latitude': params['end_lat'], 'longitude': params['end_lon']},
            'length_km': corridor_params[0] / 1000,
            'angle_degrees': corridor_params[1]
        },
        'beacons': [
            {
                'id': f'B{i+1}',
                'latitude': float(gps[0]),
                'longitude': float(gps[1])
            }
            for i, gps in enumerate(beacons_gps)
        ]
    }
    
    with open('beacons.csv', 'w') as f:
        f.write('Beacon_ID,Latitude,Longitude\n')
        for i, gps in enumerate(beacons_gps):
            f.write(f'B{i+1},{gps[0]:.8f},{gps[1]:.8f}\n')
    print("✅ Saved: beacons.csv")


# ============================================================
#   MAIN INTEGRATED WORKFLOW
# ============================================================

def main():
    """Main integrated function"""
    try:
        # Step 1: Get GPS coordinates
        gps_params = get_gps_input()
        
        # Step 2: Calculate corridor parameters
        corridor_length_m, angle_degrees, converter, end_local = calculate_corridor_parameters(
            gps_params['start_lat'], gps_params['start_lon'],
            gps_params['end_lat'], gps_params['end_lon']
        )
        
        print("\n" + "=" * 60)
        print(f"Corridor calculated: {corridor_length_m/1000:.2f} km")
        print("=" * 60)
        
        # Step 3: Run optimization to find recommended beacon count
        print("\n🔍 Running optimization to find recommended beacon configurations...")
        opt_params = {
            'corridor_length_m': corridor_length_m,
            'connectivity_radius': 4000,
            'angle': 45
        }
        
        results, recommended_config, minimum_config = find_optimal_configurations(opt_params)
        
        # Step 4: Ask user for beacon count (with recommendation)
        print("\n" + "=" * 60)
        if recommended_config:
            print(f"💡 Recommended: {recommended_config['num_gateways']} beacons")
        if minimum_config:
            print(f"⚠️  Minimum viable: {minimum_config['num_gateways']} beacons")
        print("=" * 60)
        
        while True:
            try:
                num_beacons = int(input("\nEnter number of beacons to use (minimum 4): "))
                if num_beacons < 4:
                    print("Minimum 4 beacons required")
                    continue
                break
            except ValueError:
                print("Please enter a valid number")
        
        # Step 5: Get lateral offset
        while True:
            try:
                offset_m = float(input("\nLateral offset (meters, default 500): ") or "500")
                if offset_m <= 0:
                    print("Please enter positive value")
                    continue
                break
            except ValueError:
                print("Invalid input")
        
        # Step 6: Place beacons with user's choice
        print("\n" + "=" * 60)
        print("CALCULATING BEACON PLACEMENT...")
        print("=" * 60)
        
        beacons_local, beacons_base = place_beacons_gps(
            corridor_length_m, num_beacons, offset_m, end_local
        )
        
        beacons_gps = converter.batch_local_to_gps(beacons_local)
        
        print(f"\n✅ Placement complete!")
        print(f"  Corridor: {corridor_length_m/1000:.2f} km @ {angle_degrees:.2f}°")
        print(f"  Beacons: {len(beacons_gps)}")
        print(f"  Spacing: {corridor_length_m/(len(beacons_gps)-1)/1000:.2f} km")
        
        print("\nBEACON COORDINATES:")
        for i, gps in enumerate(beacons_gps):
            print(f"  B{i+1}: {gps[0]:.6f}°, {gps[1]:.6f}°")
        
        # Step 7: Show optimization visualization
        show_opt = input("\nView optimization analysis plots? (y/n): ").lower()
        if show_opt == 'y':
            plot_optimization_results(results, recommended_config, minimum_config, opt_params)
        
        # Step 8: Create map
        corridor_params = (corridor_length_m, angle_degrees, converter, end_local)
        create_interactive_map(
            (gps_params['start_lat'], gps_params['start_lon']),
            (gps_params['end_lat'], gps_params['end_lon']),
            beacons_gps, corridor_params
        )
        
        # Step 9: Export
        export_results(beacons_gps, gps_params, corridor_params)
        
        print("\n" + "=" * 60)
        print("✅ COMPLETE! Thank you for using the TDoA Beacon System")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()