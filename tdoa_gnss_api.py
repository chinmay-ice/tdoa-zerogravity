import numpy as np
from pyproj import Proj, Transformer
from geopy.distance import geodesic
import json
import folium
from folium import plugins
import webbrowser
import os

# ============================================================
#   GPS-BASED TDoA BEACON PLACEMENT SYSTEM
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
    
    # Get number of beacons
    while True:
        try:
            num_beacons = int(input("\nNumber of beacons (minimum 4): "))
            if num_beacons < 4:
                print("Minimum 4 beacons required")
                continue
            break
        except ValueError:
            print("Please enter a valid number")
    
    # Lateral offset
    while True:
        try:
            offset_m = float(input("\nLateral offset (meters, default 500): ") or "500")
            if offset_m <= 0:
                print("Please enter positive value")
                continue
            break
        except ValueError:
            print("Invalid input")
    
    return {
        'start_lat': start_lat, 'start_lon': start_lon,
        'end_lat': end_lat, 'end_lon': end_lon,
        'num_beacons': num_beacons, 'offset_m': offset_m
    }


def create_interactive_map(start_gps, end_gps, beacons_gps, corridor_params):
    """Create interactive folium map"""
    corridor_length_m, angle_degrees, _, _ = corridor_params
    
    # Calculate map center
    center_lat = (start_gps[0] + end_gps[0]) / 2
    center_lon = (start_gps[1] + end_gps[1]) / 2
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Add satellite layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Draw corridor
    folium.PolyLine(
        locations=[[start_gps[0], start_gps[1]], [end_gps[0], end_gps[1]]],
        color='green', weight=4, opacity=0.8,
        popup=f'Corridor: {corridor_length_m/1000:.2f} km'
    ).add_to(m)
    
    # Add start/end markers
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
    
    # Add beacon markers
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
        
        # Coverage circle
        folium.Circle(
            [gps[0], gps[1]], radius=4000,
            color=color, fill=True, fillOpacity=0.1, weight=1
        ).add_to(m)
    
    # Add tools
    plugins.MeasureControl(position='topright').add_to(m)
    plugins.Fullscreen(position='topleft').add_to(m)
    folium.LayerControl().add_to(m)
    
    # Save and open
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
    
    # JSON export
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
    
    with open('beacons.json', 'w') as f:
        json.dump(export_data, f, indent=2)
    print("✅ Saved: beacons.json")
    
    # CSV export
    with open('beacons.csv', 'w') as f:
        f.write('Beacon_ID,Latitude,Longitude\n')
        for i, gps in enumerate(beacons_gps):
            f.write(f'B{i+1},{gps[0]:.8f},{gps[1]:.8f}\n')
    print("✅ Saved: beacons.csv")


def main():
    """Main function"""
    try:
        # Get input
        params = get_gps_input()
        
        # Calculate corridor
        corridor_length_m, angle_degrees, converter, end_local = calculate_corridor_parameters(
            params['start_lat'], params['start_lon'],
            params['end_lat'], params['end_lon']
        )
        corridor_params = (corridor_length_m, angle_degrees, converter, end_local)
        
        print("\n" + "=" * 60)
        print("CALCULATING BEACON PLACEMENT...")
        print("=" * 60)
        
        # Place beacons
        beacons_local, beacons_base = place_beacons_gps(
            corridor_length_m, params['num_beacons'],
            params['offset_m'], end_local
        )
        
        # Convert to GPS
        beacons_gps = converter.batch_local_to_gps(beacons_local)
        
        print(f"\n✅ Placement complete!")
        print(f"  Corridor: {corridor_length_m/1000:.2f} km @ {angle_degrees:.2f}°")
        print(f"  Beacons: {len(beacons_gps)}")
        print(f"  Spacing: {corridor_length_m/(len(beacons_gps)-1)/1000:.2f} km")
        
        # Print coordinates
        print("\nBEACON COORDINATES:")
        for i, gps in enumerate(beacons_gps):
            print(f"  B{i+1}: {gps[0]:.6f}°, {gps[1]:.6f}°")
        
        # Create map
        create_interactive_map(
            (params['start_lat'], params['start_lon']),
            (params['end_lat'], params['end_lon']),
            beacons_gps, corridor_params
        )
        
        # Export
        export_results(beacons_gps, params, corridor_params)
        
        print("\n" + "=" * 60)
        print("Done! Thank you for using the TDoA Beacon System")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()