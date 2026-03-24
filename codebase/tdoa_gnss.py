import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pyproj import Proj, transform, Transformer
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
        """
        Initialize converter with origin point
        origin_lat, origin_lon: GPS coordinates of the origin (start point)
        """
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        
        # Use UTM projection centered on origin
        # Determine UTM zone from longitude
        utm_zone = int((origin_lon + 180) / 6) + 1
        hemisphere = 'north' if origin_lat >= 0 else 'south'
        
        # Create projection
        self.proj_utm = Proj(proj='utm', zone=utm_zone, ellps='WGS84', 
                             south=(hemisphere == 'south'))
        self.proj_wgs84 = Proj(proj='latlong', datum='WGS84')
        
        # Create transformer for bidirectional conversion
        self.transformer_to_utm = Transformer.from_proj(self.proj_wgs84, self.proj_utm)
        self.transformer_to_gps = Transformer.from_proj(self.proj_utm, self.proj_wgs84)
        
        # Calculate origin in UTM
        self.origin_x, self.origin_y = self.transformer_to_utm.transform(origin_lon, origin_lat)
    
    def gps_to_local(self, lat, lon):
        """Convert GPS coordinates to local Cartesian (meters from origin)"""
        x_utm, y_utm = self.transformer_to_utm.transform(lon, lat)
        x_local = x_utm - self.origin_x
        y_local = y_utm - self.origin_y
        return x_local, y_local
    
    def local_to_gps(self, x_local, y_local):
        """Convert local Cartesian coordinates back to GPS"""
        x_utm = x_local + self.origin_x
        y_utm = y_local + self.origin_y
        lon, lat = self.transformer_to_gps.transform(x_utm, y_utm)
        return lat, lon
    
    def batch_local_to_gps(self, xy_array):
        """Convert array of local coordinates to GPS"""
        gps_coords = []
        for x, y in xy_array:
            lat, lon = self.local_to_gps(x, y)
            gps_coords.append([lat, lon])
        return np.array(gps_coords)


def calculate_corridor_parameters(start_lat, start_lon, end_lat, end_lon):
    """
    Calculate corridor parameters from GPS coordinates
    Returns: corridor_length_m, angle_degrees, converter object
    """
    # Initialize converter with start point as origin
    converter = GPSCoordinateConverter(start_lat, start_lon)
    
    # Convert end point to local coordinates
    end_x, end_y = converter.gps_to_local(end_lat, end_lon)
    
    # Calculate corridor length
    corridor_length_m = np.sqrt(end_x**2 + end_y**2)
    
    # Calculate angle (from North, clockwise)
    # In local coordinates: +X is East, +Y is North
    angle_rad = np.arctan2(end_x, end_y)  # Note: arctan2(x, y) for angle from North
    angle_degrees = np.degrees(angle_rad)
    
    # Normalize angle to 0-360
    if angle_degrees < 0:
        angle_degrees += 360
    
    return corridor_length_m, angle_degrees, converter, (end_x, end_y)


def place_beacons_gps(corridor_length_m, angle_degrees, num_beacons, offset_m=500, end_local=(0, 0)):
    """
    Place beacons along corridor with alternating offsets
    
    Args:
        corridor_length_m: Length of corridor in meters
        angle_degrees: Corridor angle from North
        num_beacons: Number of beacons
        offset_m: Lateral offset from centerline
        end_local: (end_x, end_y) coordinates in local system
    
    Returns:
        beacons_local: Beacon positions in local coordinates
        beacons_base: Beacon positions in base coordinates
        num_beacons: Final number of beacons
    """
    # Ensure minimum 4 beacons
    num_beacons = max(4, num_beacons)
    
    # Get corridor direction vector (from origin to end point)
    end_x, end_y = end_local
    corridor_vector = np.array([end_x, end_y])
    corridor_unit = corridor_vector / corridor_length_m
    
    # Calculate perpendicular vector (rotate 90 degrees)
    # Perpendicular to (x, y) is (-y, x)
    perp_unit = np.array([-corridor_unit[1], corridor_unit[0]])
    
    # Create positions along the corridor
    if num_beacons == 1:
        t_values = np.array([0.5])
    else:
        t_values = np.linspace(0, 1, num_beacons)
    
    beacons_local = []
    beacons_base = []
    
    for i, t in enumerate(t_values):
        # Position along corridor
        pos_along = t * corridor_vector
        
        # Alternating perpendicular offset
        offset_sign = 1 if i % 2 == 0 else -1
        offset_vector = offset_sign * offset_m * perp_unit
        
        # Final beacon position in local coordinates
        beacon_pos = pos_along + offset_vector
        beacons_local.append(beacon_pos)
        
        # Base position (for display/reference - corridor along x-axis)
        base_x = t * corridor_length_m
        base_y = offset_sign * offset_m
        beacons_base.append([base_x, base_y])
    
    beacons_local = np.array(beacons_local)
    beacons_base = np.array(beacons_base)
    
    return beacons_local, beacons_base, num_beacons

def get_gps_input():
    """Get GPS coordinates from user"""
    print("=" * 60)
    print("GPS-BASED TDoA BEACON PLACEMENT SYSTEM")
    print("=" * 60)
    print("\nEnter GPS coordinates for the corridor")
    print("Format: Latitude, Longitude (decimal degrees)")
    print("Example: 26.9124, 75.7873 (Jaipur, India)")
    
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
    
    # Calculate distance using geodesic
    distance_km = geodesic((start_lat, start_lon), (end_lat, end_lon)).kilometers
    print(f"\nCorridor distance: {distance_km:.2f} km")
    
    # Get number of beacons
    while True:
        try:
            num_beacons = int(input("\nEnter number of beacons (minimum 4): "))
            if num_beacons < 4:
                print("Minimum 4 beacons required for TDoA")
                continue
            break
        except ValueError:
            print("Please enter a valid number")
    
    # Lateral offset
    while True:
        try:
            offset_m = float(input("\nLateral offset from corridor centerline (meters, default 500): ") or "500")
            if offset_m <= 0:
                print("Please enter positive value")
                continue
            break
        except ValueError:
            print("Invalid input")
    
    return {
        'start_lat': start_lat,
        'start_lon': start_lon,
        'end_lat': end_lat,
        'end_lon': end_lon,
        'num_beacons': num_beacons,
        'offset_m': offset_m
    }


def create_interactive_map(start_gps, end_gps, beacons_gps, beacons_local, 
                          beacons_base, corridor_params):
    """
    Create an interactive folium map with beacons and corridor
    Opens automatically in web browser
    """
    corridor_length_m, angle_degrees, _, _ = corridor_params
    
    # Calculate map center
    center_lat = (start_gps[0] + end_gps[0]) / 2
    center_lon = (start_gps[1] + end_gps[1]) / 2
    
    # Create map with default OpenStreetMap
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add multiple tile layers for user choice with proper attribution
    # CartoDB Positron (Light theme)
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        name='CartoDB Positron',
        overlay=False,
        control=True
    ).add_to(m)
    
    # CartoDB Dark Matter (Dark theme)
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        name='CartoDB Dark',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Esri Satellite imagery
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
        name='Esri Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Esri World Street Map
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Tiles &copy; Esri',
        name='Esri Street Map',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Draw corridor line (thick green line for visibility)
    corridor_line = folium.PolyLine(
        locations=[[start_gps[0], start_gps[1]], [end_gps[0], end_gps[1]]],
        color='green',
        weight=6,
        opacity=0.9,
        popup=f'Corridor: {corridor_length_m/1000:.2f} km at {angle_degrees:.1f}°',
        tooltip='Flight Corridor'
    )
    corridor_line.add_to(m)
    
    # Add a thinner line on top for better visibility
    folium.PolyLine(
        locations=[[start_gps[0], start_gps[1]], [end_gps[0], end_gps[1]]],
        color='darkgreen',
        weight=2,
        opacity=1.0,
        dash_array='10, 5'
    ).add_to(m)
    
    # Add start marker
    folium.Marker(
        location=[start_gps[0], start_gps[1]],
        popup=folium.Popup(f"""
            <b>START POINT</b><br>
            Lat: {start_gps[0]:.6f}°<br>
            Lon: {start_gps[1]:.6f}°<br>
            <hr>
            <i>Corridor Origin</i>
        """, max_width=250),
        tooltip='Start Point',
        icon=folium.Icon(color='blue', icon='play', prefix='fa')
    ).add_to(m)
    
    # Add end marker
    folium.Marker(
        location=[end_gps[0], end_gps[1]],
        popup=folium.Popup(f"""
            <b>END POINT</b><br>
            Lat: {end_gps[0]:.6f}°<br>
            Lon: {end_gps[1]:.6f}°<br>
            <hr>
            <i>Distance: {corridor_length_m/1000:.2f} km</i>
        """, max_width=250),
        tooltip='End Point',
        icon=folium.Icon(color='purple', icon='stop', prefix='fa')
    ).add_to(m)
    
    # Add beacon markers with detailed popups
    # Color palette for beacons (cycle through different colors)
    beacon_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                     'lightred', 'darkblue', 'darkgreen', 'cadetblue', 
                     'darkpurple', 'pink', 'lightblue', 'lightgreen']
    
    # Hex colors for circles (matching the marker colors)
    circle_colors = ['#FF0000', '#0000FF', '#008000', '#800080', '#FFA500', '#8B0000',
                    '#FFB6C1', '#00008B', '#006400', '#5F9EA0', '#4B0082', '#FFC0CB',
                    '#ADD8E6', '#90EE90']
    
    # Calculate corridor direction for perpendicular lines
    corridor_lat_diff = end_gps[0] - start_gps[0]
    corridor_lon_diff = end_gps[1] - start_gps[1]
    
    for i, (gps, local, base) in enumerate(zip(beacons_gps, beacons_local, beacons_base)):
        # Calculate distance from start
        dist_from_start = geodesic(start_gps, (gps[0], gps[1])).meters
        
        # Calculate the point on corridor centerline closest to this beacon
        t = base[0] / corridor_length_m  # Parameter along corridor (0 to 1)
        centerline_lat = start_gps[0] + t * corridor_lat_diff
        centerline_lon = start_gps[1] + t * corridor_lon_diff
        
        # Select color (cycle through colors)
        color_idx = i % len(beacon_colors)
        marker_color = beacon_colors[color_idx]
        circle_color = circle_colors[color_idx]
        
        # Draw perpendicular line from corridor to beacon (shows offset)
        folium.PolyLine(
            locations=[[centerline_lat, centerline_lon], [gps[0], gps[1]]],
            color=circle_color,
            weight=2,
            opacity=0.6,
            dash_array='5, 5',
            popup=f'B{i+1} offset: {base[1]:+.0f}m'
        ).add_to(m)
        
        # Mark the centerline point
        folium.CircleMarker(
            location=[centerline_lat, centerline_lon],
            radius=4,
            color=circle_color,
            fill=True,
            fillColor='white',
            fillOpacity=1,
            weight=2
        ).add_to(m)
        
        # Create detailed popup
        popup_html = f"""
            <div style="font-family: Arial; width: 300px;">
                <h4 style="margin: 0; color: {circle_color};">🎯 Beacon B{i+1}</h4>
                <hr style="margin: 5px 0;">
                
                <b>📍 GPS Coordinates:</b><br>
                Latitude: {gps[0]:.8f}°<br>
                Longitude: {gps[1]:.8f}°<br><br>
                
                <b>📏 Local Coordinates (Rotated):</b><br>
                X: {local[0]:.1f} m<br>
                Y: {local[1]:.1f} m<br><br>
                
                <b>📐 Base Coordinates:</b><br>
                Along corridor: {base[0]:.1f} m ({base[0]/1000:.2f} km)<br>
                Lateral offset: {base[1]:+.1f} m<br><br>
                
                <b>📊 Position:</b><br>
                Distance from start: {dist_from_start/1000:.2f} km<br>
                Corridor angle: {angle_degrees:.1f}° from North<br>
                
                <b>📡 Coverage:</b><br>
                Radius: 4.0 km<br>
                
                <hr style="margin: 5px 0;">
                <i style="font-size: 11px; color: #666;">
                Click coordinates to copy • Drag map to pan
                </i>
            </div>
        """
        
        folium.Marker(
            location=[gps[0], gps[1]],
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=f'Beacon B{i+1} (4km coverage)',
            icon=folium.Icon(
                color=marker_color,
                icon='broadcast-tower',
                prefix='fa'
            )
        ).add_to(m)
        
        # Add circle around beacon to show coverage area (4km radius)
        folium.Circle(
            location=[gps[0], gps[1]],
            radius=4000,  # 4 km radius
            color=circle_color,
            fill=True,
            fillOpacity=0.15,
            opacity=0.4,
            weight=2,
            popup=f'B{i+1} Coverage Area (4km radius)',
            tooltip=f'B{i+1} Coverage: 4km'
        ).add_to(m)
    
    # Draw lines between consecutive beacons to show spacing
    for i in range(len(beacons_gps) - 1):
        folium.PolyLine(
            locations=[
                [beacons_gps[i][0], beacons_gps[i][1]],
                [beacons_gps[i+1][0], beacons_gps[i+1][1]]
            ],
            color='orange',
            weight=2,
            opacity=0.5,
            dash_array='10',
            popup=f'Spacing: {geodesic((beacons_gps[i][0], beacons_gps[i][1]), (beacons_gps[i+1][0], beacons_gps[i+1][1])).km:.2f} km'
        ).add_to(m)
    
    # Add a measurement tool
    plugins.MeasureControl(position='topright', primary_length_unit='kilometers').add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen(position='topleft').add_to(m)
    
    # Add mouse position display
    plugins.MousePosition().add_to(m)
    
    # Add minimap
    minimap = plugins.MiniMap(toggle_display=True)
    m.add_child(minimap)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add title box
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; 
                left: 50px; 
                width: 450px; 
                height: auto;
                background-color: white; 
                border:2px solid grey; 
                z-index:9999; 
                font-size:14px;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
                ">
        <h4 style="margin: 0 0 10px 0; color: #1976d2;">
            📡 TDoA Beacon Placement System
        </h4>
        <b>Corridor:</b> {corridor_length_m/1000:.2f} km @ {angle_degrees:.1f}°<br>
        <b>Beacons:</b> {len(beacons_gps)} units<br>
        <b>Avg Spacing:</b> {corridor_length_m/(len(beacons_gps)-1)/1000:.2f} km<br>
        <b>Coverage:</b> 4 km radius per beacon<br>
        <hr style="margin: 8px 0;">
        <small style="color: #666;">
        🟢 Green = Flight corridor<br>
        📍 Colored markers = Beacons (±500m offset)<br>
        ⚪ White dots = Centerline reference<br>
        - - Dashed lines = Perpendicular offsets<br>
        <br>
        Click beacons for details • Use layers to change map style
        </small>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save map
    map_filename = 'beacon_placement_map.html'
    m.save(map_filename)
    
    print(f"\n✅ Interactive map created: {map_filename}")
    
    # Open in browser
    try:
        webbrowser.open('file://' + os.path.realpath(map_filename))
        print(f"🌐 Opening map in your default web browser...")
    except:
        print(f"⚠️  Please open '{map_filename}' manually in your web browser")
    
    return map_filename


def visualize_beacon_placement_static(start_gps, end_gps, beacons_gps, beacons_local, 
                               beacons_base, corridor_params, converter):
    """Create static matplotlib visualization plots"""
    
    corridor_length_m, angle_degrees, _, (end_x, end_y) = corridor_params
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 10))
    
    # ========== PLOT 1: GPS Map View ==========
    ax1 = plt.subplot(2, 3, 1)
    
    # Plot corridor line
    lats = [start_gps[0], end_gps[0]]
    lons = [start_gps[1], end_gps[1]]
    ax1.plot(lons, lats, 'g-', linewidth=3, label='Corridor', zorder=1)
    
    # Plot beacons
    ax1.scatter(beacons_gps[:, 1], beacons_gps[:, 0], s=150, c='red', 
                marker='^', edgecolors='black', linewidth=2, label='Beacons', zorder=5)
    
    # Add labels
    for i, (lat, lon) in enumerate(beacons_gps):
        ax1.text(lon, lat, f'B{i+1}', fontsize=9, ha='center', va='bottom', 
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='yellow', alpha=0.7))
    
    # Mark start and end
    ax1.scatter(start_gps[1], start_gps[0], s=200, c='blue', marker='o', 
                edgecolors='black', linewidth=2, label='Start', zorder=6)
    ax1.scatter(end_gps[1], end_gps[0], s=200, c='purple', marker='s', 
                edgecolors='black', linewidth=2, label='End', zorder=6)
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('GPS Map View')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # ========== PLOT 2: Local Coordinates (Rotated) ==========
    ax2 = plt.subplot(2, 3, 2)
    
    # Plot corridor
    ax2.plot([0, end_x], [0, end_y], 'g-', linewidth=3, label='Corridor', zorder=1)
    
    # Plot beacons
    ax2.scatter(beacons_local[:, 0], beacons_local[:, 1], s=150, c='red', 
                marker='^', edgecolors='black', linewidth=2, label='Beacons', zorder=5)
    
    # Add labels
    for i, (x, y) in enumerate(beacons_local):
        ax2.text(x, y, f'B{i+1}', fontsize=9, ha='center', va='bottom', 
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='yellow', alpha=0.7))
    
    # Mark start and end
    ax2.scatter(0, 0, s=200, c='blue', marker='o', 
                edgecolors='black', linewidth=2, label='Start', zorder=6)
    ax2.scatter(end_x, end_y, s=200, c='purple', marker='s', 
                edgecolors='black', linewidth=2, label='End', zorder=6)
    
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.set_title(f'Local Coordinates (Rotated {angle_degrees:.1f}°)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # ========== PLOT 3: Base Coordinates (Unrotated) ==========
    ax3 = plt.subplot(2, 3, 3)
    
    # Plot corridor
    ax3.plot([0, corridor_length_m], [0, 0], 'g-', linewidth=3, label='Corridor', zorder=1)
    
    # Plot beacons
    ax3.scatter(beacons_base[:, 0], beacons_base[:, 1], s=150, c='red', 
                marker='^', edgecolors='black', linewidth=2, label='Beacons', zorder=5)
    
    # Add labels
    for i, (x, y) in enumerate(beacons_base):
        ax3.text(x, y, f'B{i+1}', fontsize=9, ha='center', va='bottom', 
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='yellow', alpha=0.7))
    
    # Mark start and end
    ax3.scatter(0, 0, s=200, c='blue', marker='o', 
                edgecolors='black', linewidth=2, label='Start', zorder=6)
    ax3.scatter(corridor_length_m, 0, s=200, c='purple', marker='s', 
                edgecolors='black', linewidth=2, label='End', zorder=6)
    
    ax3.set_xlabel('Along Corridor (meters)')
    ax3.set_ylabel('Perpendicular Offset (meters)')
    ax3.set_title('Base Coordinates (Unrotated)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-1000, corridor_length_m + 1000)
    
    # ========== PLOT 4: Beacon Spacing Analysis ==========
    ax4 = plt.subplot(2, 3, 4)
    
    # Calculate spacing along corridor
    if len(beacons_base) > 1:
        spacings = np.diff(beacons_base[:, 0]) / 1000  # Convert to km
        beacon_positions = beacons_base[:-1, 0] / 1000  # Start of each segment
        
        ax4.bar(range(1, len(spacings) + 1), spacings, color='skyblue', 
                edgecolor='black', linewidth=2)
        ax4.axhline(np.mean(spacings), color='red', linestyle='--', 
                   linewidth=2, label=f'Average: {np.mean(spacings):.2f} km')
        ax4.set_xlabel('Beacon Segment')
        ax4.set_ylabel('Spacing (km)')
        ax4.set_title('Inter-Beacon Spacing')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    # ========== PLOT 5: Coordinate Table ==========
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    # Create table data
    table_data = []
    table_data.append(['Beacon', 'Latitude', 'Longitude', 'X (m)', 'Y (m)'])
    
    for i, ((lat, lon), (x, y)) in enumerate(zip(beacons_gps, beacons_local)):
        table_data.append([
            f'B{i+1}',
            f'{lat:.6f}°',
            f'{lon:.6f}°',
            f'{x:.1f}',
            f'{y:.1f}'
        ])
    
    # Create table
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax5.set_title('Beacon Coordinates Table', fontsize=12, fontweight='bold', pad=20)
    
    # ========== PLOT 6: Configuration Summary ==========
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""CONFIGURATION SUMMARY
{'='*40}

CORRIDOR PARAMETERS
  Length: {corridor_length_m/1000:.2f} km
  Angle: {angle_degrees:.2f}° (from North)
  
START POINT
  Lat: {start_gps[0]:.6f}°
  Lon: {start_gps[1]:.6f}°
  
END POINT
  Lat: {end_gps[0]:.6f}°
  Lon: {end_gps[1]:.6f}°
  
BEACON CONFIGURATION
  Total Beacons: {len(beacons_gps)}
  Lateral Offset: ±{beacons_base[0, 1]:.0f} m
  Avg Spacing: {corridor_length_m/(len(beacons_gps)-1)/1000:.2f} km
  Coverage: {corridor_length_m/1000:.2f} km
  
COORDINATE SYSTEM
  UTM Zone: {int((start_gps[1] + 180) / 6) + 1}
  Hemisphere: {'N' if start_gps[0] >= 0 else 'S'}
"""
    
    ax6.text(0.1, 0.5, summary_text, fontfamily='monospace', fontsize=10,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def export_results(beacons_gps, beacons_local, beacons_base, params, corridor_params):
    """Export beacon coordinates to various formats"""
    
    print("\n" + "=" * 60)
    print("EXPORT OPTIONS")
    print("=" * 60)
    
    export_choice = input("\nExport beacon coordinates? (y/n): ").lower()
    
    if export_choice == 'y':
        # JSON export
        export_data = {
            'corridor': {
                'start': {
                    'latitude': params['start_lat'],
                    'longitude': params['start_lon']
                },
                'end': {
                    'latitude': params['end_lat'],
                    'longitude': params['end_lon']
                },
                'length_km': corridor_params[0] / 1000,
                'angle_degrees': corridor_params[1]
            },
            'beacons': []
        }
        
        for i, (gps, local, base) in enumerate(zip(beacons_gps, beacons_local, beacons_base)):
            beacon_data = {
                'id': f'B{i+1}',
                'gps': {
                    'latitude': float(gps[0]),
                    'longitude': float(gps[1])
                },
                'local': {
                    'x_meters': float(local[0]),
                    'y_meters': float(local[1])
                },
                'base': {
                    'x_meters': float(base[0]),
                    'y_meters': float(base[1])
                }
            }
            export_data['beacons'].append(beacon_data)
        
        # Save JSON
        json_filename = 'beacon_coordinates.json'
        with open(json_filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"\nJSON data saved to: {json_filename}")
        
        # Save CSV (GPS coordinates)
        csv_filename = 'beacon_gps_coordinates.csv'
        with open(csv_filename, 'w') as f:
            f.write('Beacon_ID,Latitude,Longitude\n')
            for i, gps in enumerate(beacons_gps):
                f.write(f'B{i+1},{gps[0]:.8f},{gps[1]:.8f}\n')
        print(f"GPS coordinates saved to: {csv_filename}")
        
        # Save CSV (Local coordinates)
        csv_local_filename = 'beacon_local_coordinates.csv'
        with open(csv_local_filename, 'w') as f:
            f.write('Beacon_ID,X_meters,Y_meters\n')
            for i, local in enumerate(beacons_local):
                f.write(f'B{i+1},{local[0]:.2f},{local[1]:.2f}\n')
        print(f"Local coordinates saved to: {csv_local_filename}")
        
        # Generate Google Maps URL
        print("\n" + "=" * 60)
        print("GOOGLE MAPS VISUALIZATION")
        print("=" * 60)
        
        # Create URL with all beacon markers
        base_url = "https://www.google.com/maps/dir/"
        waypoints = []
        for gps in beacons_gps:
            waypoints.append(f"{gps[0]},{gps[1]}")
        
        maps_url = base_url + "/".join(waypoints)
        print(f"\nGoogle Maps URL (copy and paste in browser):")
        print(maps_url)


def main():
    """Main function"""
    try:
        # Get GPS input from user
        params = get_gps_input()
        
        # Calculate corridor parameters
        corridor_length_m, angle_degrees, converter, end_local = calculate_corridor_parameters(
            params['start_lat'], params['start_lon'],
            params['end_lat'], params['end_lon']
        )
        
        corridor_params = (corridor_length_m, angle_degrees, converter, end_local)
        
        print("\n" + "=" * 60)
        print("CALCULATING BEACON PLACEMENT...")
        print("=" * 60)
        
        # Place beacons
        beacons_local, beacons_base, num_beacons = place_beacons_gps(
            corridor_length_m, 
            angle_degrees, 
            params['num_beacons'],
            params['offset_m'],
            end_local
        )
        
        # Convert beacons to GPS coordinates
        beacons_gps = converter.batch_local_to_gps(beacons_local)
        
        print(f"\nBeacon placement complete!")
        print(f"  Corridor length: {corridor_length_m/1000:.2f} km")
        print(f"  Corridor angle: {angle_degrees:.2f}° (from North)")
        print(f"  Number of beacons: {num_beacons}")
        
        # Calculate actual average spacing
        if num_beacons > 1:
            actual_spacing_km = corridor_length_m / (num_beacons - 1) / 1000
            print(f"  Average spacing: {actual_spacing_km:.2f} km")
        
        # Print beacon coordinates
        print("\n" + "=" * 60)
        print("BEACON GPS COORDINATES")
        print("=" * 60)
        for i, gps in enumerate(beacons_gps):
            print(f"Beacon B{i+1}: {gps[0]:.6f}°, {gps[1]:.6f}°")
        
        # Create interactive Google Maps-style visualization
        print("\n" + "=" * 60)
        print("CREATING INTERACTIVE MAP...")
        print("=" * 60)
        
        create_interactive_map(
            (params['start_lat'], params['start_lon']),
            (params['end_lat'], params['end_lon']),
            beacons_gps,
            beacons_local,
            beacons_base,
            corridor_params
        )
        
        # Ask if user wants static plots too
        show_static = input("\nShow static matplotlib plots? (y/n): ").lower()
        
        if show_static == 'y':
            # Visualize with static plots
            visualize_beacon_placement_static(
                (params['start_lat'], params['start_lon']),
                (params['end_lat'], params['end_lon']),
                beacons_gps,
                beacons_local,
                beacons_base,
                corridor_params,
                converter
            )
        
        # Export results
        export_results(beacons_gps, beacons_local, beacons_base, params, corridor_params)
        
        print("\n" + "=" * 60)
        print("Thank you for using the GPS-Based TDoA Beacon Placement System!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
