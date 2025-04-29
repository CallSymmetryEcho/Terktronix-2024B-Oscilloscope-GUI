import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from PIL import Image
import io
from tqdm import tqdm
import math
import os

def track_nanosphere_dynamics(video_path, display_results=True, save_output=False, output_path=None,binary_value=100):
    """
    Track nanosphere trajectory, speed, direction, and acceleration in a video.
    
    Parameters:
    -----------
    video_path : str
        Path to the input video file
    display_results : bool
        Whether to display results in the notebook
    save_output : bool
        Whether to save the output video
    output_path : str
        Path to save the output video (if save_output is True)
    
    Returns:
    --------
    dynamics_data : dict
        Dictionary containing trajectory, speeds, directions, and accelerations
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return {}
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize output video writer if needed
    if save_output:
        if output_path is None:
            output_path = video_path.rsplit('.', 1)[0] + '_analyzed.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Data structures to store dynamics information
    trajectory = []
    timestamps = []
    speeds = []
    directions = []
    accelerations = []
    
    # Initialize parameters for tracking
    prev_center = None
    prev_speed = None
    prev_direction = None
    frame_count = 0
    time_delta = 1.0 / fps if fps > 0 else 0.033  # Time between frames in seconds
    
    # Create a persistent trajectory visualization layer
    trajectory_layer = np.zeros((height, width, 3), dtype=np.uint8)
    
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc='Processing') as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            pbar.update(1)
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing to enhance the nanosphere
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, binary_value, 255, cv2.THRESH_BINARY_INV)
            
            # Apply morphological operations to clean up noise
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find nanosphere contour (assuming it's one of the larger contours)
            if contours:
                # Sort contours by area (descending)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                # Filter contours, keeping only those likely to be nanospheres
                valid_contours = []
                for cnt in contours[:5]:  # Check the 5 largest contours
                    area = cv2.contourArea(cnt)
                    if area > 10 and area < 5000:  # Adjust based on your nanosphere size
                        # Calculate circularity (nanospheres should be circular)
                        perimeter = cv2.arcLength(cnt, True)
                        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                        
                        # Nanospheres should have circularity close to 1
                        if circularity > 0.7:  # Adjust this threshold as needed
                            valid_contours.append(cnt)
                
                # If valid contours found
                if valid_contours:
                    # Use the first valid contour
                    nanosphere_contour = valid_contours[0]
                    
                    # Calculate moments to find center
                    M = cv2.moments(nanosphere_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        center = (cx, cy)
                        current_time = frame_count * time_delta
                        
                        # Add to trajectory
                        trajectory.append(center)
                        timestamps.append(current_time)
                        
                        # Calculate speed and direction if we have at least two points
                        if prev_center is not None:
                            # Calculate displacement
                            dx = center[0] - prev_center[0]
                            dy = center[1] - prev_center[1]
                            
                            # Calculate distance
                            distance = np.sqrt(dx**2 + dy**2)
                            
                            # Calculate instantaneous speed (pixels per second)
                            instant_speed = distance / time_delta
                            
                            # Apply smoothing for Brownian motion:
                            # Using exponential moving average with alpha=0.3
                            # Lower alpha gives more smoothing (0.3 means 30% weight to new reading)
                            alpha = 0.3
                            if len(speeds) > 0:
                                smoothed_speed = (alpha * instant_speed) + ((1 - alpha) * speeds[-1])
                            else:
                                smoothed_speed = instant_speed
                                
                            speeds.append(smoothed_speed)
                            
                            # Calculate direction (in radians, then convert to degrees)
                            direction = math.atan2(dy, dx)
                            direction_deg = math.degrees(direction)
                            directions.append(direction_deg)
                            
                            # Calculate acceleration if we have at least two speed measurements
                            if prev_speed is not None:
                                # Calculate speed change
                                speed_change = smoothed_speed - prev_speed
                                
                                # Calculate acceleration (pixels per second squared)
                                acceleration = speed_change / time_delta
                                
                                # Calculate tangential and normal components of acceleration
                                if prev_direction is not None:
                                    # Convert directions to radians for calculations
                                    prev_dir_rad = math.radians(prev_direction)
                                    curr_dir_rad = math.radians(direction_deg)
                                    
                                    # Calculate angular change
                                    angular_change = curr_dir_rad - prev_dir_rad
                                    # Normalize to [-π, π]
                                    angular_change = (angular_change + math.pi) % (2 * math.pi) - math.pi
                                    
                                    # Angular velocity (radians per second)
                                    angular_velocity = angular_change / time_delta
                                    
                                    # Tangential acceleration (due to speed change)
                                    tangential_acc = speed_change / time_delta
                                    
                                    # Normal acceleration (due to direction change)
                                    # a_n = v² / r = v * ω
                                    normal_acc = smoothed_speed * abs(angular_velocity)
                                    
                                    # Total acceleration magnitude
                                    total_acc = math.sqrt(tangential_acc**2 + normal_acc**2)
                                    
                                    accelerations.append((tangential_acc, normal_acc, total_acc))
                                else:
                                    accelerations.append((acceleration, 0, acceleration))
                            
                            # Store current values for next iteration
                            prev_speed = smoothed_speed
                            prev_direction = direction_deg
                        else:
                            # First point, no speed yet
                            speeds.append(0)
                            directions.append(0)
                            accelerations.append((0, 0, 0))
                        
                        # Draw current position
                        cv2.circle(frame, center, 8, (0, 255, 0), -1)
                        cv2.circle(frame, center, 10, (255, 255, 255), 2)  # White outline for visibility
                        
                        # Draw direction and speed vector if available
                        if len(speeds) > 0 and speeds[-1] > 0:
                            # Scale vector length based on speed (more direct relationship)
                            vector_scale = min(100, max(10, speeds[-1]))
                            
                            # Scale arrow thickness based on speed
                            arrow_thickness = min(5, max(2, int(speeds[-1] / 20) + 1))
                            
                            # Calculate endpoint of the vector
                            angle_rad = math.radians(directions[-1])
                            end_x = int(center[0] + vector_scale * math.cos(angle_rad))
                            end_y = int(center[1] + vector_scale * math.sin(angle_rad))
                            
                            # Draw velocity vector (green) with variable thickness
                            cv2.arrowedLine(frame, center, (end_x, end_y), (0, 255, 0), arrow_thickness)
                            
                            # Draw acceleration vector if available (red)
                            if len(accelerations) > 0 and accelerations[-1][2] > 0:
                                # Scale acceleration for visualization
                                acc_scale = min(50, max(5, accelerations[-1][2] / 10))
                                
                                # If we have at least 3 points, we can calculate acceleration direction
                                if len(trajectory) >= 3:
                                    # Get acceleration direction from last 3 points
                                    p1 = trajectory[-3]
                                    p2 = trajectory[-2]
                                    p3 = trajectory[-1]
                                    
                                    # Estimate acceleration direction
                                    # This is a simple approximation based on the change in velocity
                                    dx1 = p2[0] - p1[0]
                                    dy1 = p2[1] - p1[1]
                                    dx2 = p3[0] - p2[0]
                                    dy2 = p3[1] - p2[1]
                                    
                                    # Change in velocity components
                                    dvx = dx2 - dx1
                                    dvy = dy2 - dy1
                                    
                                    # Acceleration direction
                                    acc_angle = math.atan2(dvy, dvx)
                                    
                                    # Calculate endpoint of acceleration vector
                                    acc_end_x = int(center[0] + acc_scale * math.cos(acc_angle))
                                    acc_end_y = int(center[1] + acc_scale * math.sin(acc_angle))
                                    
                                    # Draw acceleration vector (red)
                                    cv2.arrowedLine(frame, center, (acc_end_x, acc_end_y), (0, 0, 255), 2)
                        
                        # Update the trajectory on the persistent layer
                        if len(trajectory) > 1:
                            # Draw the most recent segment with a bright color
                            cv2.line(trajectory_layer, trajectory[-2], trajectory[-1], (0, 255, 255), 2)
                            
                            # Gradually color the entire trajectory based on speed
                            max_points = min(len(trajectory), 500)  # Limit to prevent slowdown
                            for i in range(1, max_points):
                                # Get position in the trajectory (normalized)
                                idx = len(trajectory) - max_points + i
                                
                                if idx > 0:
                                    # Color based on speed
                                    if idx < len(speeds):
                                        # Normalize speed for coloring
                                        speed_norm = min(1.0, speeds[idx] / 100.0)
                                        
                                        # Create color gradient based on speed
                                        # Blue (slow) -> Green -> Yellow -> Red (fast)
                                        if speed_norm < 0.33:
                                            r = 0
                                            g = int(255 * (speed_norm / 0.33))
                                            b = int(255 * (1 - speed_norm / 0.33))
                                        elif speed_norm < 0.66:
                                            normalized = (speed_norm - 0.33) / 0.33
                                            r = int(255 * normalized)
                                            g = 255
                                            b = 0
                                        else:
                                            normalized = (speed_norm - 0.66) / 0.34
                                            r = 255
                                            g = int(255 * (1 - normalized))
                                            b = 0
                                        
                                        color = (b, g, r)  # BGR format for OpenCV
                                        thickness = max(1, min(4, int(speed_norm * 5)))
                                        cv2.line(trajectory_layer, trajectory[idx-1], trajectory[idx], color, thickness)
                    
                    # Draw nanosphere contour
                    cv2.drawContours(frame, [nanosphere_contour], -1, (0, 255, 255), 2)
                        
                        # Update previous center
                    prev_center = center
            
            # Overlay the trajectory on the frame
            display_frame = frame.copy()
            
            # Create temporary image for overlay
            overlay = np.zeros_like(display_frame)
            # Copy trajectory layer only where it has content
            overlay[trajectory_layer > 0] = trajectory_layer[trajectory_layer > 0]
            # Blend images
            display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)
            
            # Add information overlay
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Points: {len(trajectory)}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add speed and acceleration if available
            if len(speeds) > 0:
                cv2.putText(display_frame, f"Speed: {speeds[-1]:.1f} px/s", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if len(accelerations) > 0:
                cv2.putText(display_frame, f"Accel: {accelerations[-1][2]:.1f} px/s²", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add speed color legend
            legend_x, legend_y = width - 200, 30
            cv2.putText(display_frame, "Speed Scale:", (legend_x, legend_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw color gradient legend
            for i in range(100):
                pos = i / 100
                if pos < 0.33:
                    r, g, b = 0, int(255 * (pos / 0.33)), int(255 * (1 - pos / 0.33))
                elif pos < 0.66:
                    normalized = (pos - 0.33) / 0.33
                    r, g, b = int(255 * normalized), 255, 0
                else:
                    normalized = (pos - 0.66) / 0.34
                    r, g, b = 255, int(255 * (1 - normalized)), 0
                
                cv2.line(display_frame, (legend_x + i, legend_y + 15), (legend_x + i, legend_y + 25), (b, g, r), 1)
            
            cv2.putText(display_frame, "Slow", (legend_x, legend_y + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.putText(display_frame, "Fast", (legend_x + 80, legend_y + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Add direction and acceleration legend
            cv2.arrowedLine(display_frame, (legend_x + 10, legend_y + 60), (legend_x + 50, legend_y + 60), (0, 255, 0), 2)
            cv2.putText(display_frame, "Velocity", (legend_x + 60, legend_y + 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            cv2.arrowedLine(display_frame, (legend_x + 10, legend_y + 80), (legend_x + 50, legend_y + 80), (0, 0, 255), 2)
            cv2.putText(display_frame, "Acceleration", (legend_x + 60, legend_y + 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Display the result
            if display_results and frame_count % 5 == 0:  # Update display every 5 frames
                clear_output(wait=True)
                plt_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(10, 8))
                plt.imshow(plt_image)
                plt.axis('off')
                plt.show()
            
            # Write to output video if requested
            if save_output:
                out.write(display_frame)
        
    # Release resources
    cap.release()
    if save_output:
        out.release()
    
    print(f"Processed {frame_count} frames, tracked {len(trajectory)} positions")
    
    # Compile all dynamics data
    dynamics_data = {
        'trajectory': trajectory,
        'timestamps': timestamps,
        'speeds': speeds,
        'directions': directions,
        'accelerations': accelerations
    }
    
    return dynamics_data

def plot_dynamics(dynamics_data, figsize=(12, 10), save_plots=False, output_dir=None):
    """
    Visualize nanosphere dynamics with multiple plots.
    
    Parameters:
    -----------
    dynamics_data : dict
        Dictionary containing trajectory, speeds, directions, and accelerations
    figsize : tuple
        Figure size for plots
    save_plots : bool
        Whether to save the plots to files
    output_dir : str
        Directory to save the plots (if save_plots is True)
    """
    # Create output directory if saving is requested
    if save_plots:
        if output_dir is None:
            output_dir = "nanosphere_analysis"
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
    if not dynamics_data or len(dynamics_data.get('trajectory', [])) < 3:
        print("Insufficient dynamics data to plot")
        return
    
    trajectory = dynamics_data['trajectory']
    timestamps = dynamics_data['timestamps']
    speeds = dynamics_data['speeds']
    directions = dynamics_data['directions']
    accelerations = dynamics_data['accelerations']
    
    # Extract coordinates
    x = [point[0] for point in trajectory]
    y = [point[1] for point in trajectory]
    
    # Time values relative to start
    rel_times = [t - timestamps[0] for t in timestamps]
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # 1. Trajectory plot with speed coloring
    ax1 = fig.add_subplot(221)
    
    # Create a colormap based on speed
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create scalar speed values for coloring
    speed_values = np.array(speeds[1:])  # Skip first (it's 0)
    
    # Normalize for colormap
    norm = plt.Normalize(0, np.percentile(speed_values, 90))  # Use 90th percentile to avoid outliers
    
    # Plot trajectory with speed coloring
    for i, segment in enumerate(segments):
        if i < len(speed_values):
            color = plt.cm.jet(norm(speed_values[i]))
            ax1.plot([segment[0, 0], segment[1, 0]], 
                    [segment[0, 1], segment[1, 1]], 
                    color=color, 
                    linewidth=2)
    
    # Mark start and end points
    ax1.scatter(x[0], y[0], color='green', s=100, label='Start')
    ax1.scatter(x[-1], y[-1], color='red', s=100, label='End')
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label('Speed (pixels/s)')
    
    ax1.set_title('Nanosphere Trajectory (colored by speed)')
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Invert y-axis to match image coordinates
    ax1.legend()
    
    # Save trajectory plot if requested
    if save_plots:
        plt.savefig(f"{output_dir}/trajectory_plot.png", dpi=300, bbox_inches='tight')
    
    # 2. Speed vs. Time plot
    ax2 = fig.add_subplot(222)
    print(f'beforetreat{len(rel_times)}')
    #rel_times = rel_times[1:]
    print(f'aftertreat{len(rel_times)}')
    ax2.plot(rel_times, speeds, 'b-', linewidth=2)
    ax2.set_title('Speed over Time')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Speed (pixels/s)')
    ax2.grid(True, alpha=0.3)
    
    # Add moving average for trend
    window_size = min(10, len(speeds) // 5) if len(speeds) > 10 else 3
    if window_size > 1:
        moving_avg = np.convolve(speeds, np.ones(window_size)/window_size, mode='valid')
        # Plot at the correct x positions (centered)
        ma_times = rel_times[window_size-1:]
        ax2.plot(ma_times, moving_avg, 'r-', linewidth=2, label=f'{window_size}-point Moving Avg')
        ax2.legend()
    
    # Save speed plot if requested
    if save_plots:
        plt.savefig(f"{output_dir}/speed_vs_time.png", dpi=300, bbox_inches='tight')
    
    # 3. Direction (Angle) vs. Time plot
    ax3 = fig.add_subplot(223)
    
    # Normalize directions for continuous plot (avoid jumps between -180° and 180°)
    normalized_directions = np.copy(directions)
    for i in range(1, len(normalized_directions)):
        diff = normalized_directions[i] - normalized_directions[i-1]
        if diff > 180:
            normalized_directions[i:] -= 360
        elif diff < -180:
            normalized_directions[i:] += 360
    
    ax3.plot(rel_times, normalized_directions, 'g-', linewidth=2)
    ax3.set_title('Movement Direction over Time')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Direction (degrees)')
    ax3.grid(True, alpha=0.3)
    
    # Save direction plot if requested
    if save_plots:
        plt.savefig(f"{output_dir}/direction_vs_time.png", dpi=300, bbox_inches='tight')
    
    # 4. Acceleration vs. Time plot
    ax4 = fig.add_subplot(224)
    
    # Extract acceleration components
    if accelerations:
        tangential_acc = [acc[0] for acc in accelerations]
        normal_acc = [acc[1] for acc in accelerations]
        total_acc = [acc[2] for acc in accelerations]
        rel_times = rel_times[1:]
        ax4.plot(rel_times, total_acc, 'r-', label='Total', linewidth=2)
        ax4.plot(rel_times, tangential_acc, 'g--', label='Tangential', linewidth=1.5)
        ax4.plot(rel_times, normal_acc, 'b--', label='Normal', linewidth=1.5)
        ax4.set_title('Acceleration Components over Time')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Acceleration (pixels/s²)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Save acceleration plot if requested
        if save_plots:
            plt.savefig(f"{output_dir}/acceleration_vs_time.png", dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Position heatmap (separate figure)
    plt.figure(figsize=(figsize[0]//2, figsize[1]//2))
    heatmap = plt.hist2d(x, y, bins=50, cmap='hot')
    plt.colorbar(label='Frequency')
    plt.title('Nanosphere Position Heatmap')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save heatmap if requested
    if save_plots:
        plt.savefig(f"{output_dir}/position_heatmap.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # 6. Velocity vector field visualization (separate figure)
    if len(trajectory) > 10:
        plt.figure(figsize=(figsize[0]//2, figsize[1]//2))
        
        # Calculate velocities from trajectory
        vx = []
        vy = []
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            vx.append(dx / time_delta if 'time_delta' in locals() else dx)
            vy.append(dy / time_delta if 'time_delta' in locals() else dy)
        
        # Use trajectory points as positions for vectors
        points_x = x[1:]
        points_y = y[1:]
        
        # Plot vector field
        plt.quiver(points_x, points_y, vx, vy, np.array(speeds[1:]), 
                   cmap='jet', scale=500, width=0.005)
        
        plt.colorbar(label='Speed (pixels/s)')
        plt.title('Nanosphere Velocity Vector Field')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save vector field plot if requested
        if save_plots:
            plt.savefig(f"{output_dir}/velocity_vector_field.png", dpi=300, bbox_inches='tight')
        
        plt.show()

def save_dynamics_to_csv(dynamics_data, output_dir=None):
    """
    Save dynamics data to CSV files.
    
    Parameters:
    -----------
    dynamics_data : dict
        Dictionary containing trajectory, speeds, directions, and accelerations
    output_dir : str
        Directory to save the CSV files
    """
    if not dynamics_data:
        print("No dynamics data to save")
        return
    
    import os
    import pandas as pd
    
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = "nanosphere_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    trajectory = dynamics_data['trajectory']
    timestamps = dynamics_data['timestamps']
    speeds = dynamics_data['speeds']
    directions = dynamics_data['directions']
    accelerations = dynamics_data.get('accelerations', [])
    
    # Create DataFrame for position and basic dynamics
    df_basic = pd.DataFrame({
        'timestamp': timestamps,
        'x_position': [p[0] for p in trajectory],
        'y_position': [p[1] for p in trajectory],
        'speed': speeds,
        'direction': directions
    })
    
    # Save to CSV
    df_basic.to_csv(f"{output_dir}/nanosphere_basic_dynamics.csv", index=False)
    
    # Create DataFrame for acceleration if available
    if accelerations:
        df_acc = pd.DataFrame({
            'timestamp': timestamps,
            'tangential_acc': [acc[0] for acc in accelerations],
            'normal_acc': [acc[1] for acc in accelerations],
            'total_acc': [acc[2] for acc in accelerations]
        })
        
        # Save to CSV
        df_acc.to_csv(f"{output_dir}/nanosphere_acceleration.csv", index=False)
    
    print(f"Dynamics data saved to CSV files in {output_dir} directory")

def analyze_dynamics(dynamics_data, save_to_csv=False, output_dir=None):
    """
    Analyze nanosphere dynamics and print statistics.
    
    Parameters:
    -----------
    dynamics_data : dict
        Dictionary containing trajectory, speeds, directions, and accelerations
    save_to_csv : bool
        Whether to save the dynamics data to CSV files
    output_dir : str
        Directory to save the CSV files (if save_to_csv is True)
    """
    if not dynamics_data or len(dynamics_data.get('trajectory', [])) < 3:
        print("Insufficient dynamics data for analysis")
        return
    
    trajectory = dynamics_data['trajectory']
    timestamps = dynamics_data['timestamps']
    speeds = dynamics_data['speeds']
    directions = dynamics_data['directions']
    accelerations = dynamics_data.get('accelerations', [])
    
    # Calculate statistics
    duration = timestamps[-1] - timestamps[0]
    
    # Distance traveled (total path length)
    total_distance = 0
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i-1][0]
        dy = trajectory[i][1] - trajectory[i-1][1]
        total_distance += np.sqrt(dx**2 + dy**2)
    
    # Direct displacement (straight-line distance from start to end)
    dx_total = trajectory[-1][0] - trajectory[0][0]
    dy_total = trajectory[-1][1] - trajectory[0][1]
    displacement = np.sqrt(dx_total**2 + dy_total**2)
    
    # Speed statistics
    avg_speed = np.mean(speeds[1:])  # Skip first (it's 0)
    max_speed = np.max(speeds)
    min_speed = np.min(speeds[1:]) if len(speeds) > 1 else 0
    
    # Acceleration statistics
    if accelerations:
        total_accs = [acc[2] for acc in accelerations]
        avg_acc = np.mean(total_accs[1:])  # Skip first (it's 0)
        max_acc = np.max(total_accs)
    else:
        avg_acc = max_acc = 0
    
    # Straightness ratio (1.0 = perfectly straight path)
    straightness = displacement / total_distance if total_distance > 0 else 0
    
    # Print analysis
    print("\n===== Nanosphere Dynamics Analysis =====")
    print(f"Total tracking duration: {duration:.2f} seconds")
    print(f"Total path distance: {total_distance:.2f} pixels")
    print(f"Direct displacement: {displacement:.2f} pixels")
    print(f"Path straightness: {straightness:.3f} (1.0 = straight line)")
    print("\nSpeed Analysis:")
    print(f"  Average speed: {avg_speed:.2f} pixels/second")
    print(f"  Maximum speed: {max_speed:.2f} pixels/second")
    print(f"  Minimum speed: {min_speed:.2f} pixels/second")
    print("\nAcceleration Analysis:")
    print(f"  Average acceleration: {avg_acc:.2f} pixels/second²")
    print(f"  Maximum acceleration: {max_acc:.2f} pixels/second²")
    
    # Direction analysis - check if there's a consistent direction
    # Convert to radians for calculation
    dir_radians = np.radians(directions)
    
    # Calculate mean direction using circular statistics
    sin_sum = np.sum(np.sin(dir_radians))
    cos_sum = np.sum(np.cos(dir_radians))
    mean_dir_rad = np.arctan2(sin_sum, cos_sum)
    mean_dir_deg = np.degrees(mean_dir_rad)
    
    # Calculate directional consistency (0 = random, 1 = perfectly aligned)
    r = np.sqrt(sin_sum**2 + cos_sum**2) / len(dir_radians)
    
    print("\nDirection Analysis:")
    print(f"  Mean direction: {mean_dir_deg:.1f} degrees")
    print(f"  Directional consistency: {r:.3f} (0 = random, 1 = consistent)")
    
    # Additional analysis: check for periodic behavior
    if len(speeds) > 50:
        from scipy import signal
        
        # Compute auto-correlation of speed
        autocorr = signal.correlate(speeds, speeds, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # Find peaks in autocorrelation (potential periods)
        peaks, _ = signal.find_peaks(autocorr, height=0.3, distance=10)
        
        if len(peaks) > 0:
            # Calculate period in frames
            period_frames = peaks[0]
            # Convert to seconds
            period_seconds = period_frames * (timestamps[1] - timestamps[0])
            
            print("\nPeriodic Behavior Detection:")
            print(f"  Possible periodic behavior detected with period: {period_seconds:.2f} seconds")
        else:
            print("\nNo significant periodic behavior detected")
    
    # Save data to CSV if requested
    if save_to_csv:
        save_dynamics_to_csv(dynamics_data, output_dir)

# Example usage
if __name__ == "__main__":
    # Configuration
    video_path = f'/Volumes/T7_Shield/Hydrogel_sensor/20250307_data_lh_acce/constant_speed/constant_speed-1.mp4'
    output_dir = os.path.dirname(video_path)
    save_output = True
    output_path = f'{output_dir}/output.mp4'
   
    
    # Track and analyze nanosphere dynamics
    print("Starting nanosphere tracking and analysis...")
    dynamics_data = track_nanosphere_dynamics(video_path, display_results=False, 
                                             save_output=save_output, 
                                             output_path=output_path,binary_value=132)
    
    # Plot visualization of dynamics and save plots
    print("Generating dynamics plots...")
    plot_dynamics(dynamics_data, save_plots=True, output_dir=output_dir)
    
    # Print statistical analysis and save data
    print("Analyzing dynamics data...")
    analyze_dynamics(dynamics_data, save_to_csv=True, output_dir=output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir} directory.")