import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import math
from tqdm import tqdm
import pandas as pd
import os

def analyze_nanosphere_component(video_path, direction='y', display_results=True, save_output=False, 
                                output_path=None, output_dir=None,binary_value=100):
    """
    Track nanosphere motion and analyze velocity and acceleration for a specific direction component.
    
    Parameters:
    -----------
    video_path : str
        Path to the input video file
    direction : str
        Direction to analyze: 'x', 'y', or 'both'
    display_results : bool
        Whether to display results in the notebook
    save_output : bool
        Whether to save the output video
    output_path : str
        Path to save the output video (if save_output is True)
    output_dir : str
        Directory to save plots and data (defaults to 'nanosphere_{direction}_analysis')
        
    Returns:
    --------
    dynamics : dict
        Dictionary containing trajectory and directional dynamics data
    """
    # Validate direction parameter
    if direction not in ['x', 'y', 'both']:
        raise ValueError("Direction must be 'x', 'y', or 'both'")
    
    # Create output directory if needed
    treatment_dirname = os.path.dirname(video_path)
    if output_dir is None:
        output_dir = f"{treatment_dirname}/nanosphere_{direction}_analysis"
    if save_output or output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
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
            output_path = os.path.join(output_dir, f"{direction}_component_analysis.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Data structures to store dynamics information
    trajectory = []
    timestamps = []
    x_positions = []
    y_positions = []
    x_velocities = []
    y_velocities = []
    x_accelerations = []
    y_accelerations = []
    
    # Initialize parameters for tracking
    prev_center = None
    prev_x_velocity = None
    prev_y_velocity = None
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
                        
                        # Add to trajectory data
                        trajectory.append(center)
                        timestamps.append(current_time)
                        x_positions.append(cx)
                        y_positions.append(cy)
                        
                        # Calculate velocity components if we have at least two points
                        if prev_center is not None:
                            # Calculate displacements
                            dx = cx - prev_center[0]  # Positive = rightward
                            dy = cy - prev_center[1]  # Positive = downward in image coordinates
                            
                            # Calculate instantaneous velocities (pixels per second)
                            instant_x_velocity = dx / time_delta
                            instant_y_velocity = dy / time_delta
                            
                            # Apply smoothing to handle Brownian motion
                            alpha = 0.3  # Smoothing factor (lower = more smoothing)
                            
                            # X velocity smoothing
                            if x_velocities:
                                smoothed_x_velocity = (alpha * instant_x_velocity) + ((1 - alpha) * x_velocities[-1])
                            else:
                                smoothed_x_velocity = instant_x_velocity
                            
                            # Y velocity smoothing    
                            if y_velocities:
                                smoothed_y_velocity = (alpha * instant_y_velocity) + ((1 - alpha) * y_velocities[-1])
                            else:
                                smoothed_y_velocity = instant_y_velocity
                                
                            x_velocities.append(smoothed_x_velocity)
                            y_velocities.append(smoothed_y_velocity)
                            
                            # Calculate accelerations if we have at least two velocity measurements
                            if prev_x_velocity is not None and prev_y_velocity is not None:
                                # Calculate velocity changes
                                x_velocity_change = smoothed_x_velocity - prev_x_velocity
                                y_velocity_change = smoothed_y_velocity - prev_y_velocity
                                
                                # Calculate accelerations (pixels per second squared)
                                x_acceleration = x_velocity_change / time_delta
                                y_acceleration = y_velocity_change / time_delta
                                
                                # Apply smoothing to accelerations
                                if x_accelerations:
                                    smoothed_x_acceleration = (alpha * x_acceleration) + ((1 - alpha) * x_accelerations[-1])
                                    smoothed_y_acceleration = (alpha * y_acceleration) + ((1 - alpha) * y_accelerations[-1])
                                else:
                                    smoothed_x_acceleration = x_acceleration
                                    smoothed_y_acceleration = y_acceleration
                                    
                                x_accelerations.append(smoothed_x_acceleration)
                                y_accelerations.append(smoothed_y_acceleration)
                            else:
                                # First velocity, no acceleration yet
                                x_accelerations.append(0)
                                y_accelerations.append(0)
                            
                            # Store current values for next iteration
                            prev_x_velocity = smoothed_x_velocity
                            prev_y_velocity = smoothed_y_velocity
                        else:
                            # First point, no velocity yet
                            x_velocities.append(0)
                            y_velocities.append(0)
                            x_accelerations.append(0)
                            y_accelerations.append(0)
                        
                        # Draw current position
                        cv2.circle(frame, center, 8, (0, 255, 0), -1)
                        cv2.circle(frame, center, 10, (255, 255, 255), 2)  # White outline for visibility
                        
                        # Draw velocity vectors based on selected direction
                        if (direction == 'x' or direction == 'both') and len(x_velocities) > 0:
                            # X-component vector (horizontal)
                            vector_scale_x = min(50, max(5, abs(x_velocities[-1])))
                            arrow_thickness_x = max(1, min(4, int(abs(x_velocities[-1]) / 10) + 1))
                            
                            # Direction (right or left)
                            dir_x = 1 if x_velocities[-1] > 0 else -1
                            
                            # Calculate endpoint of x-velocity vector
                            end_x = int(center[0] + dir_x * vector_scale_x)
                            end_y = int(center[1])  # Same y (horizontal vector)
                            #print(f'{end_x},{end_y}')
                            # Draw x-velocity vector (green for rightward, orange for leftward)
                            color_x = (0, 255, 0) if dir_x > 0 else (0, 165, 255)
                            center_tuple = tuple(map(int, center))
                            end_tuple = (int(end_x), int(end_y))
                            cv2.arrowedLine(frame, center, (end_x, end_y), color_x, arrow_thickness_x)
                            
                        if (direction == 'y' or direction == 'both') and len(y_velocities) > 0:
                            # Y-component vector (vertical)
                            vector_scale_y = min(50, max(5, abs(y_velocities[-1])))
                            arrow_thickness_y = max(1, min(4, int(abs(y_velocities[-1]) / 10) + 1))
                            
                            # Direction (up or down)
                            dir_y = 1 if y_velocities[-1] > 0 else -1
                            
                            # Calculate endpoint of y-velocity vector
                            end_x = center[0]  # Same x (vertical vector)
                            end_y = int(center[1] + dir_y * vector_scale_y)
                            
                            # Draw y-velocity vector (blue for downward, red for upward)
                            color_y = (255, 0, 0) if dir_y > 0 else (0, 0, 255)
                            cv2.arrowedLine(frame, center, (end_x, end_y), color_y, arrow_thickness_y)
                        
                        # Update the trajectory on the persistent layer
                        if len(trajectory) > 1:
                            # Color based on selected direction
                            if direction == 'x':
                                # Color based on x-velocity
                                if len(x_velocities) > 1:
                                    velocity_norm = min(1.0, abs(x_velocities[-1]) / 50.0)
                                    intensity = int(128 + 127 * velocity_norm)
                                    
                                    if x_velocities[-1] > 0:  # Rightward
                                        color = (0, intensity, 0)  # Green
                                    else:  # Leftward
                                        color = (0, intensity//2, intensity)  # Orange-ish
                                    
                                    cv2.line(trajectory_layer, trajectory[-2], trajectory[-1], color, 2)
                            
                            elif direction == 'y':
                                # Color based on y-velocity
                                if len(y_velocities) > 1:
                                    velocity_norm = min(1.0, abs(y_velocities[-1]) / 50.0)
                                    intensity = int(128 + 127 * velocity_norm)
                                    
                                    if y_velocities[-1] > 0:  # Downward
                                        color = (intensity, 0, 0)  # Blue
                                    else:  # Upward
                                        color = (0, 0, intensity)  # Red
                                    
                                    cv2.line(trajectory_layer, trajectory[-2], trajectory[-1], color, 2)
                            
                            else:  # 'both'
                                # Use a 2D colormap based on both components
                                if len(x_velocities) > 1 and len(y_velocities) > 1:
                                    # Normalize velocities
                                    vx_norm = min(1.0, abs(x_velocities[-1]) / 50.0) * (1 if x_velocities[-1] > 0 else -1)
                                    vy_norm = min(1.0, abs(y_velocities[-1]) / 50.0) * (1 if y_velocities[-1] > 0 else -1)
                                    
                                    # Create a 2D color map (BGR format for OpenCV)
                                    # Blue channel - based on vertical direction (more for downward)
                                    b = int(128 + 127 * vy_norm) if vy_norm > 0 else 0
                                    # Green channel - based on horizontal direction (more for rightward)
                                    g = int(128 + 127 * vx_norm) if vx_norm > 0 else 0
                                    # Red channel - based on vertical direction (more for upward)
                                    r = int(128 + 127 * -vy_norm) if vy_norm < 0 else 0
                                    
                                    color = (b, g, r)
                                    cv2.line(trajectory_layer, trajectory[-2], trajectory[-1], color, 2)
                    
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
            
            # Add velocity and acceleration info based on selected direction
            y_offset = 60
            if direction == 'x' or direction == 'both':
                if len(x_velocities) > 0:
                    velocity_text = f"X-Velocity: {x_velocities[-1]:.1f} px/s"
                    velocity_color = (0, 255, 0) if x_velocities[-1] > 0 else (0, 165, 255)
                    cv2.putText(display_frame, velocity_text, (10, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, velocity_color, 2)
                    y_offset += 30
                
                if len(x_accelerations) > 0:
                    accel_text = f"X-Acceleration: {x_accelerations[-1]:.1f} px/s²"
                    accel_color = (0, 255, 0) if x_accelerations[-1] > 0 else (0, 165, 255)
                    cv2.putText(display_frame, accel_text, (10, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, accel_color, 2)
                    y_offset += 30
            
            if direction == 'y' or direction == 'both':
                if len(y_velocities) > 0:
                    velocity_text = f"Y-Velocity: {y_velocities[-1]:.1f} px/s"
                    velocity_color = (255, 0, 0) if y_velocities[-1] > 0 else (0, 0, 255)
                    cv2.putText(display_frame, velocity_text, (10, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, velocity_color, 2)
                    y_offset += 30
                
                if len(y_accelerations) > 0:
                    accel_text = f"Y-Acceleration: {y_accelerations[-1]:.1f} px/s²"
                    accel_color = (255, 0, 0) if y_accelerations[-1] > 0 else (0, 0, 255)
                    cv2.putText(display_frame, accel_text, (10, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, accel_color, 2)
                    y_offset += 30
            
            # Add legend
            legend_x, legend_y = width - 220, 30
            cv2.putText(display_frame, f"{direction.upper()}-Component Legend:", (legend_x, legend_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            legend_y += 20
            
            # Draw arrows for legend based on direction
            if direction == 'x' or direction == 'both':
                # Rightward green arrow
                arrow_start = (legend_x + 10, legend_y + 20)
                cv2.arrowedLine(display_frame, arrow_start, (arrow_start[0] + 30, arrow_start[1]), (0, 255, 0), 2)
                cv2.putText(display_frame, "Rightward (+)", (arrow_start[0] + 40, arrow_start[1] + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Leftward orange arrow
                arrow_start = (legend_x + 10, legend_y + 40)
                cv2.arrowedLine(display_frame, arrow_start, (arrow_start[0] - 30, arrow_start[1]), (0, 165, 255), 2)
                cv2.putText(display_frame, "Leftward (-)", (arrow_start[0] + 40, arrow_start[1] + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                legend_y += 40
            
            if direction == 'y' or direction == 'both':
                # Downward blue arrow
                arrow_start = (legend_x + 10, legend_y + 20)
                cv2.arrowedLine(display_frame, arrow_start, (arrow_start[0], arrow_start[1] + 30), (255, 0, 0), 2)
                cv2.putText(display_frame, "Downward (+)", (arrow_start[0] + 40, arrow_start[1] + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Upward red arrow
                arrow_start = (legend_x + 10, legend_y + 60)
                cv2.arrowedLine(display_frame, arrow_start, (arrow_start[0], arrow_start[1] - 30), (0, 0, 255), 2)
                cv2.putText(display_frame, "Upward (-)", (arrow_start[0] + 40, arrow_start[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
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
    
    # Compile dynamics data
    dynamics = {
        'trajectory': trajectory,
        'timestamps': timestamps,
        'x_positions': x_positions,
        'y_positions': y_positions,
        'x_velocities': x_velocities,
        'y_velocities': y_velocities,
        'x_accelerations': x_accelerations,
        'y_accelerations': y_accelerations
    }
    
    # Save data to CSV
    if output_dir:
        df = pd.DataFrame({
            'timestamp': timestamps,
            'x_position': x_positions,
            'y_position': y_positions,
            'x_velocity': x_velocities if direction in ['x', 'both'] else [None] * len(timestamps),
            'y_velocity': y_velocities if direction in ['y', 'both'] else [None] * len(timestamps),
            'x_acceleration': x_accelerations if direction in ['x', 'both'] else [None] * len(timestamps),
            'y_acceleration': y_accelerations if direction in ['y', 'both'] else [None] * len(timestamps)
        })
        df.to_csv(os.path.join(output_dir, f'{direction}_component_dynamics.csv'), index=False)
        print(f"Data saved to {os.path.join(output_dir, f'{direction}_component_dynamics.csv')}")
    
    # Plot and save selected component graphs
    plot_component_dynamics(dynamics, direction, output_dir)
    
    return dynamics

def plot_component_dynamics(dynamics, direction='y', output_dir=None):
    """
    Plot and save graphs for directional component velocity and acceleration.
    
    Parameters:
    -----------
    dynamics : dict
        Dictionary containing trajectory and dynamics data
    direction : str
        Direction to analyze: 'x', 'y', or 'both'
    output_dir : str
        Directory to save the plots
    """
    if not dynamics or len(dynamics.get('trajectory', [])) < 3:
        print("Insufficient dynamics data to plot")
        return
    
    timestamps = dynamics['timestamps']
    
    # Calculate relative time from start
    rel_times = [t - timestamps[0] for t in timestamps]
    
    # Make sure arrays are same length for plotting (truncate if needed)
    min_length = min(len(rel_times), 
                    len(dynamics['x_positions']), 
                    len(dynamics['y_positions']),
                    len(dynamics['x_velocities']), 
                    len(dynamics['y_velocities']),
                    len(dynamics['x_accelerations']), 
                    len(dynamics['y_accelerations']))
                    
    rel_times = rel_times[:min_length]
    
    # Plot X-component if selected
    if direction in ['x', 'both']:
        x_positions = dynamics['x_positions'][:min_length]
        x_velocities = dynamics['x_velocities'][:min_length]
        x_accelerations = dynamics['x_accelerations'][:min_length]
        
        # 1. X-Position vs Time
        plt.figure(figsize=(10, 6))
        plt.plot(rel_times, x_positions, 'g-', linewidth=2)
        plt.title('X-Position vs Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('X-Position (pixels)')
        plt.grid(True, alpha=0.3)
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'x_position_vs_time.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. X-Velocity vs Time
        plt.figure(figsize=(10, 6))
        plt.plot(rel_times, x_velocities, 'g-', linewidth=2)
        
        # Add zero line for reference
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Add colored regions for rightward/leftward motion
        plt.fill_between(rel_times, x_velocities, 0, where=(np.array(x_velocities) > 0), 
                        color='green', alpha=0.2, label='Rightward motion')
        plt.fill_between(rel_times, x_velocities, 0, where=(np.array(x_velocities) < 0), 
                        color='orange', alpha=0.2, label='Leftward motion')
        
        plt.title('X-Velocity vs Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('X-Velocity (pixels/s)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'x_velocity_vs_time.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. X-Acceleration vs Time
        plt.figure(figsize=(10, 6))
        plt.plot(rel_times, x_accelerations, 'g-', linewidth=2)
        
        # Add zero line for reference
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Add colored regions for acceleration/deceleration
        plt.fill_between(rel_times, x_accelerations, 0, where=(np.array(x_accelerations) > 0), 
                        color='lightgreen', alpha=0.2, label='Acceleration rightward')
        plt.fill_between(rel_times, x_accelerations, 0, where=(np.array(x_accelerations) < 0), 
                        color='darkorange', alpha=0.2, label='Acceleration leftward')
        
        plt.title('X-Acceleration vs Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('X-Acceleration (pixels/s²)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'x_acceleration_vs_time.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Plot Y-component if selected
    if direction in ['y', 'both']:
        y_positions = dynamics['y_positions'][:min_length]
        y_velocities = dynamics['y_velocities'][:min_length]
        y_accelerations = dynamics['y_accelerations'][:min_length]
        
        # 1. Y-Position vs Time
        plt.figure(figsize=(10, 6))
        plt.plot(rel_times, y_positions, 'b-', linewidth=2)
        plt.title('Y-Position vs Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Y-Position (pixels)')
        plt.grid(True, alpha=0.3)
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'y_position_vs_time.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Y-Velocity vs Time
        plt.figure(figsize=(10, 6))
        plt.plot(rel_times, y_velocities, 'b-', linewidth=2)
        
        # Add zero line for reference
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Add colored regions for upward/downward motion
        plt.fill_between(rel_times, y_velocities, 0, where=(np.array(y_velocities) > 0), 
                        color='blue', alpha=0.2, label='Downward motion')
        plt.fill_between(rel_times, y_velocities, 0, where=(np.array(y_velocities) < 0), 
                        color='red', alpha=0.2, label='Upward motion')
        
        plt.title('Y-Velocity vs Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Y-Velocity (pixels/s)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'y_velocity_vs_time.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Y-Acceleration vs Time
        plt.figure(figsize=(10, 6))
        plt.plot(rel_times, y_accelerations, 'b-', linewidth=2)
        
        # Add zero line for reference
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Add colored regions for acceleration/deceleration
        plt.fill_between(rel_times, y_accelerations, 0, where=(np.array(y_accelerations) > 0), 
                        color='lightblue', alpha=0.2, label='Acceleration downward')
        plt.fill_between(rel_times, y_accelerations, 0, where=(np.array(y_accelerations) < 0), 
                        color='salmon', alpha=0.2, label='Acceleration upward')
        
        plt.title('Y-Acceleration vs Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Y-Acceleration (pixels/s²)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'y_acceleration_vs_time.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Combined plot if analyzing both directions
    if direction == 'both':
        # 1. Combined Position Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(dynamics['x_positions'][:min_length], dynamics['y_positions'][:min_length], 
                   c=rel_times, cmap='viridis', s=30, alpha=0.7)
        plt.colorbar(label='Time (seconds)')
        plt.title('Nanosphere 2D Trajectory (colored by time)')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.grid(True, alpha=0.3)
        if output_dir:
            plt.savefig(os.path.join(output_dir, '2d_trajectory.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # 1.1 Create figure with speed mapping at xdirection

        x = dynamics['x_positions'][:min_length]
        y = dynamics['y_positions'][:min_length]
        plt.figure(figsize=(10, 8))
        
        # 1. Trajectory plot with speed coloring
         
        # Create a colormap based on speed
        
        # Create scalar speed values for coloring
        speed_values = -np.array(dynamics['x_velocities'])  # Skip first (it's 0)
        
        # Normalize speed values for colormap
        norm = plt.Normalize(vmin=speed_values.min(), vmax=speed_values.max())
        # Plot trajectory with speed coloring
        plt.scatter(x, y, c=speed_values, cmap=plt.cm.jet, s=30, alpha=0.7)

    
        # Mark start and end points
        plt.scatter(x[0], y[0], color='green', s=100, label='Start')
        plt.scatter(x[-1], y[-1], color='red', s=100, label='End')
        
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Speed (pixels/s)')
        
        plt.title('Nanosphere Trajectory (colored by speed)')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        # ax1.set_title('Nanosphere Trajectory (colored by speed)')
        # ax1.set_xlabel('X Position (pixels)')
        # ax1.set_ylabel('Y Position (pixels)')
        #ax1.grid(True, alpha=0.3)
        plt.ylim(0,800)
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        plt.legend()
        
        # Save trajectory plot if requested
        if output_dir:
            plt.savefig(f"{output_dir}/trajectory_plot_speed_mapping.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Velocity Vector Plot
        plt.figure(figsize=(10, 8))
        plt.quiver(dynamics['x_positions'][:min_length], dynamics['y_positions'][:min_length],
                  dynamics['x_velocities'][:min_length], dynamics['y_velocities'][:min_length],
                  np.sqrt(np.array(dynamics['x_velocities'][:min_length])**2 + 
                         np.array(dynamics['y_velocities'][:min_length])**2),
                  cmap='jet', scale=500, width=0.005)
        plt.colorbar(label='Velocity Magnitude (pixels/s)')
        plt.title('Nanosphere Velocity Vectors')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.grid(True, alpha=0.3)
        plt.ylim(0,800)
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'velocity_vectors.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Combined X and Y Components Dashboard
        plt.figure(figsize=(15, 10))
        
        # Position subplot
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(rel_times, dynamics['x_positions'][:min_length], 'g-', linewidth=2)
        ax1.set_title('X-Position vs Time')
        ax1.set_ylabel('X-Position (pixels)')
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(rel_times, dynamics['y_positions'][:min_length], 'b-', linewidth=2)
        ax2.set_title('Y-Position vs Time')
        ax2.set_ylabel('Y-Position (pixels)')
        ax2.grid(True, alpha=0.3)
        
        # Velocity subplot
        ax3 = plt.subplot(3, 2, 3, sharex=ax1)
        ax3.plot(rel_times, dynamics['x_velocities'][:min_length], 'g-', linewidth=2)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_title('X-Velocity vs Time')
        ax3.set_ylabel('X-Velocity (pixels/s)')
        ax3.grid(True, alpha=0.3)
        
        ax4 = plt.subplot(3, 2, 4, sharex=ax2)
        ax4.plot(rel_times, dynamics['y_velocities'][:min_length], 'b-', linewidth=2)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_title('Y-Velocity vs Time')
        ax4.set_ylabel('Y-Velocity (pixels/s)')
        ax4.grid(True, alpha=0.3)
        
        # Acceleration subplot
        ax5 = plt.subplot(3, 2, 5, sharex=ax1)
        ax5.plot(rel_times, dynamics['x_accelerations'][:min_length], 'g-', linewidth=2)
        ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax5.set_title('X-Acceleration vs Time')
        ax5.set_xlabel('Time (seconds)')
        ax5.set_ylabel('X-Acceleration (pixels/s²)')
        ax5.grid(True, alpha=0.3)
        
        ax6 = plt.subplot(3, 2, 6, sharex=ax2)
        ax6.plot(rel_times, dynamics['y_accelerations'][:min_length], 'b-', linewidth=2)
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax6.set_title('Y-Acceleration vs Time')
        ax6.set_xlabel('Time (seconds)')
        ax6.set_ylabel('Y-Acceleration (pixels/s²)')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'combined_xy_dynamics.png'), dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    video_path = '/Users/lianbin/Library/CloudStorage/Box-Box/Tough_hydrogel_using_ppy_and_pss/Demo_of_sensing/bending mapping speed/short_version/accelerate_/accelerate_-1.mp4'
    dynamics = analyze_nanosphere_component(
        video_path, 
        direction='x',
        display_results=False,
        save_output=True,
        binary_value=130
    )