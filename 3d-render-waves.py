import numpy as np
from mayavi import mlab
from tvtk.api import tvtk
from pyface.api import GUI
import os

# Parameters
radius = 1
c = 1
duration =300
frames = 800
scaling_factor = 0.25 

# Original wave function
def wave(theta, phi, t):
    # Source point (e.g., North Pole)
    source_theta = 0
    source_phi = 0

    # Calculate the spherical distance from the source point
    cos_angle = np.cos(source_theta) * np.cos(theta) + np.sin(source_theta) * np.sin(theta) * np.cos(phi - source_phi)
    angle = np.arccos(np.clip(cos_angle, -1, 1))  # Spherical distance

    # Wave parameters
    wave_speed = 0.5  # Speed of the wave
    frequency = 3   # Frequency of the wave
    decay_rate = 0.0005  # Decay rate of the wave amplitude

    # Distance the wave has traveled from the source point
    wave_front = wave_speed * t

    # Sinusoidal wave with Gaussian decay
    return np.sin(frequency * (angle - wave_front)) * np.exp(-decay_rate * (angle - wave_front)**2)

def wave2(theta, phi, t, wave_params):
    start_time, lifespan, reference_theta, reference_phi = wave_params['start_time'], wave_params['lifespan'], wave_params['reference_theta'], wave_params['reference_phi']
    if t < start_time or t > start_time + lifespan:
        return 0  # No wave before the start time or after the end of its lifespan

    # Adjust time relative to the start of the wave
    adjusted_time = t - start_time

    # Increase the base amplitude and adjust decay rate
    base_amplitude = 1.5  # Increase this value for larger amplitude
    decay_rate = 0.2  # Decrease this value for slower decay

    # Calculate the spherical distance from the reference point
    cos_angle = np.cos(reference_theta) * np.cos(theta) + np.sin(reference_theta) * np.sin(theta) * np.cos(phi - reference_phi)
    angle = np.arccos(np.clip(cos_angle, -1, 1))  # Clipping for numerical stability

    # Propagate the wave by changing the reference point over time
    moving_theta = (np.pi / duration) * adjusted_time * 4  # This moves from 0 to pi over the duration

    # Adjust the wave's amplitude based on the spherical distance
    return base_amplitude * np.exp(-((angle - moving_theta) ** 2) / (2 * decay_rate ** 2))

num_additional_waves = 20 
interval_length = (duration / num_additional_waves)*1.1

additional_waves = []
for i in range(num_additional_waves):
    start_time = i * interval_length + np.random.uniform(0, interval_length)
    lifespan = np.random.uniform(200, 300)  # Or any other range you prefer
    reference_theta = np.random.uniform(0, np.pi)
    reference_phi = np.random.uniform(0, 2 * np.pi)

    additional_waves.append({
        'start_time': start_time,
        'lifespan': lifespan,
        'reference_theta': reference_theta,
        'reference_phi': reference_phi
    })

# Pre-calculate the base coordinates for latitude/longitude lines on a unit sphere
def precalculate_lat_long_lines(num_lines=20):
    lat_base_lines = []
    long_base_lines = []
    for i in range(num_lines):
        # Latitude
        theta = np.linspace(0, np.pi, 100)
        phi = np.ones(100) * (2 * np.pi * i / num_lines)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        lat_base_lines.append((x, y, z))

        # Longitude
        phi = np.linspace(0, 2 * np.pi, 100)
        theta = np.ones(100) * (np.pi * i / num_lines)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        long_base_lines.append((x, y, z))

    return lat_base_lines, long_base_lines


# Create a Mayavi figure
fig = mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))

# Generate base lines
lat_base_lines, long_base_lines = precalculate_lat_long_lines()

# Plot lines
line_objects = []
for line in lat_base_lines + long_base_lines:
    line_obj = mlab.plot3d(line[0], line[1], line[2], color=(0, 0, 0), tube_radius=None, figure=fig)
    line_objects.append(line_obj)

# Create a mesh grid for the sphere
theta, phi = np.linspace(0, np.pi, 100), np.linspace(0, 2 * np.pi, 100)
theta, phi = np.meshgrid(theta, phi)
x = radius * np.sin(theta) * np.cos(phi)
y = radius * np.sin(theta) * np.sin(phi)
z = radius * np.cos(theta)


# Increase ambient light if needed
#fig.scene.light_manager.ambient_light = (0.5, 0.5, 0.5)  # Adjust RGB values as needed

# Initial plot
mesh = mlab.mesh(x, y, z, color=(0, 0.7, 0.7), figure=fig)

# Adjust material properties
mesh.actor.property.specular = 0.1  # Adjust the shininess
mesh.actor.property.diffuse = 0.1   # Adjust the diffuseness
mesh.actor.property.ambient = 0.9   # Adjust the ambient light


def update_mesh(frame):
    t = frame * duration / frames

    # Calculate the amplitude of the first wave
    wave1_amplitude = wave(theta, phi, t)

    # Initialize total wave amplitude with the first wave
    total_wave_amplitude = wave1_amplitude

    # Add the amplitude of each additional wave
    for wave_params in additional_waves:
        if t >= wave_params['start_time'] and t <= wave_params['start_time'] + wave_params['lifespan']:
            additional_wave_amplitude = wave2(theta, phi, t, wave_params)
            total_wave_amplitude += additional_wave_amplitude

    r = radius + scaling_factor * total_wave_amplitude
    x_new = r * np.sin(theta) * np.cos(phi)
    y_new = r * np.sin(theta) * np.sin(phi)
    z_new = r * np.cos(theta)

    mesh.mlab_source.set(x=x_new, y=y_new, z=z_new)

    # Remove existing line objects
    for line_obj in line_objects:
        line_obj.remove()

    # Recreate latitude/longitude lines with updated positions
    line_objects.clear()
    tube_radius = 0.005  # Adjust this value to change the thickness of the lines
    for base_line in lat_base_lines + long_base_lines:
        scaled_line = []
        for j in range(len(base_line[0])):
            theta_val = np.arccos(base_line[2][j])
            phi_val = np.arctan2(base_line[1][j], base_line[0][j])
            wave_amplitude = wave(theta_val, phi_val, t) + sum(wave2(theta_val, phi_val, t, wp) for wp in additional_waves)
            r_scaled = radius + scaling_factor * wave_amplitude
            scaled_line.append((r_scaled * np.sin(theta_val) * np.cos(phi_val), r_scaled * np.sin(theta_val) * np.sin(phi_val), r_scaled * np.cos(theta_val)))

        x_scaled, y_scaled, z_scaled = zip(*scaled_line)
        line_obj = mlab.plot3d(x_scaled, y_scaled, z_scaled, color=(0, 0, 0), tube_radius=tube_radius, figure=fig)
        line_objects.append(line_obj)


# Directory to save frames
frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)


# Animation function
@mlab.animate(delay=10)
def anim():
    for i in range(frames):
        update_mesh(i)
        filename = os.path.join(frames_dir, f"frame_{i:04d}.png")
        mlab.savefig(filename)  # Save each frame
        yield

# Start the animation
a = anim()

# Start the event loop
GUI().start_event_loop()
mlab.show()

# After the animation completes, create a video using 
# ffmpeg -r 30 -f image2 -s 1600x1512 -i frames/frame_%04d.png -vcodec libx264 -crf 5  -pix_fmt yuv420p output.mp4

