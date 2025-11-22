import math
import time

import mujoco
import mujoco.viewer


def main():
    # Load model directly
    model_path = "/Users/tristan/Projects/shadow-gym/assets/hand/reach.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Launch passive viewer
    viewer = mujoco.viewer.launch_passive(model, data)

    # Set initial camera position to see the robot (hand is at ~1, 1.25, 0.15)
    viewer.cam.lookat[0] = 1.0
    viewer.cam.lookat[1] = 1.25
    viewer.cam.lookat[2] = 0.3
    viewer.cam.distance = 0.8
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20

    last_log_time = time.time()
    log_interval = 1.0  # Log every 1 second

    print("Camera logging started...")
    print("-" * 80)
    print()

    try:
        while viewer.is_running():
            # Step simulation
            mujoco.mj_step(model, data)

            # Log camera position periodically
            current_time = time.time()
            if current_time - last_log_time >= log_interval:
                cam = viewer.cam

                # Calculate camera position and orientation from lookat, distance, and angles
                az_rad = math.radians(cam.azimuth)
                el_rad = math.radians(cam.elevation)

                # Camera position calculation
                cam_x = cam.lookat[0] + cam.distance * math.cos(el_rad) * math.sin(
                    az_rad
                )
                cam_y = cam.lookat[1] - cam.distance * math.cos(el_rad) * math.cos(
                    az_rad
                )
                cam_z = cam.lookat[2] + cam.distance * math.sin(el_rad)

                # Calculate camera orientation (xyaxes)
                # Forward vector (from camera to lookat)
                forward_x = cam.lookat[0] - cam_x
                forward_y = cam.lookat[1] - cam_y
                forward_z = cam.lookat[2] - cam_z
                forward_len = math.sqrt(forward_x**2 + forward_y**2 + forward_z**2)
                forward_x /= forward_len
                forward_y /= forward_len
                forward_z /= forward_len

                # Right vector (perpendicular to forward and world up)
                world_up_x, world_up_y, world_up_z = 0, 0, 1
                right_x = forward_y * world_up_z - forward_z * world_up_y
                right_y = forward_z * world_up_x - forward_x * world_up_z
                right_z = forward_x * world_up_y - forward_y * world_up_x
                right_len = math.sqrt(right_x**2 + right_y**2 + right_z**2)
                if right_len > 0:
                    right_x /= right_len
                    right_y /= right_len
                    right_z /= right_len

                # Up vector (perpendicular to forward and right)
                up_x = forward_y * right_z - forward_z * right_y
                up_y = forward_z * right_x - forward_x * right_z
                up_z = forward_x * right_y - forward_y * right_x

                # Print camera info
                print(f"Camera Settings:")
                print(
                    f"  Lookat: [{cam.lookat[0]:.3f}, {cam.lookat[1]:.3f}, {cam.lookat[2]:.3f}]"
                )
                print(f"  Distance: {cam.distance:.3f}")
                print(f"  Azimuth: {cam.azimuth:.1f}°")
                print(f"  Elevation: {cam.elevation:.1f}°")
                print()
                print(f"Calculated Camera Position:")
                print(f'  pos="{cam_x:.3f} {cam_y:.3f} {cam_z:.3f}"')
                print()
                print(f"Copy this EXACT line to reach.xml:")
                print(f'  <camera name="default_view" mode="fixed"')
                print(f'          pos="{cam_x:.3f} {cam_y:.3f} {cam_z:.3f}"')
                print(
                    f'          xyaxes="{right_x:.4f} {right_y:.4f} {right_z:.4f} {up_x:.4f} {up_y:.4f} {up_z:.4f}" />'
                )
                print("-" * 80)
                print()

                last_log_time = current_time

            # Sync viewer
            viewer.sync()

    except KeyboardInterrupt:
        print("\n\nStopping viewer...")
    finally:
        viewer.close()
        print("\nViewer closed.")


if __name__ == "__main__":
    main()
