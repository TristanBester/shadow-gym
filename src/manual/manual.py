import time

import mujoco
import mujoco.viewer
import numpy as np

# Target site positions (goal positions for fingertips)
TARGET_POSITIONS = {
    "target0": np.array([0.966, 0.794, 0.153]),  # First finger
    "target1": np.array([0.988, 0.790, 0.153]),  # Middle finger
    "target2": np.array([1.010, 0.794, 0.153]),  # Ring finger
    "target3": np.array([1.033, 0.801, 0.154]),  # Little finger
    "target4": np.array([0.922, 0.886, 0.167]),  # Thumb
}


def set_target_positions(model, data):
    """Set target site positions to the configured TARGET_POSITIONS."""
    # Calculate offset between global and local positions
    site_offset = data.site_xpos - model.site_pos

    for target_name, target_pos in TARGET_POSITIONS.items():
        try:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, target_name)
            # Convert global target position to local position
            goal_position_local = target_pos - site_offset[site_id]
            model.site_pos[site_id] = goal_position_local
        except Exception as e:
            print(f"Warning: Could not set position for {target_name}: {e}")


def main():
    # Load model directly
    model_path = "/Users/tristan/Projects/shadow-gym/assets/hand/reach.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Reset to initial state
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Launch passive viewer
    viewer = mujoco.viewer.launch_passive(model, data)

    # Set camera to the position from the logger
    viewer.cam.lookat[0] = 0.952
    viewer.cam.lookat[1] = 0.948
    viewer.cam.lookat[2] = 0.243
    viewer.cam.distance = 0.619
    viewer.cam.azimuth = 15.3
    viewer.cam.elevation = -13.5

    last_log_time = time.time()
    log_interval = 2.0  # Log every 2 seconds

    # Store control ranges for normalization
    ctrl_range = model.actuator_ctrlrange.copy()
    half_ranges = (ctrl_range[:, 1] - ctrl_range[:, 0]) / 2.0
    centers = (ctrl_range[:, 1] + ctrl_range[:, 0]) / 2.0

    # Get fingertip site IDs
    fingertip_sites = [
        "robot0:S_fftip",
        "robot0:S_mftip",
        "robot0:S_rftip",
        "robot0:S_lftip",
        "robot0:S_thtip",
    ]
    fingertip_site_ids = []
    for site_name in fingertip_sites:
        try:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            fingertip_site_ids.append(site_id)
        except Exception as e:
            print(f"Warning: Could not find site {site_name}: {e}")
            fingertip_site_ids.append(None)

    try:
        while viewer.is_running():
            # Update target site positions (like the renderer does every frame)
            set_target_positions(model, data)

            # Step simulation with current control values
            mujoco.mj_step(model, data)

            # Log normalized actions periodically
            current_time = time.time()
            if current_time - last_log_time >= log_interval:
                # Convert current ctrl values to normalized actions [-1, 1]
                # This reverses the mapping: action = (ctrl - center) / half_range
                normalized_actions = (data.ctrl[:] - centers) / half_ranges
                normalized_actions = normalized_actions.clip(-1.0, 1.0)

                # Print the action array
                print("\nNormalized Actions [-1, 1]:")
                print("  np.array([", end="")
                for i in range(20):
                    if i > 0:
                        print(", ", end="")
                    if i > 0 and i % 5 == 0:
                        print("\n            ", end="")
                    print(f"{normalized_actions[i]:+.3f}", end="")
                print("])")

                # Print fingertip positions
                print("\nFingertip Positions [x, y, z]:")
                for site_name, site_id in zip(fingertip_sites, fingertip_site_ids):
                    if site_id is not None:
                        pos = data.site_xpos[site_id]
                        print(
                            f"  {site_name:18s}: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]"
                        )
                    else:
                        print(f"  {site_name:18s}: [NOT FOUND]")

                print("-" * 80)

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
