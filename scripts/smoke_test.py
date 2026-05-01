"""
Smoke test — just brings up the scene, lets it idle. Useful for verifying
Isaac Sim install and asset paths before running the full pipeline.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["CARB_LOG_LEVEL"] = "error"


def main():
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

    from src.env import DualArmEnv
    env = DualArmEnv(config_path="configs/default.yaml", arms=("left", "right"))

    try:
        import numpy as np
        from isaacsim.core.utils.viewports import set_camera_view
        from omni.kit.viewport.utility import get_active_viewport, frame_viewport_prims
        viewport = get_active_viewport()
        if viewport is not None:
            set_camera_view(
                eye=np.array([0.0, -1.45, 1.55]),
                target=np.array([0.0, 0.42, 0.78]),
                camera_prim_path="/OmniverseKit_Persp",
                viewport_api=viewport,
            )
            frame_viewport_prims(viewport, ["/World/Table/top", "/World/Franka_left", "/World/Franka_right"])
    except Exception as exc:
        print(f"[WARN] Could not configure viewport camera: {exc}")

    print("[INFO] Scene built. Running idle loop — close window or Ctrl-C to exit.")
    while simulation_app.is_running():
        env.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
