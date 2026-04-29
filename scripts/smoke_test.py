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

    print("[INFO] Scene built. Running idle loop — close window or Ctrl-C to exit.")
    while simulation_app.is_running():
        env.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
