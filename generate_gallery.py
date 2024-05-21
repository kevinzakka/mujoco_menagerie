
import math
from pathlib import Path
from dm_control import mjcf
from PIL import Image
import cv2
import enum
import numpy as np
from tqdm.auto import tqdm
from mdutils.mdutils import MdUtils

class ModelType(int, enum.Enum):
  ARM = 0
  DUAL_ARM = 1
  END_EFFECTOR = 2
  MOBILE_MANIPULATOR = 3
  QUADRUPED = 4
  BIPED = 5
  HUMANOID = 6
  DRONE = 7
  BIOMECHANICAL = 8
  MISC = 9

def write_bottom_left_corner(img, text):
  pass


NAME_MAP = {
  "franka_emika_panda/panda.xml": "panda",
  "franka_emika_panda/hand.xml": "panda gripper",
  "ufactory_lite6/lite6.xml": "lite6",
  "flybody/fruitfly.xml": "fruitfly",
  "wonik_allegro/left_hand.xml": "left allegro",
  "wonik_allegro/right_hand.xml": "right allegro",
  "shadow_hand/left_hand.xml": "left shadow",
  "shadow_hand/right_hand.xml": "right shadow",
  "skydio_x2/x2.xml": "skydio x2",
  "unitree_h1/h1.xml": "h1",
  "bitcraze_crazyflie_2/cf2.xml": "crazyflie 2",
  "google_robot/robot.xml": "google robot",
  "unitree_a1/a1.xml": "a1",
  "google_barkour_v0/barkour_v0.xml": "barkour v0",
  "anybotics_anymal_b/anymal_b.xml": "anymal b",
  "unitree_go1/go1.xml": "go1",
  "unitree_z1/z1.xml": "z1",
  "anybotics_anymal_c/anymal_c.xml": "anymal c",
  "agility_cassie/cassie.xml": "cassie",
  "realsense_d435i/d435i.xml": "d435i",
  "universal_robots_ur5e/ur5e.xml": "ur5e",
  "aloha/aloha.xml": "aloha 2",
  "rethink_robotics_sawyer/sawyer.xml": "sawyer",
  "robotis_op3/op3.xml": "op3",
  "universal_robots_ur10e/ur10e.xml": "ur10e",
  "kuka_iiwa_14/iiwa14.xml": "iiwa 14",
  "trossen_vx300s/vx300s.xml": "vx300s",
  "unitree_g1/g1.xml": "g1",
  "robotiq_2f85/2f85.xml": "2f85",
  "ufactory_xarm7/hand.xml": "xarm7 gripper",
  "ufactory_xarm7/xarm7.xml": "xarm7",
  "hello_robot_stretch/stretch.xml": "stretch 2",
  "google_barkour_vb/barkour_vb.xml": "barkour vb",
  "unitree_go2/go2.xml": "go2",
}

MODEL_MAP = {
  "franka_emika_panda/panda.xml": ModelType.ARM,
  "franka_emika_panda/hand.xml": ModelType.END_EFFECTOR,
  "ufactory_lite6/lite6.xml": ModelType.ARM,
  "flybody/fruitfly.xml": ModelType.BIOMECHANICAL,
  "wonik_allegro/left_hand.xml": ModelType.END_EFFECTOR,
  "wonik_allegro/right_hand.xml": ModelType.END_EFFECTOR,
  "shadow_hand/left_hand.xml": ModelType.END_EFFECTOR,
  "shadow_hand/right_hand.xml": ModelType.END_EFFECTOR,
  "skydio_x2/x2.xml": ModelType.DRONE,
  "unitree_h1/h1.xml": ModelType.HUMANOID,
  "bitcraze_crazyflie_2/cf2.xml": ModelType.DRONE,
  "google_robot/robot.xml": ModelType.MOBILE_MANIPULATOR,
  "unitree_a1/a1.xml": ModelType.QUADRUPED,
  "google_barkour_v0/barkour_v0.xml": ModelType.QUADRUPED,
  "anybotics_anymal_b/anymal_b.xml": ModelType.QUADRUPED,
  "unitree_go1/go1.xml": ModelType.QUADRUPED,
  "unitree_z1/z1.xml": ModelType.ARM,
  "anybotics_anymal_c/anymal_c.xml": ModelType.QUADRUPED,
  "agility_cassie/cassie.xml": ModelType.BIPED,
  "realsense_d435i/d435i.xml": ModelType.MISC,
  "universal_robots_ur5e/ur5e.xml": ModelType.ARM,
  "aloha/aloha.xml": ModelType.DUAL_ARM,
  "rethink_robotics_sawyer/sawyer.xml": ModelType.ARM,
  "robotis_op3/op3.xml": ModelType.HUMANOID,
  "universal_robots_ur10e/ur10e.xml": ModelType.ARM,
  "kuka_iiwa_14/iiwa14.xml": ModelType.ARM,
  "trossen_vx300s/vx300s.xml": ModelType.ARM,
  "unitree_g1/g1.xml": ModelType.HUMANOID,
  "robotiq_2f85/2f85.xml": ModelType.END_EFFECTOR,
  "ufactory_xarm7/hand.xml": ModelType.END_EFFECTOR,
  "ufactory_xarm7/xarm7.xml": ModelType.ARM,
  "hello_robot_stretch/stretch.xml": ModelType.MOBILE_MANIPULATOR,
  "google_barkour_vb/barkour_vb.xml": ModelType.QUADRUPED,
  "unitree_go2/go2.xml": ModelType.QUADRUPED,
}

# ============================================================================ #

# Grab XML files from subdirectories.
ROOT_DIR = Path(__file__).parent
(ROOT_DIR / "assets").mkdir(parents=True, exist_ok=True)
MODEL_DIRS = [f for f in ROOT_DIR.iterdir() if f.is_dir()]
MODEL_XMLS = []
def _get_xmls(pattern: str):
  for d in MODEL_DIRS:
    yield from d.glob(pattern)
MODEL_XMLS = list(_get_xmls('*.xml'))

# Filter out unwanted XML files.
filter_words = ["scene", "keyframe", "mjx", "actuator", "nohand"]
for word in filter_words:
  MODEL_XMLS = [f for f in MODEL_XMLS if word not in f.stem]

# Sort XML files.
def sort_func(xml):
  name = f"{xml.parent.stem}/{xml.name}"
  return (MODEL_MAP[name], name)
MODEL_XMLS = sorted(MODEL_XMLS, key=sort_func)

paths = []
for xml in tqdm(MODEL_XMLS):
  try:
    model_xml = mjcf.from_path(xml.as_posix(), escape_separators=True)
    for light in model_xml.find_all("light"):
      light.remove()

    arena = mjcf.RootElement()
    getattr(arena.visual, "global").offheight = 720
    getattr(arena.visual, "global").offwidth = 1280
    getattr(arena.visual, "global").azimuth = 140
    getattr(arena.visual, "global").elevation = -20
    arena.visual.quality.shadowsize = 8192
    arena.asset.add(
      "texture",
      type="skybox",
      builtin="gradient",
      height=512,
      width=512,
      rgb1="1 1 1",
      rgb2="1 1 1"
    )
    arena.include_copy(model_xml, override_attributes=True)

    arena.worldbody.add("light", pos="0 0 2", directional="true")

    physics = mjcf.Physics.from_mjcf_model(arena)

    try:
      physics.reset(keyframe_id=0)
    except:
      physics.reset()
    physics.forward()

    img = physics.render(height=500, width=500)
    title = f"{xml.parent.stem}/{xml.name}"
    img = cv2.putText(img.copy(), NAME_MAP[title], (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1, cv2.LINE_AA)

    maker = xml.parent.stem
    robot = xml.stem
    filename = f"assets/{maker}-{robot}.png"
    Image.fromarray(img).save(filename)
    paths.append(filename)
  except:
    print(f"failed to load {xml.as_posix()}")

# Create markdown table.
N_MODELS = len(paths)
N_COLS = 5
N_ROWS = int(math.ceil(N_MODELS / N_COLS))
table = []
for r in range(N_ROWS):
  row = []
  for c in range(N_COLS):
    i = r * N_COLS + c
    if i >= N_MODELS:
      row.append("")
    else:
      row.append(f"<img src='{paths[i]}' width=100>")
  table.extend(row)
mdFile = MdUtils(file_name='gallery')
mdFile.new_table(columns=N_COLS, rows=N_ROWS, text=table, text_align="center")
mdFile.create_md_file()
