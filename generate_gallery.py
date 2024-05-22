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


NAME_MAP = {
    "franka_emika_panda/panda": "panda",
    "franka_emika_panda/hand": "panda gripper",
    "ufactory_lite6/lite6": "lite6",
    "flybody/fruitfly": "fruitfly",
    "wonik_allegro/left_hand": "left allegro",
    "wonik_allegro/right_hand": "right allegro",
    "shadow_hand/left_hand": "left shadow",
    "shadow_hand/right_hand": "right shadow",
    "skydio_x2/x2": "skydio x2",
    "unitree_h1/h1": "h1",
    "bitcraze_crazyflie_2/cf2": "crazyflie 2",
    "google_robot/robot": "google robot",
    "unitree_a1/a1": "a1",
    "google_barkour_v0/barkour_v0": "barkour v0",
    "anybotics_anymal_b/anymal_b": "anymal b",
    "unitree_go1/go1": "go1",
    "unitree_z1/z1": "z1",
    "anybotics_anymal_c/anymal_c": "anymal c",
    "agility_cassie/cassie": "cassie",
    "realsense_d435i/d435i": "d435i",
    "universal_robots_ur5e/ur5e": "ur5e",
    "aloha/aloha": "aloha 2",
    "rethink_robotics_sawyer/sawyer": "sawyer",
    "robotis_op3/op3": "op3",
    "universal_robots_ur10e/ur10e": "ur10e",
    "kuka_iiwa_14/iiwa14": "iiwa 14",
    "trossen_vx300s/vx300s": "vx300s",
    "unitree_g1/g1": "g1",
    "robotiq_2f85/2f85": "2f85",
    "ufactory_xarm7/hand": "xarm7 gripper",
    "ufactory_xarm7/xarm7": "xarm7",
    "hello_robot_stretch/stretch": "stretch 2",
    "google_barkour_vb/barkour_vb": "barkour vb",
    "unitree_go2/go2": "go2",
}

MODEL_MAP = {
    "franka_emika_panda/panda": ModelType.ARM,
    "franka_emika_panda/hand": ModelType.END_EFFECTOR,
    "ufactory_lite6/lite6": ModelType.ARM,
    "flybody/fruitfly": ModelType.BIOMECHANICAL,
    "wonik_allegro/left_hand": ModelType.END_EFFECTOR,
    "wonik_allegro/right_hand": ModelType.END_EFFECTOR,
    "shadow_hand/left_hand": ModelType.END_EFFECTOR,
    "shadow_hand/right_hand": ModelType.END_EFFECTOR,
    "skydio_x2/x2": ModelType.DRONE,
    "unitree_h1/h1": ModelType.HUMANOID,
    "bitcraze_crazyflie_2/cf2": ModelType.DRONE,
    "google_robot/robot": ModelType.MOBILE_MANIPULATOR,
    "unitree_a1/a1": ModelType.QUADRUPED,
    "google_barkour_v0/barkour_v0": ModelType.QUADRUPED,
    "anybotics_anymal_b/anymal_b": ModelType.QUADRUPED,
    "unitree_go1/go1": ModelType.QUADRUPED,
    "unitree_z1/z1": ModelType.ARM,
    "anybotics_anymal_c/anymal_c": ModelType.QUADRUPED,
    "agility_cassie/cassie": ModelType.BIPED,
    "realsense_d435i/d435i": ModelType.MISC,
    "universal_robots_ur5e/ur5e": ModelType.ARM,
    "aloha/aloha": ModelType.DUAL_ARM,
    "rethink_robotics_sawyer/sawyer": ModelType.ARM,
    "robotis_op3/op3": ModelType.HUMANOID,
    "universal_robots_ur10e/ur10e": ModelType.ARM,
    "kuka_iiwa_14/iiwa14": ModelType.ARM,
    "trossen_vx300s/vx300s": ModelType.ARM,
    "unitree_g1/g1": ModelType.HUMANOID,
    "robotiq_2f85/2f85": ModelType.END_EFFECTOR,
    "ufactory_xarm7/hand": ModelType.END_EFFECTOR,
    "ufactory_xarm7/xarm7": ModelType.ARM,
    "hello_robot_stretch/stretch": ModelType.MOBILE_MANIPULATOR,
    "google_barkour_vb/barkour_vb": ModelType.QUADRUPED,
    "unitree_go2/go2": ModelType.QUADRUPED,
}

CAMERA_MAP = {
    "skydio_x2/x2": dict(
        pos="-0.580 -0.260 0.622",
        xyaxes="0.442 -0.897 -0.000 0.428 0.211 0.879",
        fovy=60,
    ),
    "flybody/fruitfly": dict(
        pos="0.430 -0.361 0.326", xyaxes="0.589 0.808 0.000 -0.486 0.354 0.799", fovy=50
    ),
    "wonik_allegro/right_hand": dict(
        pos="0.002 -0.044 0.432", xyaxes="0.039 -0.999 0.000 0.999 0.039 0.017"
    ),
    "wonik_allegro/left_hand": dict(
        pos="0.002 0.043 0.432", xyaxes="0.052 -0.999 0.000 0.998 0.052 0.017"
    ),
    "ufactory_xarm7/xarm7": dict(
        pos="0.852 -0.383 0.860", xyaxes="0.487 0.874 0.000 -0.354 0.197 0.914", fovy=50
    ),
    "ufactory_xarm7/hand": dict(
        pos="-0.282 0.013 0.118",
        xyaxes="-0.047 -0.999 0.000 0.160 -0.007 0.987",
        fovy=50,
    ),
    "shadow_hand/left_hand": dict(
        pos="0.304 0.076 0.570",
        xyaxes="-0.599 0.801 0.000 -0.785 -0.587 0.195",
        fovy=50,
    ),
    "shadow_hand/right_hand": dict(
        pos="0.304 0.076 0.570",
        xyaxes="-0.599 0.801 0.000 -0.785 -0.587 0.195",
        fovy=50,
    ),
    "franka_emika_panda/panda": dict(
        pos="0.412 1.106 0.849",
        xyaxes="-0.994 0.108 0.000 -0.040 -0.369 0.928",
        fovy=50,
    ),
    "franka_emika_panda/hand": dict(
        pos="0.340 0.008 0.059",
        xyaxes="-0.023 1.000 0.000 -0.084 -0.002 0.996",
        fovy=50,
    ),
    "kuka_iiwa_14/iiwa14": dict(
        pos="0.212 1.138 0.977",
        xyaxes="-1.000 0.027 0.000 -0.012 -0.441 0.898",
        fovy=50,
    ),
    "robotis_op3/op3": dict(
        pos="0.673 -0.024 0.447", xyaxes="0.035 0.999 0.000 -0.252 0.009 0.968", fovy=50
    ),
    "ufactory_lite6/lite6": dict(
        pos="0.077 -0.778 0.528",
        xyaxes="1.000 -0.004 -0.000 0.001 0.274 0.962",
        fovy=50,
    ),
    "unitree_g1/g1": dict(
        pos="1.083 -0.998 1.323",
        xyaxes="0.686 0.728 -0.000 -0.296 0.279 0.914",
        fovy=50,
    ),
    "universal_robots_ur5e/ur5e": dict(
        pos="0.603 1.012 0.595",
        xyaxes="-0.932 0.363 -0.000 -0.080 -0.206 0.975",
        fovy=50,
    ),
    "universal_robots_ur10e/ur10e": dict(
        pos="1.286 -0.798 0.889",
        xyaxes="0.696 0.718 -0.000 -0.224 0.218 0.950",
        fovy=50,
    ),
    "unitree_z1/z1": dict(
        pos="0.305 -0.400 0.552", xyaxes="0.755 0.656 0.000 -0.359 0.413 0.837", fovy=50
    ),
    "unitree_h1/h1": dict(
        pos="1.944 -0.828 1.894", xyaxes="0.415 0.910 0.000 -0.369 0.168 0.914", fovy=50
    ),
    "unitree_go2/go2": dict(
        pos="0.753 -0.427 0.433", xyaxes="0.518 0.856 0.000 -0.284 0.172 0.943", fovy=50
    ),
    "unitree_go1/go1": dict(
        pos="0.679 -0.553 0.530",
        xyaxes="0.638 0.770 -0.000 -0.328 0.272 0.905",
        fovy=50,
    ),
    "unitree_a1/a1": dict(
        pos="0.654 -0.564 0.536",
        xyaxes="0.676 0.737 -0.000 -0.327 0.299 0.896",
        fovy=50,
    ),
    "trossen_vx300s/vx300s": dict(
        pos="0.583 0.317 0.549",
        xyaxes="-0.531 0.847 0.000 -0.434 -0.272 0.859",
        fovy=50,
    ),
    "robotiq_2f85/2f85": dict(
        pos="-0.009 -0.251 0.107",
        xyaxes="0.999 -0.033 -0.000 0.005 0.150 0.989",
        fovy=50,
    ),
    "rethink_robotics_sawyer/sawyer": dict(
        pos="1.014 -0.494 0.876",
        xyaxes="0.555 0.832 -0.000 -0.372 0.248 0.895",
        fovy=50,
    ),
    "realsense_d435i/d435i": dict(
        pos="-0.000 -0.002 0.128",
        xyaxes="1.000 -0.000 0.000 0.000 1.000 0.017",
        fovy=50,
    ),
    "hello_robot_stretch/stretch": dict(
        pos="1.394 -0.761 1.411",
        xyaxes="0.491 0.871 -0.000 -0.318 0.179 0.931",
        fovy=50,
    ),
    "google_robot/robot": dict(
        pos="2.049 -0.945 1.503", xyaxes="0.409 0.913 0.000 -0.222 0.099 0.970", fovy=50
    ),
    "google_barkour_vb/barkour_vb": dict(
        pos="0.887 0.338 0.565",
        xyaxes="-0.388 0.922 0.000 -0.460 -0.194 0.867",
        fovy=50,
    ),
    "google_barkour_v0/barkour_v0": dict(
        pos="0.733 0.320 0.558",
        xyaxes="-0.416 0.909 -0.000 -0.441 -0.202 0.875",
        fovy=50,
    ),
    "bitcraze_crazyflie_2/cf2": dict(
        pos="0.037 -0.142 0.206", xyaxes="0.963 0.268 0.000 -0.167 0.599 0.783", fovy=50
    ),
    "anybotics_anymal_b/anymal_b": dict(
        pos="0.930 -1.239 1.221", xyaxes="0.809 0.587 0.000 -0.308 0.424 0.852", fovy=50
    ),
    "anybotics_anymal_c/anymal_c": dict(
        pos="1.423 -0.825 0.895", xyaxes="0.542 0.841 0.000 -0.286 0.184 0.940", fovy=50
    ),
    "aloha/aloha": dict(
        pos="0.484 1.158 0.836",
        xyaxes="-0.939 0.345 -0.000 -0.162 -0.441 0.883",
        fovy=50,
    ),
    "agility_cassie/cassie": dict(
        pos="1.277 -1.122 1.053", xyaxes="0.655 0.756 0.000 -0.196 0.170 0.966", fovy=50
    ),
}

KEEP_LIGHT = ["go1", "a1", "op3", "aloha", "left_hand", "right_hand"]


def create_arena():
    arena = mjcf.RootElement()

    arena.visual.quality.shadowsize = 8192
    arena.visual.headlight.diffuse = (0.6,) * 3
    arena.visual.headlight.ambient = (0.3,) * 3
    arena.visual.headlight.specular = (0.2,) * 3

    getattr(arena.visual, "global").offheight = 720
    getattr(arena.visual, "global").offwidth = 1280

    arena.asset.add(
        "texture",
        type="skybox",
        builtin="gradient",
        height=512,
        width=512,
        rgb1="1 1 1",
        rgb2="1 1 1",
    )

    return arena


# ============================================================================ #

# Grab XML files from subdirectories.
ROOT_DIR = Path(__file__).parent
(ROOT_DIR / "assets").mkdir(parents=True, exist_ok=True)
MODEL_DIRS = [f for f in ROOT_DIR.iterdir() if f.is_dir()]
MODEL_XMLS = []


def _get_xmls(pattern: str):
    for d in MODEL_DIRS:
        yield from d.glob(pattern)


MODEL_XMLS = list(_get_xmls("*.xml"))

# Filter out unwanted XML files.
filter_words = ["scene", "keyframe", "mjx", "actuator", "nohand"]
for word in filter_words:
    MODEL_XMLS = [f for f in MODEL_XMLS if word not in f.stem]


# Sort XML files.
def sort_func(xml):
    name = f"{xml.parent.stem}/{xml.stem}"
    return (MODEL_MAP[name], xml.stem)


MODEL_XMLS = sorted(MODEL_XMLS, key=sort_func)

paths = []
for xml in tqdm(MODEL_XMLS):
    try:
        robot_maker = xml.parent.stem
        robot_name = xml.stem
        robot = f"{robot_maker}/{robot_name}"

        if robot not in CAMERA_MAP:
            continue

        arena = create_arena()

        model_xml = mjcf.from_path(xml.as_posix(), escape_separators=True)
        for light in model_xml.find_all("light"):
            if robot_name not in KEEP_LIGHT:
                light.remove()

        if robot in CAMERA_MAP:
            camera_kwargs = CAMERA_MAP[robot]
            arena.worldbody.add("camera", name="thumbnail", **camera_kwargs)

        arena.include_copy(model_xml, override_attributes=True)

        physics = mjcf.Physics.from_mjcf_model(arena)

        try:
            physics.reset(keyframe_id=0)
        except:
            physics.reset()

        physics.forward()

        if robot in CAMERA_MAP:
            img = physics.render(height=500, width=500, camera_id="thumbnail")
        else:
            img = physics.render(height=500, width=500)

        img = cv2.putText(
            img.copy(),
            NAME_MAP[robot],
            (5, 480),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        filename = f"assets/{robot_maker}-{robot_name}.png"
        paths.append(filename)

        png = np.zeros((500, 500, 4), dtype=np.uint8)
        u, v = np.where(np.all(img == 255, axis=-1))
        png[u, v, -1] = 0
        png[u, v, :3] = 0
        u, v = np.where(np.any(img != 255, axis=-1))
        png[u, v, :3] = img[u, v]
        png[u, v, -1] = 255
        Image.fromarray(png).save(filename)
    except Exception as e:
        print(e)
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

mdFile = MdUtils(file_name="gallery")
mdFile.new_table(columns=N_COLS, rows=N_ROWS, text=table, text_align="center")
mdFile.create_md_file()
