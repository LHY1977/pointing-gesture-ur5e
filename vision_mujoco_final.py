import cv2
import numpy as np
import mujoco
import mujoco.viewer
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#  1. 仿真模型与控制器参数
MODEL_PATH = "universal_robots_ur5e/ur5e.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# 末端控制参考点：UR5e 法兰盘安装点(TCP)
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

# --- CLIK 控制器增益 ---
K_POS = 50.0          # 位置误差增益
K_ORI = 15.0          # 姿态误差增益
LAMBDA = 0.1          # 阻尼最小二乘的阻尼系数
MAX_QDOT = 2.0        # 单步关节速度上限
dt = model.opt.timestep

# --- 任务参数 ---
HOVER_DIST = 0.07     # 末端悬停在目标前方的距离，米

# --- 初始关节姿态(规整的非奇异姿态) ---
data.qpos[:] = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
mujoco.mj_forward(model, data)

# 雅可比缓存(位置部分jacp + 姿态部分jacr，各3×nv)
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))

# 目标关节角
q_target = data.qpos.copy()

# 6个关节的限位(弧度)，每步钳制防止超界
JOINT_LIMITS = [
    (-2 * np.pi, 2 * np.pi),   # joint1 shoulder_pan
    (-2 * np.pi, 2 * np.pi),   # joint2 shoulder_lift
    (-np.pi,     np.pi),       # joint3 elbow
    (-2 * np.pi, 2 * np.pi),   # joint4 wrist_1
    (-2 * np.pi, 2 * np.pi),   # joint5 wrist_2
    (-2 * np.pi, 2 * np.pi),   # joint6 wrist_3
]

#  2. 控制器核心函数
def orientation_error(R_current, R_target):
    R_err = R_target @ R_current.T
    angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
    if abs(angle) < 1e-6:
        return np.zeros(3)
    axis = np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ]) / (2 * np.sin(angle))
    return axis * angle


def look_at_rotation(z_dir):
    z = z_dir / (np.linalg.norm(z_dir) + 1e-9)
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def clik_step(ball_pos):
    global q_target

    # --- 末端当前位置与姿态 ---
    x_current = data.site_xpos[ee_site_id].copy()
    R_current = data.site_xmat[ee_site_id].reshape(3, 3).copy()

    # --- 期望指向：从末端指向目标 ---
    to_ball = ball_pos - x_current
    dist = np.linalg.norm(to_ball)
    z_dir = R_current[:, 2] if dist < 1e-6 else to_ball / dist

    # --- 期望位置(目标前方HOVER_DIST处) 与 期望姿态(z轴对准目标) ---
    x_desired = ball_pos - z_dir * HOVER_DIST
    R_desired = look_at_rotation(z_dir)

    # --- 6维误差：位置(3) + 姿态(3)，各乘以对应增益 ---
    e_pos = x_desired - x_current
    e_ori = orientation_error(R_current, R_desired)
    e = np.concatenate([K_POS * e_pos, K_ORI * e_ori])

    # --- 完整6×6几何雅可比(位置jacp + 姿态jacr) ---
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
    J = np.vstack([jacp, jacr])

    # --- 阻尼最小二乘解关节速度: q̇ = Jᵀ(JJᵀ + λ²I)⁻¹ e ---
    JJt = J @ J.T + (LAMBDA ** 2) * np.eye(6)
    q_dot = J.T @ np.linalg.solve(JJt, e)

    # --- 速度限幅 ---
    norm = np.linalg.norm(q_dot)
    if norm > MAX_QDOT:
        q_dot = q_dot * (MAX_QDOT / norm)

    # --- 积分更新关节角 + 硬限位 ---
    q_target += q_dot * dt
    for i in range(model.nv):
        q_target[i] = np.clip(q_target[i], JOINT_LIMITS[i][0], JOINT_LIMITS[i][1])

    data.ctrl[:] = q_target
    for _ in range(16):
        mujoco.mj_step(model, data)

    return np.linalg.norm(e_pos)


def add_sphere(scn, pos, radius, rgba):
    """在 MuJoCo viewer 的用户场景层添加一个可视化球(不参与物理碰撞)"""
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([radius, 0, 0]),
                        pos=np.asarray(pos, dtype=np.float64),
                        mat=np.eye(3).flatten(),
                        rgba=np.asarray(rgba, dtype=np.float32))
    scn.ngeom += 1


#  3. 视觉-机器人映射参数
# 桌面目标中心
X_CENTER = 0.45
Y_CENTER = 0.0

SENSITIVITY = 8.0         # 指向方向 → 桌面位移的放大系数
SMOOTH = 0.15            # 低通滤波系数(越小越平滑、响应越慢)，防抖
SIGN_X = -1.0            # 上下指向 → 前后的符号
SIGN_Y = 1.0            # 左右指向 → 左右的符号

# 目标点可达范围(限制在机械臂工作区内)
X_RANGE = (0.25, 0.60)   # 前后
Y_RANGE = (-0.35, 0.35)  # 左右
Z_RANGE = (0.05, 0.35)   # 高度

# 手部图像高度 → 目标Z高度的映射区间
HAND_Y_TOP = 0.25        # 手举到此图像高度(归一化y)时目标最高
HAND_Y_BOTTOM = 0.75     # 手降到此图像高度时目标最低

#  4. 手势识别参数与函数
# MediaPipe 21关节点中本项目使用的索引
IDX_THUMB_TIP = 4        # 拇指尖(确认手势)
IDX_A = 5               # 食指根部(指向射线起点)
IDX_B = 8               # 食指指尖(指向射线方向)

THUMB_OPEN_THRESHOLD_3D = 0.055   # 拇指张开判定阈值(米)
TRIGGER_FRAMES = 5                # 连续满足阈值的帧数(防误触发)


def thumb_is_open_3d(wl):
    """用拇指尖到食指根部的3D距离判断拇指是否张开('手枪'确认手势)"""
    tx, ty, tz = wl[IDX_THUMB_TIP].x, wl[IDX_THUMB_TIP].y, wl[IDX_THUMB_TIP].z
    rx, ry, rz = wl[IDX_A].x, wl[IDX_A].y, wl[IDX_A].z
    dist = np.linalg.norm([tx - rx, ty - ry, tz - rz])
    return dist > THUMB_OPEN_THRESHOLD_3D, dist


def hand_height_to_z(hand_y):
    """把食指根部的图像y坐标(0~1)映射到目标高度Z。手越高→目标越高。"""
    t = (HAND_Y_BOTTOM - hand_y) / (HAND_Y_BOTTOM - HAND_Y_TOP)
    t = np.clip(t, 0.0, 1.0)
    return Z_RANGE[0] + t * (Z_RANGE[1] - Z_RANGE[0])

#  5. MediaPipe 初始化
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("摄像头打不开！")
    exit()

#  6. 主控制循环
z_init = (Z_RANGE[0] + Z_RANGE[1]) / 2
preview = np.array([X_CENTER, Y_CENTER, z_init])   # 黄球：瞄准预览点
target = np.array([X_CENTER, Y_CENTER, z_init])    # 红球：机械臂实际目标
has_confirmed_once = False                          # 首次确认前机械臂不动
current_open_frames = 0                             # 拇指连续张开帧计数

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        ok, frame = cap.read()
        if not ok:
            break

        # --- 读帧、镜像、转给MediaPipe ---
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        confirmed = False

        if result.hand_landmarks and result.hand_world_landmarks:
            h, w, _ = frame.shape
            lm = result.hand_landmarks[0]         # 图像坐标(画点用)
            wl = result.hand_world_landmarks[0]   # 世界坐标(算方向用)

            # --- 在画面上绘制食指指向射线 ---
            ax, ay = int(lm[IDX_A].x * w), int(lm[IDX_A].y * h)
            bx, by = int(lm[IDX_B].x * w), int(lm[IDX_B].y * h)
            cv2.circle(frame, (ax, ay), 8, (255, 150, 50), -1)
            cv2.circle(frame, (bx, by), 8, (50, 255, 150), -1)
            cv2.line(frame, (ax, ay), (bx, by), (255, 255, 255), 2)

            # --- 桌面XY：用食指方向向量映射 ---
            A_w = np.array([wl[IDX_A].x, wl[IDX_A].y, wl[IDX_A].z])
            B_w = np.array([wl[IDX_B].x, wl[IDX_B].y, wl[IDX_B].z])
            d = B_w - A_w
            tx = np.clip(X_CENTER + SIGN_X * SENSITIVITY * d[1], *X_RANGE)
            ty = np.clip(Y_CENTER + SIGN_Y * SENSITIVITY * d[0], *Y_RANGE)

            # --- 高度Z：用食指根部的图像垂直位置映射 ---
            tz = hand_height_to_z(lm[IDX_A].y)

            # --- 低通滤波更新预览点(防抖) ---
            new_preview = np.array([tx, ty, tz])
            preview = (1 - SMOOTH) * preview + SMOOTH * new_preview

            # --- 拇指确认手势(连续TRIGGER_FRAMES帧才触发) ---
            is_open, dist = thumb_is_open_3d(wl)
            current_open_frames = current_open_frames + 1 if is_open else 0
            if current_open_frames >= TRIGGER_FRAMES:
                confirmed = True
                target = preview.copy()           # 锁定目标
                has_confirmed_once = True          # 解锁机械臂
                px, py = int(lm[IDX_THUMB_TIP].x * w), int(lm[IDX_THUMB_TIP].y * h)
                cv2.circle(frame, (px, py), 10, (0, 0, 255), -1)

            # --- 屏幕状态提示 ---
            status = "CONFIRMED" if confirmed else "aiming..."
            color = (0, 0, 255) if confirmed else (200, 200, 200)
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"thumb: {dist:.3f}m | target Z: {preview[2]:.2f}m",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Hand Control (q to quit)", frame)

        # --- 首次确认后，每帧多步CLIK让机械臂更快跟上 ---
        if has_confirmed_once:
            for _ in range(3):
                clik_step(target)

        # --- 绘制黄球(预览)与红球(目标) ---
        viewer.user_scn.ngeom = 0
        add_sphere(viewer.user_scn, preview, 0.02, [1.0, 0.85, 0.1, 0.6])
        add_sphere(viewer.user_scn, target, 0.025, [1.0, 0.1, 0.1, 1.0])
        viewer.sync()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
landmarker.close()