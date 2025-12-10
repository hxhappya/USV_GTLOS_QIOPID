import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class PIDController:
    def __init__(self, kp, ki, kd, constraint=50):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.constraint = constraint
        self.acc_error = 0
        self.last_error = 0

    def input_error(self, e):
        self.acc_error += e
        control = self.kp * e + self.ki * self.acc_error + self.kd * (e - self.last_error)
        self.last_error = e
        return np.clip(control, -self.constraint, self.constraint)

class Differential_USV:
    def __init__(self, state, dt):
        self.m11 = 50.05
        self.m22 = 84.36
        self.m33 = 17.21
        self.Xu = 151.57
        self.Yv = 132.5
        self.Nr = 34.56
        self.c = -1.60e-4
        self.d = 5.04e-3
        self.dp = 0.26
        self.dt = dt
        self.state = np.array(state, dtype=float)
        self.disturbance_r = 1  # 固定偏航干扰 (rad/s^2)
        self.disturbance_v = 0.4 # 固定侧向干扰 (m/s^2)

    def update(self, n1, n2):
        u, v, r, x, y, psi = self.state
        V = np.sqrt(u**2 + v**2)
        Xp1 = n1
        Xp2 = n2
        tau_u = Xp1 + Xp2
        tau_r = (Xp1 - Xp2) * self.dp
        u_dot = (1 / self.m11) * (tau_u + self.m22 * v * r - self.Xu * u)
        v_dot = (1 / self.m22) * (-self.m11 * u * r - self.Yv * v) + self.disturbance_v  # 添加侧向干扰
        r_dot = (1 / self.m33) * (tau_r - (self.m22 - self.m11) * u * v - self.Nr * r) + self.disturbance_r  # 保留偏航干扰
        u_new = u + u_dot * self.dt
        v_new = v + v_dot * self.dt
        r_new = r + r_dot * self.dt
        x_new = x + (u_new * np.cos(psi) - v_new * np.sin(psi)) * self.dt
        y_new = y + (u_new * np.sin(psi) + v_new * np.cos(psi)) * self.dt
        psi_new = (psi + r_new * self.dt) % (2 * np.pi)
        self.state = np.array([u_new, v_new, r_new, x_new, y_new, psi_new], dtype=float)
        return self.state.copy()

class LOSPathPlanning:
    def __init__(self, usv, path_points, dt, sim_time, Ld=1.5, R=1.0, debug_interval=100):
        self.usv = usv
        self.path_points = np.array(path_points)
        self.dt = dt
        self.sim_time = sim_time
        self.Ld = Ld  # 前视距离
        self.R = R    # 接纳半径
        self.trajectory = [usv.state[3:5].copy()]
        self.performance = []
        self.debug_interval = debug_interval
        self.step = 0
        self.pid = PIDController(kp=8, ki=0.01, kd=5, constraint=50)
        self.base_rpm = 50  # 增加基础转速以保证速度
        self.scale = 5.0    # 放大偏航控制
        self.current_segment = 0

    def los_guidance(self, current_pos):
        distances = np.linalg.norm(self.path_points - current_pos, axis=1)
        closest_idx = np.argmin(distances)
        self.current_segment = min(closest_idx, len(self.path_points) - 2)

        if self.current_segment >= len(self.path_points) - 1:
            return None, None, None, None
        p1 = self.path_points[self.current_segment]
        p2 = self.path_points[self.current_segment + 1]
        dist_to_p2 = np.linalg.norm(current_pos - p2)
        if dist_to_p2 < self.R and self.current_segment < len(self.path_points) - 2:
            self.current_segment += 1
            p1 = self.path_points[self.current_segment]
            p2 = self.path_points[self.current_segment + 1]
        psi_path = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        alpha = np.arctan2(current_pos[1] - p1[1], current_pos[0] - p1[0]) - psi_path
        dist_to_line = np.linalg.norm(current_pos - p1) * np.sin(alpha)
        los_psi = psi_path + np.arctan2(-dist_to_line, self.Ld)
        return los_psi, dist_to_line**2, p1, p2

    def control_input(self, los_psi, psi):
        delta_psi = (los_psi - psi + np.pi) % (2 * np.pi) - np.pi
        control_rudder = self.pid.input_error(delta_psi)
        n1 = self.base_rpm + self.scale * control_rudder / 2
        n2 = self.base_rpm - self.scale * control_rudder / 2
        n1 = np.clip(n1, -100, 200)
        n2 = np.clip(n2, -100, 200)
        return delta_psi, n1, n2, control_rudder**2

    def simulate(self):
        t = 0
        C1 = 0  # 跟踪误差成本
        C2 = 0  # 控制努力成本
        while t < self.sim_time:
            state = self.usv.state
            pos = state[3:5]
            psi = state[5]
            dist_to_path = np.min(np.linalg.norm(self.path_points - pos, axis=1))
            if dist_to_path < 2 and np.linalg.norm(pos - self.path_points[-1]) < 1.0:
                print("到达路径终点！")
                break
            los_psi, e_squared, p1, p2 = self.los_guidance(pos)
            if los_psi is None:
                print("路径结束！")
                break
            delta_psi, n1, n2, tau_r_squared = self.control_input(los_psi, psi)
            if self.step % self.debug_interval == 0:
                print(f"时间: {t:.2f}s, 路径偏差: {dist_to_path:.4f}m, 航向误差: {delta_psi:.4f}rad, "
                      f"段索引: {self.current_segment}, 当前路径点: p1=({p1[0]:.2f},{p1[1]:.2f}), p2=({p2[0]:.2f},{p2[1]:.2f}), "
                      f"n1: {n1:.1f}, n2: {n2:.1f}")
            self.usv.update(n1, n2)
            self.trajectory.append(self.usv.state[3:5].copy())
            C1 += e_squared * self.dt
            C2 += tau_r_squared * self.dt
            t += self.dt
            self.step += 1
        return C1, C2

    def visualize(self):
        trajectory = np.array(self.trajectory)
        plt.figure(figsize=(10, 8))
        plt.plot(self.path_points[:, 0], self.path_points[:, 1], 'g--', label='参考路径')
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='无人艇轨迹')
        plt.scatter(self.path_points[:, 0], self.path_points[:, 1], c='green', marker='o', s=0, label='路径点')
        plt.scatter(self.path_points[0, 0], self.path_points[0, 1], c='green', marker='o', s=50, label='路径起点')
        plt.scatter(self.path_points[-1, 0], self.path_points[-1, 1], c='red', marker='x', s=50, label='路径终点')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(f'LOS路径跟踪 (Ld={self.Ld}, R={self.R}) - 偏航干扰(0.2 rad/s²) + 侧向干扰(0.1 m/s²)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-10, 100)
        plt.ylim(-10, 100)
        plt.show()

# 主程序
if __name__ == "__main__":
    # # 定义 S 型路径，50 个点，间距约 2 m
    # x = np.linspace(0, 60, 35)
    # y = 40 + 30 * np.sin(2 * np.pi * x / 50)  # S 形正弦波
    # path_points = np.column_stack((x, y))
    center = [30, 50]  # 圆心
    radius = 30        # 半径
    # 路径点生成
    theta = np.linspace(np.pi, 2*np.pi, 15)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    path_points = np.column_stack((x, y))
    print(path_points)
    # 初始化仿真
    initial_state = np.array([0.5, 0.0, 0.0, 0.0, 60.0, 0])  # 初始位置 (0, 50)
    dt = 0.01
    sim_time = 400.0
    usv = Differential_USV(initial_state, dt)
    sim = LOSPathPlanning(usv, path_points, dt, sim_time, Ld=2.0, R=0.3)
    C1, C2 = sim.simulate()
    print(f"Tracking Error Cost (C1): {C1:.2f} m²·s, Control Effort Cost (C2): {C2:.2f} rad²·s")
    sim.visualize()
