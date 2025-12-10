import numpy as np
import matplotlib.pyplot as plt
import random

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
        # 添加积分项限制，防止积分饱和
        self.integral_max = 800

    def input_error(self, e):
        self.acc_error += e
        # 限制积分项，防止积分饱和
        self.acc_error = np.clip(self.acc_error, -self.integral_max, self.integral_max)
        control = self.kp * e + self.ki * self.acc_error + self.kd * (e - self.last_error)
        self.last_error = e
        return np.clip(control, -self.constraint, self.constraint)

    def set_params(self, kp, ki, kd):
        """设置PID参数"""
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def reset(self):
        """重置PID控制器状态"""
        self.acc_error = 0
        self.last_error = 0

class QuantumPIDOptimizer:
    """量子启发PID参数优化器"""
    def __init__(self, disturbance_r, disturbance_v, dt):
        self.disturbance_r = disturbance_r
        self.disturbance_v = disturbance_v
        self.dt = dt

        # 量子角度初始化 (Kp, Ki, Kd)
        self.quantum_angles = np.array([np.pi/4, np.pi/4, np.pi/4])

        # PID参数范围 - 更加保守的范围
        self.param_bounds = np.array([
            [800, 6000],  # Kp范围
            [0.1, 1.0],   # Ki范围
            [1, 15]       # Kd范围
        ])

        # 最佳参数和适应度
        self.best_params = [1500, 0.8, 8]  # 更保守的初始值
        self.best_fitness = float('inf')

        # 历史数据存储
        self.historical_data = []
        self.history_size = 400  # 增加历史数据大小

        # 优化参数
        self.rotation_step = 0.02  # 减小量子旋转步长
        self.population_size = 12   # 减小种群大小
        self.mutation_rate = 0.03  # 减小变异率

        # 性能跟踪
        self.performance_history = []

    def quantum_observe(self):
        """量子观察：生成经典解"""
        solutions = []

        # 总是包含当前最佳解
        solutions.append(self.best_params.copy())

        for _ in range(self.population_size - 1):
            # 从概率幅生成解
            params = []
            for i, angle in enumerate(self.quantum_angles):
                prob_1 = np.sin(angle)**2  # 概率幅的平方代表概率
                # 基于概率生成参数值
                param_val = self.param_bounds[i][0] + prob_1 * (
                    self.param_bounds[i][1] - self.param_bounds[i][0])
                params.append(param_val)

            # 添加小幅度变异
            if random.random() < self.mutation_rate:
                idx = random.randint(0, 2)
                mutation = random.gauss(0, 0.05)  # 小幅度高斯变异
                params[idx] = np.clip(
                    params[idx] * (1 + mutation),
                    self.param_bounds[idx][0],
                    self.param_bounds[idx][1]
                )

            solutions.append(params)

        return solutions

    def evaluate_params(self, params, historical_window=150):
        """评估PID参数性能"""
        if len(self.historical_data) < historical_window:
            return float('inf')

        # 使用历史数据模拟参数性能
        kp, ki, kd = params
        temp_pid = PIDController(kp, ki, kd)

        # 重放历史数据
        itae = 0
        control_effort = 0
        max_overshoot = 0
        error_variance = 0
        errors = []

        for i, data in enumerate(self.historical_data[-historical_window:]):
            error = data['error']
            control = temp_pid.input_error(error)

            # 计算性能指标
            itae += abs(error) * (i + 1) * self.dt
            control_effort += abs(control)
            if abs(error) > max_overshoot:
                max_overshoot = abs(error)
            errors.append(error)

        # 计算误差方差
        error_variance = np.var(errors) if len(errors) > 1 else 0

        # 综合适应度函数 - 更加注重稳定性
        fitness = (itae +
                   0.3 * control_effort +
                   15 * max_overshoot +
                   5 * error_variance)

        # 添加稳定性惩罚项
        if max_overshoot > 1.2:  # 如果超调太大，增加惩罚
            fitness *= (1 + max_overshoot)

        return fitness

    def update_quantum_state(self, solutions, fitness_values):
        """更新量子状态"""
        best_idx = np.argmin(fitness_values)
        best_solution = solutions[best_idx]
        best_fitness = fitness_values[best_idx]

        # 记录性能历史
        self.performance_history.append(best_fitness)
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)

        # 只有当明显更好时才更新最佳解
        improvement_threshold = 0.98  # 至少提高2%
        if best_fitness < self.best_fitness * improvement_threshold:
            self.best_params = best_solution
            self.best_fitness = best_fitness

        # 更新量子角度 (量子旋转门) - 更保守的更新
        for i in range(3):
            current_angle = self.quantum_angles[i]
            best_param = best_solution[i]

            # 将参数值映射到概率空间
            param_range = self.param_bounds[i][1] - self.param_bounds[i][0]
            normalized_param = (best_param - self.param_bounds[i][0]) / param_range

            # 计算目标角度 (使sin²(θ)接近normalized_param)
            target_angle = np.arcsin(np.sqrt(np.clip(normalized_param, 0.001, 0.999)))

            # 旋转角度 - 更小的步长
            rotation_direction = 1 if target_angle > current_angle else -1
            # 根据性能改进程度调整步长
            performance_improvement = 1.0
            if len(self.performance_history) > 5:
                recent_avg = np.mean(self.performance_history[-5:])
                performance_improvement = min(1.0, self.best_fitness / recent_avg)

            rotation_step = self.rotation_step * performance_improvement
            self.quantum_angles[i] += rotation_direction * rotation_step

            # 保持角度在合理范围内
            self.quantum_angles[i] = np.clip(self.quantum_angles[i], 0.1, np.pi/2 - 0.1)

    def optimize(self):
        """执行一代优化"""
        if len(self.historical_data) < 100:  # 增加最小数据要求
            return None

        # 生成候选解
        solutions = self.quantum_observe()

        # 评估候选解
        fitness_values = []
        for params in solutions:
            fitness = self.evaluate_params(params, historical_window=min(150, len(self.historical_data)))
            fitness_values.append(fitness)

        # 更新量子状态
        self.update_quantum_state(solutions, fitness_values)

        return self.best_params

    def add_data_point(self, error, control, path_error):
        """添加数据点到历史记录"""
        # 只添加有意义的数据点
        if abs(error) < 2.5:  # 过滤掉过大的误差
            self.historical_data.append({
                'error': error,
                'control': control,
                'path_error': path_error
            })

            # 保持历史数据大小
            if len(self.historical_data) > self.history_size:
                self.historical_data.pop(0)

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
        self.disturbance_r = 0.6 # 固定偏航干扰 (rad/s^2)
        self.disturbance_v = 0.2  # 固定侧向干扰 (m/s^2)

    def update(self, n1, n2):
        u, v, r, x, y, psi = self.state
        V = np.sqrt(u**2 + v**2)
        Xp1 = n1
        Xp2 = n2
        tau_u = Xp1 + Xp2
        tau_r = (Xp1 - Xp2) * self.dp
        u_dot = (1 / self.m11) * (tau_u + self.m22 * v * r - self.Xu * u)
        v_dot = (1 / self.m22) * (-self.m11 * u * r - self.Yv * v) + self.disturbance_v
        r_dot = (1 / self.m33) * (tau_r - (self.m22 - self.m11) * u * v - self.Nr * r) + self.disturbance_r
        u_new = u + u_dot * self.dt
        v_new = v + v_dot * self.dt
        r_new = r + r_dot * self.dt
        x_new = x + (u_new * np.cos(psi) - v_new * np.sin(psi)) * self.dt
        y_new = y + (u_new * np.sin(psi) + v_new * np.cos(psi)) * self.dt
        psi_new = (psi + r_new * self.dt) % (2 * np.pi)
        self.state = np.array([u_new, v_new, r_new, x_new, y_new, psi_new], dtype=float)
        return self.state.copy(), V

class LOSPathPlanning:
    def __init__(self, usv, path_points, dt, sim_time, Ld=1.5, R=2.0, debug_interval=100, update_interval=100, horizon=100):
        self.usv = usv
        self.path_points = np.array(path_points)
        self.dt = dt
        self.sim_time = sim_time
        self.Ld = Ld
        self.R = R
        self.trajectory = [usv.state[3:5].copy()]
        self.performance = []
        self.debug_interval = debug_interval
        self.update_interval = update_interval
        self.horizon = horizon
        self.step = 0
        # 使用更保守的初始PID参数
        self.pid = PIDController(kp=1500, ki=0.8, kd=8)
        self.base_rpm = 50  # 增加基础转速
        self.scale = 5.0
        self.current_segment = 0

        # 添加量子启发优化器
        self.pid_optimizer = QuantumPIDOptimizer(
            usv.disturbance_r,
            usv.disturbance_v,
            dt
        )
        self.optimization_interval = 600  # 增加优化间隔
        self.last_optimization_step = 0
        self.stability_counter = 0  # 稳定性计数器
        self.turn_detected = False  # 转弯检测
        self.turn_start_step = 0    # 转弯开始步数

        # 转弯专用PID参数
        self.turn_pid_params = [1800, 0.5, 12]  # 转弯时更激进的参数

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

        # 限制航向误差范围，避免过大误差导致控制不稳定
        delta_psi = np.clip(delta_psi, -np.pi/2, np.pi/2)

        control_rudder = self.pid.input_error(delta_psi)
        n1 = self.base_rpm + self.scale * control_rudder / 2
        n2 = self.base_rpm - self.scale * control_rudder / 2
        n1 = np.clip(n1, 0, 200)  # 确保正转速
        n2 = np.clip(n2, 0, 200)
        return delta_psi, n1, n2, control_rudder**2

    def estimate_curvature(self):
        """估算当前路径段曲率"""
        if self.current_segment < len(self.path_points) - 2:
            p1 = self.path_points[self.current_segment]
            p2 = self.path_points[self.current_segment + 1]
            p3 = self.path_points[min(self.current_segment + 2, len(self.path_points) - 1)]
            dx1 = p2[0] - p1[0]
            dy1 = p2[1] - p1[1]
            dx2 = p3[0] - p2[0]
            dy2 = p3[1] - p2[1]
            theta1 = np.arctan2(dy1, dx1)
            theta2 = np.arctan2(dy2, dx2)
            dtheta = abs((theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi)
            dist = np.linalg.norm(p2 - p1)
            curvature = dtheta / dist if dist > 0 else 0
        else:
            curvature = 0
        return curvature

    def detect_turn(self, curvature, delta_psi):
        """检测是否处于转弯状态"""
        # 高曲率或大航向误差表明正在转弯
        if curvature > 0.03 or abs(delta_psi) > 0.3:
            if not self.turn_detected:
                self.turn_detected = True
                self.turn_start_step = self.step
                # 切换到转弯专用参数
                self.pid.set_params(*self.turn_pid_params)
                print("检测到转弯，切换到转弯专用PID参数")
            return True
        else:
            # 转弯结束后保持一段时间再切换回正常参数
            if self.turn_detected and (self.step - self.turn_start_step > 200):
                self.turn_detected = False
                print("转弯结束，切换回正常PID参数")
            return False

    def simulate_future(self, current_state, Ld, R, steps, dist_to_path):
        """模拟未来 steps 步的轨迹，计算局部 C1_h, C2_h，带偏差惩罚"""
        usv_future = Differential_USV(current_state, self.dt)
        C1_h = 0
        C2_h = 0
        for _ in range(steps):
            pos = usv_future.state[3:5]
            psi = usv_future.state[5]
            self.Ld = Ld
            self.R = R
            los_psi, e_squared, p1, p2 = self.los_guidance(pos)
            if los_psi is None:
                break
            delta_psi, n1, n2, tau_r_squared = self.control_input(los_psi, psi)
            usv_future.update(n1, n2)
            C1_h += e_squared * self.dt
            C2_h += tau_r_squared * self.dt
            # 偏差惩罚
            if np.sqrt(e_squared) > 1.0:
                C1_h += (np.sqrt(e_squared) - 1.0) ** 2 * self.dt * 20.0  # 加强惩罚
        return C1_h, C2_h

    def dynamic_nash_update(self, current_state):
        """动态博弈：从当前状态模拟未来，求局部纳什均衡，考虑曲率"""
        curvature = self.estimate_curvature()
        # 根据曲率调整策略空间，缩小以加速
        if curvature > 0.05:  # 高曲率
            Ld_local = [0.5, 0.75, 1.0, 1.25]
            R_local = [0.1,0.2,0.3,0.4,0.5]
        else:  # 低曲率
            Ld_local = [1.4,1.6,1.8,2.0,2.2]
            R_local = [0.1,0.2,0.3,0.4,0.5]

        payoff_local = np.zeros((len(Ld_local), len(R_local)))
        C1_local = np.zeros((len(Ld_local), len(R_local)))
        C2_local = np.zeros((len(Ld_local), len(R_local)))
        dist_to_path = np.min(np.linalg.norm(self.path_points - current_state[3:5], axis=1))

        # 计算局部支付矩阵
        for i, Ld in enumerate(Ld_local):
            for j, R in enumerate(R_local):
                C1_h, C2_h = self.simulate_future(current_state, Ld, R, self.horizon, dist_to_path)
                J_h = 0.8 * C1_h + 0.2 * C2_h  # 优先偏差
                payoff_local[i, j] = J_h
                C1_local[i, j] = C1_h
                C2_local[i, j] = C2_h

        # 寻找局部纳什均衡
        nash_local = []
        for i, Ld in enumerate(Ld_local):
            for j, R in enumerate(R_local):
                is_nash = True
                current_J = payoff_local[i, j]
                # Ld 检查
                for i_other in range(len(Ld_local)):
                    if i_other != i and payoff_local[i_other, j] < current_J:
                        is_nash = False
                        break
                # R 检查
                for j_other in range(len(R_local)):
                    if j_other != j and payoff_local[i, j_other] < current_J:
                        is_nash = False
                        break
                if is_nash:
                    nash_local.append((Ld, R, C1_local[i, j], C2_local[i, j], current_J))

        if nash_local:
            best_local = min(nash_local, key=lambda x: x[4])
        else:
            min_idx = np.unravel_index(np.argmin(payoff_local), payoff_local.shape)
            best_local = (Ld_local[min_idx[0]], R_local[min_idx[1]],
                          C1_local[min_idx], C2_local[min_idx], payoff_local[min_idx])

        self.Ld, self.R = best_local[0], best_local[1]
        print(f"动态更新: Ld={self.Ld:.1f}, R={self.R:.1f}, J_h={best_local[4]:.2f}, 曲率={curvature:.4f}")
        return best_local

    def check_stability(self, delta_psi, control_rudder):
        """检查系统稳定性"""
        # 如果航向误差和控制量都很大，可能不稳定
        if abs(delta_psi) > 1.0 and abs(control_rudder) > 40:
            self.stability_counter += 1
        else:
            self.stability_counter = max(0, self.stability_counter - 1)

        # 如果不稳定计数超过阈值，重置PID控制器
        if self.stability_counter > 50:
            print("检测到不稳定状态，重置PID控制器")
            self.pid.reset()
            self.stability_counter = 0
            return True
        return False

    def simulate(self):
        t = 0
        C1 = 0
        C2 = 0
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

            # 检测转弯状态
            curvature = self.estimate_curvature()
            is_turning = self.detect_turn(curvature, delta_psi)

            # 检查稳定性
            if self.check_stability(delta_psi, np.sqrt(tau_r_squared)):
                # 如果不稳定，跳过本次循环的剩余部分
                state, V = self.usv.update(n1, n2)
                self.trajectory.append(self.usv.state[3:5].copy())
                t += self.dt
                self.step += 1
                continue

            # 收集数据用于PID优化 (不在转弯时收集)
            if not is_turning:
                self.pid_optimizer.add_data_point(delta_psi, np.sqrt(tau_r_squared), np.sqrt(e_squared))

            # 定期执行PID参数优化 (不在转弯时优化)
            if (not is_turning and
                self.step - self.last_optimization_step >= self.optimization_interval):
                best_params = self.pid_optimizer.optimize()
                if best_params:
                    # 平滑更新PID参数
                    kp, ki, kd = best_params
                    alpha = 0.9  # 增加平滑因子
                    new_kp = alpha * self.pid.kp + (1 - alpha) * kp
                    new_ki = alpha * self.pid.ki + (1 - alpha) * ki
                    new_kd = alpha * self.pid.kd + (1 - alpha) * kd

                    self.pid.set_params(new_kp, new_ki, new_kd)
                    print(f"PID参数优化: Kp={new_kp:.2f}, Ki={new_ki:.4f}, Kd={new_kd:.2f}")
                    self.last_optimization_step = self.step

            # 原有的动态博弈更新
            if self.step % self.update_interval == 0:
                self.dynamic_nash_update(state.copy())

            if self.step % self.debug_interval == 0:
                print(f"时间: {t:.2f}s, 路径偏差: {dist_to_path:.4f}m, 航向误差: {delta_psi:.4f}rad, "
                      f"段索引: {self.current_segment}, 当前路径点: p1=({p1[0]:.2f},{p1[1]:.2f}), p2=({p2[0]:.2f},{p2[1]:.2f}), "
                      f"Ld={self.Ld:.1f}, R={self.R:.1f}, n1: {n1:.1f}, n2: {n2:.1f}, 转弯: {is_turning}")

            state, V = self.usv.update(n1, n2)
            self.trajectory.append(self.usv.state[3:5].copy())
            C1 += e_squared * self.dt
            C2 += tau_r_squared * self.dt
            t += self.dt
            self.step += 1

        J = 0.5 * C1 + 0.5 * C2
        return C1, C2, J

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
        plt.title(f'量子启发优化PID - 动态博弈LOS路径跟踪')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-10, 100)
        plt.ylim(-10, 100)
        plt.show()

# 主程序
if __name__ == "__main__":
    center = [30, 50]  # 圆心
    radius = 30        # 半径
    # 路径点生成
    theta = np.linspace(np.pi, 2*np.pi, 15)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    path_points = np.column_stack((x, y))
    initial_state = np.array([0.5, 0.0, 0.0, 0.0, 60.0, 0])
    dt = 0.01
    sim_time = 300.0
    usv = Differential_USV(initial_state, dt)
    sim = LOSPathPlanning(usv, path_points, dt, sim_time, Ld=1.5, R=2.0)
    C1, C2, J = sim.simulate()
    print(f"Tracking Error Cost (C1): {C1:.2f} m²·s, Control Effort Cost (C2): {C2:.2f} rad²·s, J: {J:.2f}")
    sim.visualize()
