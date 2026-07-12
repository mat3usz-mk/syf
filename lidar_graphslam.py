"""
LiDAR D500 / STL-19P - GraphSLAM pipeline (bez zewnetrznych zaleznosci SLAM)
Wlasny solver Gauss-Newton do optymalizacji grafu pozycji
Wymagania: pip install pyserial numpy scipy matplotlib
"""

import serial
import struct
import math
import threading
import collections
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import KDTree

# ─────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────
PORT             = "COM5"
BAUD             = 230400
SCAN_POINTS_MIN  = 60      # min punktow w skanie
SCAN_COLLECT_SEC = 0.5     # co ile sekund nowy skan
ICP_MAX_ITER     = 30
ICP_MAX_DIST     = 0.5     # [m]
LOOP_THRESH_M    = 1.2     # prog loop closure [m]
LOOP_MIN_GAP     = 20      # min roznica indeksow wezlow
OPT_EVERY        = 10      # optymalizuj co N wezlow
OPT_ITERS        = 20      # iteracje Gauss-Newton

# ─────────────────────────────────────────
# PARSER STL-19P
# ─────────────────────────────────────────
CRC_TABLE = [
    0x00,0x4d,0x9a,0xd7,0x79,0x34,0xe3,0xae,0xf2,0xbf,0x68,0x25,0x8b,0xc6,0x11,0x5c,
    0xa9,0xe4,0x33,0x7e,0xd0,0x9d,0x4a,0x07,0x5b,0x16,0xc1,0x8c,0x22,0x6f,0xb8,0xf5,
    0x1f,0x52,0x85,0xc8,0x66,0x2b,0xfc,0xb1,0xed,0xa0,0x77,0x3a,0x94,0xd9,0x0e,0x43,
    0xb6,0xfb,0x2c,0x61,0xcf,0x82,0x55,0x18,0x44,0x09,0xde,0x93,0x3d,0x70,0xa7,0xea,
    0x3e,0x73,0xa4,0xe9,0x47,0x0a,0xdd,0x90,0xcc,0x81,0x56,0x1b,0xb5,0xf8,0x2f,0x62,
    0x97,0xda,0x0d,0x40,0xee,0xa3,0x74,0x39,0x65,0x28,0xff,0xb2,0x1c,0x51,0x86,0xcb,
    0x21,0x6c,0xbb,0xf6,0x58,0x15,0xc2,0x8f,0xd3,0x9e,0x49,0x04,0xaa,0xe7,0x30,0x7d,
    0x88,0xc5,0x12,0x5f,0xf1,0xbc,0x6b,0x26,0x7a,0x37,0xe0,0xad,0x03,0x4e,0x99,0xd4,
    0x7c,0x31,0xe6,0xab,0x05,0x48,0x9f,0xd2,0x8e,0xc3,0x14,0x59,0xf7,0xba,0x6d,0x20,
    0xd5,0x98,0x4f,0x02,0xac,0xe1,0x36,0x7b,0x27,0x6a,0xbd,0xf0,0x5e,0x13,0xc4,0x89,
    0x63,0x2e,0xf9,0xb4,0x1a,0x57,0x80,0xcd,0x91,0xdc,0x0b,0x46,0xe8,0xa5,0x72,0x3f,
    0xca,0x87,0x50,0x1d,0xb3,0xfe,0x29,0x64,0x38,0x75,0xa2,0xef,0x41,0x0c,0xdb,0x96,
    0x42,0x0f,0xd8,0x95,0x3b,0x76,0xa1,0xec,0xb0,0xfd,0x2a,0x67,0xc9,0x84,0x53,0x1e,
    0xeb,0xa6,0x71,0x3c,0x92,0xdf,0x08,0x45,0x19,0x54,0x83,0xce,0x60,0x2d,0xfa,0xb7,
    0x5d,0x10,0xc7,0x8a,0x24,0x69,0xbe,0xf3,0xaf,0xe2,0x35,0x78,0xd6,0x9b,0x4c,0x01,
    0xf4,0xb9,0x6e,0x23,0x8d,0xc0,0x17,0x5a,0x06,0x4b,0x9c,0xd1,0x7f,0x32,0xe5,0xa8,
]
FULL_LEN     = 47
POINTS_COUNT = 12

def crc8(data):
    crc = 0x00
    for b in data:
        crc = CRC_TABLE[(crc ^ b) & 0xFF]
    return crc

def parse_packet(pkt):
    if len(pkt) != FULL_LEN or pkt[0] != 0x54 or pkt[1] != 0x2C:
        return None
    if crc8(pkt[0:46]) != pkt[46]:
        return None
    start_angle = struct.unpack_from('<H', pkt, 4)[0]
    end_angle   = struct.unpack_from('<H', pkt, 42)[0]
    step = ((end_angle - start_angle) % 36000) / 11
    points = []
    for i in range(POINTS_COUNT):
        offset  = 6 + i * 3
        dist_mm = struct.unpack_from('<H', pkt, offset)[0]
        if 30 < dist_mm < 12000:
            angle_deg = (int(start_angle + step * i) % 36000) / 100.0
            a = math.radians(angle_deg)
            d = dist_mm / 1000.0
            points.append((d * math.cos(a), d * math.sin(a)))
    return points

# ─────────────────────────────────────────
# WATEK SERIAL
# ─────────────────────────────────────────
raw_buf   = collections.deque(maxlen=50000)
stop_flag = threading.Event()

def serial_thread(ser):
    buf = bytearray()
    while not stop_flag.is_set():
        n = ser.in_waiting
        if not n:
            continue
        buf.extend(ser.read(min(n, 4096)))
        while len(buf) >= FULL_LEN:
            idx = -1
            for i in range(len(buf) - 1):
                if buf[i] == 0x54 and buf[i+1] == 0x2C:
                    idx = i
                    break
            if idx == -1:
                buf = buf[-1:]
                break
            if idx > 0:
                buf = buf[idx:]
            if len(buf) < FULL_LEN:
                break
            r = parse_packet(bytes(buf[:FULL_LEN]))
            if r is not None:
                raw_buf.extend(r)
                buf = buf[FULL_LEN:]
            else:
                buf = buf[1:]

# ─────────────────────────────────────────
# ICP 2D
# ─────────────────────────────────────────
def icp_2d(src, dst):
    src = src.copy().astype(np.float64)
    dst = dst.astype(np.float64)
    R_total = np.eye(2)
    t_total = np.zeros(2)
    tree = KDTree(dst)
    prev_err = np.inf

    for _ in range(ICP_MAX_ITER):
        dists, idx = tree.query(src, distance_upper_bound=ICP_MAX_DIST)
        mask = dists < ICP_MAX_DIST
        if mask.sum() < 10:
            return R_total, t_total, False
        s = src[mask]
        d = dst[idx[mask]]
        cs, cd = s.mean(0), d.mean(0)
        U, _, Vt = np.linalg.svd((s - cs).T @ (d - cd))
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = cd - R @ cs
        src = (R @ src.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t
        err = np.mean(dists[mask])
        if abs(prev_err - err) < 1e-6:
            break
        prev_err = err

    return R_total, t_total, True

def R_to_theta(R):
    return math.atan2(R[1, 0], R[0, 0])

# ─────────────────────────────────────────
# WLASNY SOLVER GAUSS-NEWTON (SE2)
# ─────────────────────────────────────────
def angle_wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def pose_compose(p1, p2):
    """Zloz dwie pozy SE2: p1 o p2."""
    c, s = math.cos(p1[2]), math.sin(p1[2])
    return np.array([
        p1[0] + c * p2[0] - s * p2[1],
        p1[1] + s * p2[0] + c * p2[1],
        angle_wrap(p1[2] + p2[2])
    ])

def pose_inv(p):
    """Odwrotnosc pozy SE2."""
    c, s = math.cos(p[2]), math.sin(p[2])
    return np.array([
        -(c * p[0] + s * p[1]),
        -(-s * p[0] + c * p[1]),
        angle_wrap(-p[2])
    ])

def pose_error(pi, pj, z_ij):
    """Blad krawedzi: e = inv(z_ij) o (inv(pi) o pj)."""
    return pose_compose(pose_inv(z_ij), pose_compose(pose_inv(pi), pj))

def jacobian_e_ij(pi, pj):
    """Jakobian bledu wzgledem pi i pj."""
    c, s = math.cos(pi[2]), math.sin(pi[2])
    dx = pj[0] - pi[0]
    dy = pj[1] - pi[1]
    # J wzgledem pi
    Ji = np.array([
        [-c, -s,  -s * dx + c * dy],
        [ s, -c,  -c * dx - s * dy],
        [ 0,  0,  -1.0            ]
    ], dtype=np.float64)
    # J wzgledem pj
    Jj = np.array([
        [c,  s,  0],
        [-s, c,  0],
        [0,  0,  1.0]
    ], dtype=np.float64)
    return Ji, Jj

def optimize_graph(poses, edges, n_iter=OPT_ITERS):
    """
    poses: lista [x, y, theta] (N x 3)
    edges: lista (i, j, z_ij=[dx,dy,dtheta], info=3x3)
    Zwraca: poprawione poses
    """
    n = len(poses)
    x = np.array(poses, dtype=np.float64).flatten()   # 3N

    for iteration in range(n_iter):
        H = np.zeros((3 * n, 3 * n))
        b = np.zeros(3 * n)

        for (i, j, z_ij, info) in edges:
            pi = x[3*i:3*i+3]
            pj = x[3*j:3*j+3]
            e  = pose_error(pi, pj, z_ij)
            e[2] = angle_wrap(e[2])
            Ji, Jj = jacobian_e_ij(pi, pj)

            # wklad do macierzy H i wektora b
            Hii = Ji.T @ info @ Ji
            Hij = Ji.T @ info @ Jj
            Hjj = Jj.T @ info @ Jj
            bi  = Ji.T @ info @ e
            bj  = Jj.T @ info @ e

            H[3*i:3*i+3, 3*i:3*i+3] += Hii
            H[3*i:3*i+3, 3*j:3*j+3] += Hij
            H[3*j:3*j+3, 3*i:3*i+3] += Hij.T
            H[3*j:3*j+3, 3*j:3*j+3] += Hjj
            b[3*i:3*i+3]             += bi
            b[3*j:3*j+3]             += bj

        # ufiksuj pierwsze wezel (nie ruszaj poza startowa)
        H[:3, :] = 0
        H[:, :3] = 0
        H[:3, :3] = np.eye(3) * 1e10

        # rozwiaz H * dx = -b
        try:
            dx = np.linalg.solve(H, -b)
        except np.linalg.LinAlgError:
            break

        x += dx
        for k in range(n):
            x[3*k+2] = angle_wrap(x[3*k+2])

        if np.linalg.norm(dx) < 1e-5:
            break

    return x.reshape(n, 3).tolist()

# ─────────────────────────────────────────
# SLAM STATE
# ─────────────────────────────────────────
class SlamState:
    def __init__(self):
        self.poses      = []          # [[x, y, theta], ...]
        self.scans      = []          # [Nx2 array, ...]
        self.edges      = []          # [(i, j, z_ij, info), ...]
        self.map_pts    = np.zeros((0, 2))
        self.lock       = threading.Lock()
        self.loop_count = 0
        self.stats      = {"nodes": 0, "loops": 0, "opt": 0}

    def add_scan(self, pts_xy):
        scan = np.array(pts_xy, dtype=np.float64)
        if len(scan) < SCAN_POINTS_MIN:
            return

        with self.lock:
            n = len(self.poses)

            if n == 0:
                self.poses.append([0.0, 0.0, 0.0])
                self.scans.append(scan)
                self.stats["nodes"] = 1
                self._update_map()
                print("[SLAM] Start — wezel #0")
                return

            R, t, ok = icp_2d(scan, self.scans[-1])
            if not ok:
                return

            dtheta = R_to_theta(R)
            dx, dy = float(t[0]), float(t[1])

            prev = self.poses[-1]
            c, s = math.cos(prev[2]), math.sin(prev[2])
            new_pose = [
                prev[0] + c * dx - s * dy,
                prev[1] + s * dx + c * dy,
                angle_wrap(prev[2] + dtheta)
            ]
            self.poses.append(new_pose)
            self.scans.append(scan)

            z_ij = np.array([dx, dy, dtheta])
            info = np.eye(3) * 500.0
            self.edges.append((n - 1, n, z_ij, info))
            self.stats["nodes"] = n + 1

            self._check_loop(n, scan)

            if (n + 1) % OPT_EVERY == 0:
                self.poses = optimize_graph(self.poses, self.edges)
                self.stats["opt"] += 1
                print(f"[SLAM] Optymalizacja #{self.stats['opt']}  "
                      f"({self.stats['nodes']} wezlow, {self.loop_count} petli)")

            self._update_map()

    def _check_loop(self, cur_idx, cur_scan):
        if cur_idx < LOOP_MIN_GAP + 5:
            return
        check_poses = self.poses[:cur_idx - LOOP_MIN_GAP]
        xy = np.array([[p[0], p[1]] for p in check_poses])
        tree = KDTree(xy)
        cur_xy = self.poses[cur_idx][:2]
        candidates = tree.query_ball_point(cur_xy, LOOP_THRESH_M)
        for ci in candidates:
            R, t, ok = icp_2d(cur_scan, self.scans[ci],)
            if not ok:
                continue
            dx, dy = float(t[0]), float(t[1])
            if abs(dx) < LOOP_THRESH_M and abs(dy) < LOOP_THRESH_M:
                dtheta = R_to_theta(R)
                z_ij   = np.array([dx, dy, dtheta])
                info   = np.eye(3) * 800.0
                self.edges.append((ci, cur_idx, z_ij, info))
                self.loop_count += 1
                self.stats["loops"] = self.loop_count
                print(f"[SLAM] Loop closure! #{cur_idx} <-> #{ci}")
                break

    def _update_map(self):
        parts = []
        for pose, scan in zip(self.poses, self.scans):
            x, y, theta = pose
            c, s = math.cos(theta), math.sin(theta)
            R = np.array([[c, -s], [s, c]])
            parts.append((R @ scan.T).T + np.array([x, y]))
        if parts:
            self.map_pts = np.vstack(parts)

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print(f"[INFO] Lacze z {PORT} @ {BAUD}...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0)
    except serial.SerialException as e:
        print(f"[BLAD] {e}")
        input("Nacisnij Enter...")
        return
    print("[OK] Polaczono!\n")
    print("Poruszaj robotem powoli. Zamknij okno zeby zapisac mape.\n")

    t_ser = threading.Thread(target=serial_thread, args=(ser,), daemon=True)
    t_ser.start()

    slam = SlamState()
    scan_buf  = []
    scan_lock = threading.Lock()
    last_t    = time.time()

    def scan_collector():
        nonlocal scan_buf, last_t
        while not stop_flag.is_set():
            with scan_lock:
                while raw_buf:
                    scan_buf.append(raw_buf.popleft())
            now = time.time()
            if now - last_t >= SCAN_COLLECT_SEC:
                with scan_lock:
                    pts, scan_buf = list(scan_buf), []
                last_t = now
                if pts:
                    slam.add_scan(pts)
            time.sleep(0.01)

    t_col = threading.Thread(target=scan_collector, daemon=True)
    t_col.start()

    # --- GUI ---
    fig = plt.figure(figsize=(16, 8), facecolor="#0d0d1a")
    ax_map  = fig.add_subplot(121)
    ax_scan = fig.add_subplot(122, projection='polar')

    for ax in [ax_map, ax_scan]:
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="#666688")
        ax.grid(color="#222244", linewidth=0.5)

    ax_map.set_aspect('equal')
    ax_map.set_title("Mapa globalna (GraphSLAM)", color="white", fontsize=12)
    sc_map   = ax_map.scatter([], [], s=0.8, c='#4fc3f7', alpha=0.5)
    path_ln, = ax_map.plot([], [], color='#ffcc00', linewidth=1.2,
                           alpha=0.8, label="Trasa")
    ax_map.legend(facecolor="#1a1a2e", labelcolor="white")

    ax_scan.set_title("Biezacy skan", color="white", fontsize=12, pad=18)
    ax_scan.set_ylim(0, 6)
    ax_scan.set_rticks([1, 2, 3, 4, 5, 6])
    sc_scan = ax_scan.scatter([], [], s=2, c=[], cmap="plasma",
                              vmin=0, vmax=6, alpha=0.9)
    cbar = fig.colorbar(sc_scan, ax=ax_scan, pad=0.1, shrink=0.65)
    cbar.set_label("m", color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    def update(_):
        with slam.lock:
            mp    = slam.map_pts.copy() if len(slam.map_pts) > 0 else None
            poses = list(slam.poses)
            st    = dict(slam.stats)
            cur   = slam.scans[-1].copy() if slam.scans else None

        if mp is not None and len(mp) > 0:
            sc_map.set_offsets(mp)
            if len(poses) > 1:
                xs = [p[0] for p in poses]
                ys = [p[1] for p in poses]
                path_ln.set_data(xs, ys)
                ax_map.relim()
                ax_map.autoscale_view()
        ax_map.set_title(
            f"Mapa  |  węzłów: {st['nodes']}  "
            f"pętli: {st['loops']}  opt: {st['opt']}",
            color="white", fontsize=11
        )

        if cur is not None and len(cur) > 0:
            ang = np.arctan2(cur[:, 1], cur[:, 0])
            dst = np.hypot(cur[:, 0], cur[:, 1])
            sc_scan.set_offsets(np.column_stack([ang, dst]))
            sc_scan.set_array(dst)

        return sc_map, path_ln, sc_scan

    ani = animation.FuncAnimation(fig, update, interval=200,
                                  blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

    stop_flag.set()
    ser.close()

    # --- zapis mapy ---
    with slam.lock:
        if len(slam.map_pts) > 0:
            np.save("lidar_map.npy",   slam.map_pts)
            np.save("lidar_poses.npy", np.array(slam.poses))
            print(f"\n[ZAPIS] lidar_map.npy  — {len(slam.map_pts)} punktow")
            print(f"[ZAPIS] lidar_poses.npy — {len(slam.poses)} wezlow")

            fig2, ax2 = plt.subplots(figsize=(12, 12),
                                     facecolor="#0d0d1a")
            ax2.set_facecolor("#0d0d1a")
            ax2.scatter(slam.map_pts[:, 0], slam.map_pts[:, 1],
                        s=0.5, c="#4fc3f7", alpha=0.4)
            p = np.array(slam.poses)
            ax2.plot(p[:, 0], p[:, 1], color="#ffcc00", linewidth=1.2)
            ax2.set_aspect("equal")
            ax2.set_title("GraphSLAM — mapa finalna", color="white")
            ax2.tick_params(colors="#666688")
            plt.tight_layout()
            plt.savefig("lidar_map_final.png", dpi=150, bbox_inches="tight")
            print("[ZAPIS] lidar_map_final.png")

if __name__ == "__main__":
    main()