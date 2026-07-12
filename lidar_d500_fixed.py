"""
LiDAR D500 / STL-19P - wizualizacja 360 stopni (Windows / VS Code / USB)
v4 - poprawiony parser: sync=0x54, pakiet=47B, CRC bajt[46] od bajt[0..45]
Wymagania: pip install pyserial numpy matplotlib
"""

import serial
import struct
import math
import threading
import collections
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

PORT  = "COM5"
BAUD  = 230400
MAX_D = 10.0
BUF   = 3000

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

# Struktura pakietu STL-19P / LD19 (47 bajtow):
# [0]      0x54        sync
# [1]      0x2C        ver_len (staly)
# [2-3]    speed       u16 LE, deg/s
# [4-5]    start_angle u16 LE, *0.01 deg
# [6-41]   12 punktow  po 3 bajty: dist u16 LE (mm) + intensity u8
# [42-43]  end_angle   u16 LE, *0.01 deg
# [44-45]  timestamp   u16 LE, ms
# [46]     crc8        liczony od bajtu [0] do [45]

FULL_LEN     = 47
POINTS_COUNT = 12

def crc8(data):
    crc = 0x00
    for b in data:
        crc = CRC_TABLE[(crc ^ b) & 0xFF]
    return crc

def parse_packet(pkt):
    if len(pkt) != FULL_LEN:
        return None
    if pkt[0] != 0x54 or pkt[1] != 0x2C:
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
            points.append((math.radians(angle_deg), dist_mm / 1000.0))
    return points

# ── WATEK SERIAL ─────────────────────────────────────────────────────────────
point_queue  = collections.deque(maxlen=10000)
stats        = {"ok": 0, "fail": 0}
stop_flag    = threading.Event()

def serial_thread(ser):
    buf = bytearray()
    while not stop_flag.is_set():
        n = ser.in_waiting
        if n == 0:
            continue
        buf.extend(ser.read(min(n, 4096)))

        while len(buf) >= FULL_LEN:
            # szukaj sync byte 0x54 z poprawnym ver_len 0x2C na kolejnym bajcie
            idx = -1
            for i in range(len(buf) - 1):
                if buf[i] == 0x54 and buf[i+1] == 0x2C:
                    idx = i
                    break
            if idx == -1:
                buf = buf[-1:] if len(buf) > 1 else buf
                break
            if idx > 0:
                buf = buf[idx:]
            if len(buf) < FULL_LEN:
                break

            pkt = bytes(buf[:FULL_LEN])
            result = parse_packet(pkt)
            if result is not None:
                stats["ok"] += 1
                point_queue.extend(result)
                buf = buf[FULL_LEN:]
            else:
                # CRC fail - przeskocz o 1 bajt i sprobuj ponownie
                stats["fail"] += 1
                buf = buf[1:]

# ── GUI ──────────────────────────────────────────────────────────────────────
def main():
    print(f"[INFO] Lacze z {PORT} @ {BAUD}...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0)
    except serial.SerialException as e:
        print(f"[BLAD] {e}")
        input("Nacisnij Enter...")
        return
    print("[OK] Polaczono!")

    t = threading.Thread(target=serial_thread, args=(ser,), daemon=True)
    t.start()

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(9, 9))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")
    ax.set_ylim(0, MAX_D)
    ax.tick_params(colors="#666688")
    ax.grid(color="#222244", linewidth=0.7)
    ax.set_rticks([1, 2, 3, 4, 5, 6])

    sc = ax.scatter([], [], s=3, c=[], cmap="plasma", vmin=0, vmax=MAX_D, alpha=0.9)
    cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label("Odleglosc [m]", color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    angles = collections.deque(maxlen=BUF)
    dists  = collections.deque(maxlen=BUF)

    def update(_):
        while point_queue:
            a, d = point_queue.popleft()
            angles.append(a)
            dists.append(d)
        if angles:
            sc.set_offsets(np.column_stack([list(angles), list(dists)]))
            sc.set_array(np.array(list(dists)))
            ax.set_title(
                f"LiDAR D500  |  {len(angles)} pkt  |  OK:{stats['ok']}  FAIL:{stats['fail']}",
                color="white", fontsize=12, pad=18
            )
        return (sc,)

    ani = animation.FuncAnimation(fig, update, interval=100,
                                  blit=True, cache_frame_data=False)
    plt.show()
    stop_flag.set()
    ser.close()
    print(f"[INFO] Koniec. Pakietow OK: {stats['ok']}, fail: {stats['fail']}")

if __name__ == "__main__":
    main()