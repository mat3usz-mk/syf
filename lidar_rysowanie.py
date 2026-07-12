"""
LiDAR D500 / STL-19P – wizualizacja 360° w czasie rzeczywistym
Działa na Windows przez USB (adapter CP2102 / CH340)
Wymagania: pip install pyserial numpy matplotlib
"""

import serial
import serial.tools.list_ports
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ─────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────
PORT  = "AUTO"       # "AUTO" = wykryj sam, albo wpisz np. "COM3"
BAUD  = 230400
MAX_DIST_M = 6.0     # maksymalny zasięg wyświetlania [m]
POINTS_BUF = 2500    # ile punktów trzymać w buforze (ok. 5 obrotów)


# ─────────────────────────────────────────
# WYKRYWANIE PORTU
# ─────────────────────────────────────────
def find_lidar_port():
    keywords = ["CP210", "CH340", "USB Serial", "USB-SERIAL", "UART"]
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or "") + (p.manufacturer or "")
        if any(kw.lower() in desc.lower() for kw in keywords):
            print(f"[OK] Znaleziono adapter: {p.device}  →  {desc}")
            return p.device
    # fallback – pokaż wszystkie porty i poproś o wybór
    print("\n[!] Nie wykryto adaptera automatycznie.")
    print("Dostępne porty COM:")
    for p in ports:
        print(f"    {p.device}  –  {p.description}")
    choice = input("\nWpisz port ręcznie (np. COM3): ").strip()
    return choice


# ─────────────────────────────────────────
# PARSOWANIE PAKIETÓW STL-19P
# ─────────────────────────────────────────
HEADER = b'\xaa\x55'

def parse_packet(buf):
    """Zwraca listę (kąt_rad, odległość_m) z jednego pakietu."""
    points = []
    if len(buf) < 10:
        return points
    start_deg = (buf[6] | (buf[7] << 8)) / 100.0
    end_deg   = (buf[8] | (buf[9] << 8)) / 100.0
    n = (len(buf) - 10) // 2
    if n < 1:
        return points
    diff = (end_deg - start_deg) % 360.0
    step = diff / (n - 1) if n > 1 else 0.0
    for i in range(n):
        angle_deg = (start_deg + i * step) % 360.0
        idx = 10 + i * 2
        if idx + 1 >= len(buf):
            break
        dist_mm = buf[idx] | (buf[idx + 1] << 8)
        if 30 < dist_mm < 12000:
            points.append((math.radians(angle_deg), dist_mm / 1000.0))
    return points


def serial_reader(ser):
    """Generator – czyta ciągłe bajty z UART i zwraca sparsowane punkty."""
    buf = bytearray()
    while True:
        waiting = ser.in_waiting
        if waiting:
            buf.extend(ser.read(waiting))
        # znajdź nagłówek
        idx = buf.find(HEADER)
        if idx == -1:
            if len(buf) > 256:
                buf = buf[-10:]
            continue
        buf = buf[idx:]
        if len(buf) < 4:
            continue
        # liczba punktów zakodowana w bajcie 3
        n_pts = buf[3] if buf[2] == 0x00 else 0
        pkt_len = 10 + n_pts * 2
        if len(buf) < pkt_len:
            continue
        for pt in parse_packet(buf[:pkt_len]):
            yield pt
        buf = buf[pkt_len:]


# ─────────────────────────────────────────
# WIZUALIZACJA (polar plot)
# ─────────────────────────────────────────
def main():
    port = PORT if PORT != "AUTO" else find_lidar_port()
    print(f"[INFO] Łączę z {port} @ {BAUD} baud...")

    try:
        ser = serial.Serial(port, BAUD, timeout=1)
        print(f"[OK] Połączono! Zamknij okno wykresu, żeby zakończyć.\n")
    except serial.SerialException as e:
        print(f"\n[BŁĄD] Nie można otworzyć portu: {e}")
        print("Sprawdź: (1) czy adapter jest podłączony, (2) czy właściwy port COM,")
        print("         (3) czy żaden inny program nie używa portu.")
        input("\nNaciśnij Enter, żeby wyjść...")
        return

    reader = serial_reader(ser)
    angles_buf, dists_buf = [], []

    # --- setup wykresu ---
    plt.style.use("dark_background")
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(9, 9))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")
    ax.set_ylim(0, MAX_DIST_M)
    ax.set_title(
        "LiDAR D500 STL-19P  –  mapa 360°",
        color="white", fontsize=13, pad=18
    )
    ax.tick_params(colors="#555577")
    ax.grid(color="#222244", linewidth=0.8)

    # siatka odległości co 1 m
    ax.set_rticks([1, 2, 3, 4, 5, 6])
    ax.set_rlabel_position(22.5)

    scatter = ax.scatter(
        [], [], s=2,
        c=[], cmap="plasma",
        vmin=0, vmax=MAX_DIST_M,
        alpha=0.85
    )

    # pasek kolorów
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label("Odległość [m]", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    info_text = ax.text(
        0, MAX_DIST_M * 1.12, "",
        ha="center", color="#aaaacc", fontsize=9
    )

    def update(_frame):
        # zbierz nowe punkty
        for _ in range(150):
            try:
                a, d = next(reader)
                angles_buf.append(a)
                dists_buf.append(d)
            except StopIteration:
                break

        # ogranicz bufor
        if len(angles_buf) > POINTS_BUF:
            del angles_buf[:-POINTS_BUF]
            del dists_buf[:-POINTS_BUF]

        if angles_buf:
            data = np.column_stack([angles_buf, dists_buf])
            scatter.set_offsets(data)
            scatter.set_array(np.array(dists_buf))
            info_text.set_text(f"Punktów w buforze: {len(angles_buf)}")

        return scatter, info_text

    ani = animation.FuncAnimation(
        fig, update, interval=80, blit=True, cache_frame_data=False
    )

    plt.tight_layout()
    try:
        plt.show()
    finally:
        ser.close()
        print("[INFO] Port zamknięty.")


if __name__ == "__main__":
    main()