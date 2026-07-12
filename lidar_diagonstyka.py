"""
DIAGNOSTYKA – sprawdza czy LiDAR wysyla dane przez COM5
Uruchom najpierw TEN skrypt, zeby zobaczyc surowe bajty.
"""
import serial, time

PORT = "COM5"
BAUD = 230400

print(f"Lacze z {PORT}...")
ser = serial.Serial(PORT, BAUD, timeout=2)
time.sleep(0.5)

print(f"Polaczono. Czytam dane przez 3 sekundy...\n")
start = time.time()
total = 0
buf = bytearray()

while time.time() - start < 3:
    n = ser.in_waiting
    if n:
        raw = ser.read(n)
        total += n
        buf.extend(raw)
        hex_str = " ".join(f"{b:02X}" for b in raw[:20])
        print(f"[{total:5d} bajtow] hex: {hex_str}")

ser.close()

if total == 0:
    print("\n[BLAD] Brak danych! Sprawdz:")
    print("  1. Czy LiDAR jest zasilany (silnik sie kreci)?")
    print("  2. TX skanera -> RX adaptera (nie TX->TX!)")
    print("  3. Czy masa (GND) jest polaczona?")
else:
    if b'\xaa\x55' in bytes(buf):
        print(f"\n[OK] Naglowek 0xAA 0x55 znaleziony! Dane sa poprawne.")
    else:
        print(f"\n[UWAGA] Dane sa ({total} bajtow), ale bez naglowka 0xAA 0x55.")
        print("  Mozliwe odwrocone TX/RX lub zly baudrate.")

input("\nNacisnij Enter zeby wyjsc...")