import sys, numpy as np, cv2
sys.path.insert(0, sys.argv[1] + "/sd170")
src = open(sys.argv[1] + "/sd170/proto.py").read().split("SP = Path(sys.argv[1])")[0]
exec(src)
img = cv2.imread(sys.argv[1] + "/lq_std_out/cases/clean_scan/clean_scan-000/image.jpg")
ppm = 12.0; H, W = img.shape[:2]
A = 1.2 * ppm; L = 40 * ppm
xs = np.arange(W, dtype=np.float32); true_dx = A * np.sin(2 * np.pi * xs / L)
mapx = np.tile(xs - true_dx, (H, 1)).astype(np.float32); mapy = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))
warped = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
y0, y1 = int(0.35 * H), int(0.35 * H + 16 * ppm); x0, x1 = int(0.1 * W), int(0.9 * W)
for period in (1.0,):
    for sm in (2.0, 4.0, 8.0):
        dx, q = grid_dx2(warped, y0, y1, x0, x1, ppm, smooth_mm=sm)
        t = true_dx[x0:x1] - np.polyval(np.polyfit(np.arange(x1 - x0), true_dx[x0:x1], 1), np.arange(x1 - x0))
        e = dx - t
        print(f"period {period} smooth {sm}: rms err {np.sqrt(np.mean(e**2))/ppm:.3f} mm, true rms {np.sqrt(np.mean(t**2))/ppm:.3f} mm, corr {np.corrcoef(dx, t)[0,1]:.3f}, amp ratio {q:.2f}")
dx0, _ = grid_dx2(img, y0, y1, x0, x1, ppm)
print("unwarped sheet: dx rms", round(float(np.sqrt(np.mean(dx0**2)))/ppm, 3), "mm")
