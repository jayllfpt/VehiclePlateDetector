import cv2
from lib_detection import load_model, detect_lp, im2single

img_path = "test_img/xemay1.jpg"

# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

Ivehicle = cv2.imread(img_path)
cv2.imshow("Anh goc",Ivehicle)
cv2.waitKey(0)

Dmax = 608
Dmin = 288

ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
side = int(ratio * Dmin)
bound_dim = min(side, Dmax)

_ , LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)


if (len(LpImg)):

    cv2.imshow("Bien so", cv2.cvtColor(LpImg[0],cv2.COLOR_RGB2BGR ))
    cv2.waitKey()

cv2.destroyAllWindows()