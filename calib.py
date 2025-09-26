from calib_utils import calibrate_single_camera_npz, load_calibration_npz

def main():
    # 10x7 squares = 9x6 inner corners, 30 mm squares
    rows, cols, square = 6, 9, 0.03
    calibrate_single_camera_npz(0, rows, cols, square, samples=15, save_path="calib_cam0.npz")
    calibrate_single_camera_npz(2, rows, cols, square, samples=15, save_path="calib_cam2.npz")

    K0,D0 = load_calibration_npz("calib_cam0.npz")
    K2,D2 = load_calibration_npz("calib_cam2.npz")
    print("Cam0 K, D:", K0, D0)
    print("Cam2 K, D:", K2, D2)

if __name__=="__main__":
    main()