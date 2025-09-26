def transform(x_w_mm, y_w_mm):
    # Keep inputs in mm (since coefficients are given for mm units)
    x_w = x_w_mm
    y_w = y_w_mm

    # Apply equations
    x_r = (-0.887 * x_w**2
           +0.889 * y_w**2
           +1.714 * x_w * y_w
           -1.007 * x_w
           +1.978 * y_w
           +0.096)

    y_r = (0.287 * x_w**2
           +1.371 * y_w**2
           -0.769 * x_w * y_w
           +0.758 * x_w
           -1.080 * y_w
           -0.284)

    return x_r, y_r


if __name__ == "__main__":
    x_w_mm = float(input("Enter x_w (mm): "))
    y_w_mm = float(input("Enter y_w (mm): "))
    x_r, y_r = transform(x_w_mm, y_w_mm)

    print(f"x_r = {x_r:.3f}")
    print(f"y_r = {y_r:.3f}")



