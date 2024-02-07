import sys
import math
import cv2
import rawpy
import imageio

# scale factor
SCALE_FACTOR = 1
AUTO_BRIGHT_OFF = False
AUTO_BRIGHT_THR = 0.001

# read and postprocess the raw image
def processRAW(path, scale_factor=SCALE_FACTOR, auto_bright_off=AUTO_BRIGHT_OFF, auto_bright_thr=AUTO_BRIGHT_THR):
    # read the file
    with rawpy.imread(path) as raw_file:
        num_bits = int(math.log(raw_file.white_level + 1, 2))
        rgb = raw_file.postprocess(gamma=(1,1), no_auto_bright=auto_bright_off, auto_bright_thr=auto_bright_thr, output_bps=16, use_camera_wb=True)

        resized_rgb = rgb

        # optionally reduce the size
        if scale_factor != 1:
            dim = (rgb.shape[1] // scale_factor, rgb.shape[0] // scale_factor)
            resized_rgb = cv2.resize(rgb, dim, interpolation=cv2.INTER_AREA)

        return resized_rgb


def main(argv):
    if len(argv) < 2:
        print("Usage: python %s <image path>" % (argv[0]))
        return

    for filename in argv[1:]:
        suffix = filename.split(".")[-1].lower()
        if suffix not in ['dng', 'cr2']:
            print(f"Input image {filename} is not a supported raw image file.")
            continue

        print(f"Processing {filename}")
        rgb = processRAW(filename)

        # save as a 16-bit TIFF
        words = filename.rsplit(".", 1)
        newfilename = words[0] + ".tif"

        print(f"Writing {newfilename}")
        imageio.imsave(newfilename, rgb)

    print("Terminating")

if __name__ == "__main__":
    main(sys.argv)
