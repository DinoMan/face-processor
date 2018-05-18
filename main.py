import argparse
import os
from processor import face_processor
import numpy as np
from skimage.viewer import ImageViewer
import progressbar
import skvideo.io
import skimage.io


def swap_extension(file, ext):
    return os.path.splitext(file)[0] + ext


parser = argparse.ArgumentParser()

parser.add_argument("--scale", type=float, help="the file extension for the video", default=1.0)
parser.add_argument("--smoothing_window", "-w", type=int, help="The window to use for smoothing", default=7)
parser.add_argument("--offset", nargs='+', type=float, help="offsets the mean face")
parser.add_argument("--input", "-i", help="folder containing input videos")
parser.add_argument("--landmarks", "-l", help="folder containing landmarks")
parser.add_argument("--output", "-o", help="folder containing output videos")
parser.add_argument("--subjects", "-s", nargs='+', help="the specific subjects to obtain")
parser.add_argument("--mean", "-m", nargs='?', help="gets the mean face")
parser.add_argument("--calculate_mean", "-c", action='store_true', help="gets the mean face", default=False)
parser.add_argument("--gpu", "-g", action='store_true', help="uses the gpu for face alignment", default=False)
parser.add_argument("--visualise", "-v", nargs='?', help="visualises the landmarks")
parser.add_argument("--ext_video", help="the file extension for the video")
parser.add_argument("--append_extension", "-a", help="path to append after subject")
parser.add_argument("--picture", "-p", action='store_true', help="processes images", default=False)
parser.add_argument("--reference", "-r", help="reference image")

args = parser.parse_args()

mean_face = None
reference = None

if args.append_extension is None:
    extension = "/"
else:
    extension = "/" + args.append_extension + "/"

if args.visualise is not None:
    mean_face = np.load(args.visualise)

    if args.offset is not None:
        face_processor.offset_mean_face(mean_face, offset_percentage=args.offset[:2])

    tl_corner, br_corner = face_processor.find_corners(mean_face)
    canvas = np.zeros([br_corner[1], br_corner[0], 3]).astype(np.uint8)

    face_processor.draw_points(canvas, mean_face, tag=False, in_place=True)
    viewer = ImageViewer(canvas)
    viewer.show()
    exit(0)

if args.mean is not None:
    mean_face = args.scale * np.load(args.mean)

if args.offset is not None:
    mean_face = face_processor.offset_mean_face(mean_face, offset_percentage=args.offset[:2])

if args.reference is not None:
    reference = skimage.io.imread(args.reference)
    crop_height = reference.shape[0]
    crop_width = reference.shape[1]
else:
    face_width, face_height = face_processor.get_width_height(mean_face)
    crop_height = int(face_height * (1 + args.offset[1] + args.offset[2]))
    crop_width = int(face_width * (1 + 2 * args.offset[0]))

print("Size will be W: " + str(crop_width) + " H: " + str(crop_height))

fp = face_processor(cuda=args.gpu, mean_face=mean_face, ref_img=reference)

if args.picture:
    pictures = os.listdir(args.input)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    for picture in pictures:
        img = skimage.io.imread(args.input + "/" + picture)
        warped_img = fp.process_image(img)
        if warped_img is not None:
            skimage.io.imsave(args.output + "/" + picture, warped_img[:crop_height, :crop_width, :])

    exit()

if args.subjects is None:
    subject_folder_list = os.listdir(args.input)
else:
    subject_folder_list = [str(s) for s in args.subjects]

files = []
landmarks = []
out_files = []
for subject_folder in subject_folder_list:
    if not os.path.exists(args.output + "/" + subject_folder):
        os.makedirs(args.output + "/" + subject_folder)
    for video_file in os.listdir(args.input + "/" + subject_folder + extension):
        if args.ext_video is not None and not video_file.endswith(args.ext_video):
            continue
        files.append(args.input + "/" + subject_folder + extension + video_file)
        landmarks.append(args.landmarks + "/" + subject_folder + extension + swap_extension(video_file, ".csv"))
        out_files.append(args.output + "/" + subject_folder + "/" + video_file)

if args.calculate_mean:
    mean_face = fp.find_mean_face(files)
    np.save("mean_face.npy", mean_face)

bar = progressbar.ProgressBar(max_value=len(files)).start()

rate = face_processor.get_frame_rate(files[0])
for i, file in enumerate(files):
    new_video = fp.normalise_face(file, landmarks[i], window_size=args.smoothing_window)
    bar.update(i + 1)
    if new_video is None:
        continue

    cropped_video = new_video[:, :crop_height, :crop_width, :]
    skvideo.io.vwrite(out_files[i], cropped_video)

bar.finish()
