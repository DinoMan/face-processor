import argparse
import os
from face_processor import FaceProcessor
import numpy as np
from skimage.viewer import ImageViewer
import progressbar
import skvideo.io
import skimage.io
import multiprocessing as mp


def list_files(folder):
    file_list = []
    for root, dirs, files in os.walk(folder, topdown=False):
        if not dirs and files:
            for f in files:
                file_path = os.path.join(root, f)
                file_list.append(file_path)

    return file_list


def process_video(worker_no, fp, bar, files, landmarks, out_files, queue, rate, window_size, stabilities, mean_stab):
    # Worker loop
    win_size = window_size
    while True:
        task = queue.get()
        if task is not None:
            if stabilities is not None:
                win_size = int(max(window_size * (1 / (1 + np.exp(-(mean_stab - stabilities[task])))), 1))
            try:
                new_video, projected_landmarks = fp.normalise_face(files[task], landmarks[task], window_size=win_size)
            except Exception as e:
                print("Exception Handled: ", e)
                print(files[task])
                queue.task_done()
                continue

            if new_video is None:
                queue.task_done()
                continue

            cropped_video = new_video[:, :crop_height, :crop_width, :]
            vid_out = skvideo.io.FFmpegWriter(out_files[task], inputdict={'-r': rate, }, outputdict={'-r': rate, })
            for frame_no in range(cropped_video.shape[0]):
                vid_out.writeFrame(cropped_video[frame_no])
            vid_out.close()

            queue.task_done()
        else:
            break

        # Only the first worker updates the progress
        if worker_no == 0:
            bar.update(task)

    queue.task_done()


def swap_extension(file, ext):
    return os.path.splitext(file)[0] + ext


parser = argparse.ArgumentParser()

parser.add_argument("--scale", type=float, help="the file extension for the video", default=1.0)
parser.add_argument("--smoothing_window", "-w", type=int, help="The window to use for smoothing", default=7)
parser.add_argument("--file_list", "-f", nargs='?', help="The file list")
parser.add_argument("--stability", type=float, help="file list also has stability metric")
parser.add_argument("--offset", nargs='+', type=float, help="offsets the mean face")
parser.add_argument("--input", "-i", help="folder containing input videos")
parser.add_argument("--landmarks", "-l", help="folder containing landmarks")
parser.add_argument("--output", "-o", help="folder containing output videos")
parser.add_argument("--mean", "-m", nargs='?', help="gets the mean face")
parser.add_argument("--calculate_mean", "-c", action='store_true', help="gets the mean face", default=False)
parser.add_argument("--gpu", "-g", action='store_true', help="uses the gpu for face alignment", default=False)
parser.add_argument("--visualise", "-v", nargs='?', help="visualises the landmarks")
parser.add_argument("--ext_video", help="the file extension for the video")
parser.add_argument("--ext_lmks", default=".csv", help="the file extension for the landmarks")
parser.add_argument("--add_extension", action='store_true', help="appends extension to files", default=False)
parser.add_argument("--picture", "-p", action='store_true', help="processes images", default=False)
parser.add_argument("--append_folder", action='store_true', help="appends input folder to files", default=False)
parser.add_argument("--fill_missing", action='store_true', help="fills in missing landmarks", default=False)
parser.add_argument("--reference", "-r", help="reference image")
parser.add_argument("--workers", type=int, help="number of workers to use")
parser.add_argument("--out_size", nargs='+', type=int, help="offsets the mean face")

args = parser.parse_args()

mean_face = None
reference = None

no_workers = mp.cpu_count()
if args.workers is not None:
    no_workers = min(args.workers, mp.cpu_count())

if args.visualise is not None:
    mean_face = np.load(args.visualise)

    if args.offset is not None:
        FaceProcessor.offset_mean_face(mean_face, offset_percentage=args.offset[:2])

    tl_corner, br_corner = FaceProcessor.find_corners(mean_face)
    canvas = np.zeros([br_corner[1], br_corner[0], 3]).astype(np.uint8)

    FaceProcessor.draw_points(canvas, mean_face, tag=True, in_place=True)
    viewer = ImageViewer(canvas)
    viewer.show()
    exit(0)

if args.mean is not None:
    mean_face = args.scale * np.load(args.mean)

height_border = 0
width_border = 0

if args.offset is not None:
    mean_face = FaceProcessor.offset_mean_face(mean_face, offset_percentage=args.offset[:2])
    height_border += args.offset[1] + args.offset[2]
    width_border += 2 * args.offset[0]

if args.out_size is not None:
    crop_height = args.out_size[0]
    crop_width = args.out_size[1]
elif args.reference is not None:
    reference = skimage.io.imread(args.reference)
    crop_height = reference.shape[0]
    crop_width = reference.shape[1]
else:
    face_width, face_height = FaceProcessor.get_width_height(mean_face)
    crop_height = int(face_height * (1 + height_border))
    crop_width = int(face_width * (1 + width_border))

print("Size will be W: " + str(crop_width) + " H: " + str(crop_height))

fp = FaceProcessor(cuda=args.gpu, mean_face=mean_face, ref_img=args.reference, img_size=(crop_height, crop_width), fill_missing=args.fill_missing)

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

files = []
out_files = []
landmarks = []

queue = mp.JoinableQueue()
progress = 0

files = []

stabilities = None
if args.file_list is None:
    files = list_files(args.input)
else:
    fh = open(args.file_list, "r")
    if args.stability is not None:
        files = []
        stabilities = []
        for f in fh.readlines():
            fn, stab = f.split(",")
            files.append(fn)
            stabilities.append(float(stab))
    else:
        files = fh.readlines()
    fh.close()
    added_ext = ""
    if args.add_extension:
        added_ext = args.ext_video
    if args.append_folder:
        files = [args.input + "/" + f.rstrip('\n') + added_ext for f in files]

for i, f in enumerate(files):
    ext = os.path.splitext(f)[-1]
    landmarks.append(f.replace(args.input, args.landmarks).replace(ext, args.ext_lmks))

    out_path = f.replace(args.input, args.output + "/")
    out_folder = os.path.dirname(os.path.abspath(out_path))
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    out_files.append(out_path)
    queue.put(i)

if args.calculate_mean:
    mean_face = fp.find_mean_face(files)
    np.save("mean_face.npy", mean_face)

bar = progressbar.ProgressBar(max_value=len(files)).start()
rate = FaceProcessor.get_frame_rate(files[0])

workers = []
for i in range(no_workers):
    queue.put(None)  # Place the poison pills for the workers

for i in range(no_workers):
    workers.append(mp.Process(target=process_video,
                              args=((i, fp, bar, files, landmarks, out_files, queue, rate, args.smoothing_window, stabilities, args.stability))))

for worker in workers:
    worker.start()

try:
    queue.join()
except KeyboardInterrupt:
    for worker in workers:
        worker.terminate()

bar.finish()
