import argparse
import os
from processor import face_processor
import numpy as np
from skimage.viewer import ImageViewer
import progressbar
import skvideo.io
import skimage.io
import multiprocessing as mp


def process_video(worker_no, fp, bar, files, landmarks, out_files, queue, rate, window_size):
    # Worker loop
    while True:
        task = queue.get()
        if task is not None:
            new_video, projected_landmarks = fp.normalise_face(files[task], landmarks[task], window_size=window_size)

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
parser.add_argument("--workers", type=int, help="number of workers to use")
parser.add_argument("--out_size", nargs='+', type=int, help="offsets the mean face")

args = parser.parse_args()

mean_face = None
reference = None

no_workers = mp.cpu_count()
if args.workers is not None:
    no_workers = min(args.workers, mp.cpu_count())

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

    face_processor.draw_points(canvas, mean_face, tag=True, in_place=True)
    viewer = ImageViewer(canvas)
    viewer.show()
    exit(0)

if args.mean is not None:
    mean_face = args.scale * np.load(args.mean)

height_border = 0
width_border = 0

if args.offset is not None:
    mean_face = face_processor.offset_mean_face(mean_face, offset_percentage=args.offset[:2])
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
    face_width, face_height = face_processor.get_width_height(mean_face)
    crop_height = int(face_height * (1 + height_border))
    crop_width = int(face_width * (1 + width_border))

print("Size will be W: " + str(crop_width) + " H: " + str(crop_height))

fp = face_processor(cuda=args.gpu, mean_face=mean_face, ref_img=args.reference)

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
out_files = []
landmarks = []

queue = mp.JoinableQueue()
progress = 0
for subject_folder in subject_folder_list:
    if not os.path.exists(args.output + "/" + subject_folder):
        os.makedirs(args.output + "/" + subject_folder)
    for video_file in os.listdir(args.input + "/" + subject_folder + extension):
        if args.ext_video is not None and not video_file.endswith(args.ext_video):
            continue

        files.append(args.input + "/" + subject_folder + extension + video_file)
        if args.landmarks is None:
            landmarks.append(None)
        else:
            landmarks.append(args.landmarks + "/" + subject_folder + extension + swap_extension(video_file, ".csv"))

        out_files.append(args.output + "/" + subject_folder + "/" + video_file)
        queue.put(progress)
        progress += 1

if args.calculate_mean:
    mean_face = fp.find_mean_face(files)
    np.save("mean_face.npy", mean_face)

bar = progressbar.ProgressBar(max_value=len(files)).start()
rate = face_processor.get_frame_rate(files[0])

workers = []
for i in range(no_workers):
    queue.put(None)  # Place the poison pills for the workers

for i in range(no_workers):
    workers.append(mp.Process(target=process_video,
                              args=((i, fp, bar, files, landmarks, out_files, queue, rate, args.smoothing_window))))

for worker in workers:
    worker.start()

queue.join()

bar.finish()
