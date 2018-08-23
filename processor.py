import face_alignment
import numpy as np
import skvideo.io
import cv2
import csv
from skimage import transform as tf

stablePntsIDs = [33, 36, 39, 42, 45]


class face_processor():
    def __init__(self, mean_face=None, ref_img=None, cuda=True):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=cuda, flip_input=False)

        if ref_img is not None:
            tmp_fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=False, flip_input=False)
            self.mean_face = tmp_fa.get_landmarks(ref_img)[0]
        else:
            if isinstance(mean_face, str):
                self.mean_face = np.load(mean_face)
            else:
                self.mean_face = mean_face

    def process_image(self, image):
        try:
            landmarks = self.fa.get_landmarks(image)[0]
        except:
            return None
        stable_points = landmarks[stablePntsIDs, :]
        warped_img, trans = self.warp_img(stable_points,
                                          self.mean_face[stablePntsIDs, :],
                                          image)
        return warped_img

    def normalise_face(self, video_input, landmarks_input=None, window_size=7):

        # Check if we should read from file
        if isinstance(video_input, str):
            video = skvideo.io.vread(video_input)
        else:
            video = video_input

        if window_size % 2 == 0:
            window_size += 1

        # If we have landmarks
        if landmarks_input is not None:
            if isinstance(landmarks_input, str):
                landmarks = self.parse_landmarks_file(landmarks_input)
            else:
                landmarks = landmarks_input

        if video.shape[0] < window_size or len(landmarks) == 0:
            return None

        trans = None
        projected_landmarks = []
        for frame_no in range(0, video.shape[0]):
            if frame_no + window_size < video.shape[0]:
                avg_stable_points = np.zeros([len(stablePntsIDs), 2])
                for i in range(0, window_size):
                    if landmarks_input is None:
                        avg_stable_points += self.fa.get_landmarks(video[frame_no + i])[0][stablePntsIDs, :]
                    else:
                        avg_stable_points += landmarks[frame_no + i][stablePntsIDs, :]

                avg_stable_points /= window_size
                video[frame_no], trans = self.warp_img(avg_stable_points,
                                                       self.mean_face[stablePntsIDs, :],
                                                       video[frame_no])

            else:
                video[frame_no] = self.apply_transform(trans, video[frame_no])

            projected_landmarks.append(trans(landmarks[frame_no]))

        return video, projected_landmarks

    @staticmethod
    def apply_transform(transform, img):
        warped = tf.warp(img, inverse_map=transform.inverse)
        warped = warped * 255  # note output from wrap is double image (value range [0,1])
        warped = warped.astype('uint8')
        return warped

    @staticmethod
    def warp_img(src, dst, img):
        tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
        warped = tf.warp(img, inverse_map=tform.inverse)  # wrap the frame image
        warped = warped * 255  # note output from wrap is double image (value range [0,1])
        warped = warped.astype('uint8')
        return warped, tform

    @staticmethod
    def draw_points(image, points, tag=True, in_place=False, color=(255, 255, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX

        if in_place:
            img = image
        else:
            img = np.copy(image)

        for i in range(points.shape[0]):
            if tag:
                cv2.putText(img, str(i), (int(points[i, 0]), int(points[i, 1])), font, 0.23, color)
            else:
                cv2.circle(img, (int(points[i, 0]), int(points[i, 1])), 1, color)

        return img

    @staticmethod
    def get_width_height(points):
        tl_corner, br_corner = face_processor.find_corners(points)

        width = (br_corner - tl_corner)[0]
        height = (br_corner - tl_corner)[1]

        return width, height

    @staticmethod
    def find_corners(points):
        tl = np.array([points[:, 0].min(), points[:, 1].min()])
        br = np.array([points[:, 0].max(), points[:, 1].max()])

        return tl.astype(int), br.astype(int)

    @staticmethod
    def get_frame_rate(video_file):
        return skvideo.io.ffprobe(video_file)["video"]["@r_frame_rate"]

    @staticmethod
    def parse_landmarks_file(landmarks_file):
        video_landmarks = []
        with open(landmarks_file, 'rt', encoding="ascii") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')

            for frame_no, landmarks in enumerate(csvreader):
                frame_landmarks = np.zeros([68, 2])
                for point in range(1, len(landmarks), 2):
                    frame_landmarks[point // 2, 0] = int(landmarks[point + 1])
                    frame_landmarks[point // 2, 1] = int(landmarks[point])

                    if int(landmarks[point]) == -1:
                        return []
                video_landmarks.append(frame_landmarks)

        return video_landmarks

    @staticmethod
    def offset_mean_face(mean_landmarks, offset_percentage=[0, 0]):
        tl_corner, br_corner = face_processor.find_corners(mean_landmarks)

        width = (br_corner - tl_corner)[0]
        height = (br_corner - tl_corner)[1]

        offset = np.array([int(offset_percentage[0] * width), int(offset_percentage[1] * height)])
        return mean_landmarks - tl_corner + offset

    def find_mean_face(self, files, offset_percentage=[0, 0]):
        mean_landmarks = np.zeros([68, 2])
        number_of_faces = 0
        for video_file in files:
            video = skvideo.io.vread(video_file)
            for frame_no in range(0, video.shape[0], 10):
                number_of_faces += 1
                mean_landmarks += self.fa.get_landmarks(video[frame_no])[0]

            if number_of_faces == 1000:
                break

        mean_face = np.multiply(1 / number_of_faces, mean_landmarks)

        self.mean_face = self.offset_mean_face(mean_face, offset_percentage)

        return self.mean_face
