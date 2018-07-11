# face-processor
The face processor provides the neccessary tools for aligning faces. It can be used to align faces in both images and videos (only one face per frame).

## Aligning faces in images
python main.py -i pics_folder -m mean_face -p -o out_folder --offset 0.11 0.335 0.155

## Aligning faces in videos
python main.py -i video_folder -m mean_face.npy -o out_folder -l landmarks_file -w smoothing_window -s scale --offset 0.11 0.336 0.155
