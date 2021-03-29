import cv2
import os
import glob
import shutil

def reduce_fps(video_path, out_path_suffix, ratio):
    assert os.path.isfile(video_path), f"{video_path} do not exist"
    assert isinstance(ratio, int), "ratio must be integer"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_fps = fps/ratio
    basename = os.path.basename(video_path)
    name, _ = os.path.splitext(basename)
    out_path = f"{name}_{out_path_suffix}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h))
    
    ret, frame = cap.read()
    count = 0
    while ret:
        if count % ratio == 0:
            writer.write(frame)
        ret, frame = cap.read()
        count += 1
    cap.release()
    writer.release()

def video_to_frame(video_path, ratio):
    assert os.path.isfile(video_path), f"{video_path} do not exist"
    cap = cv2.VideoCapture(video_path)
    basename = os.path.basename(video_path)
    name, ext = os.path.splitext(basename)
    save_dir = f"{name}_({ratio})"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    ret, frame = cap.read()    
    count = 0
    while ret:
        if count % ratio == 0:
            cv2.imwrite(os.path.join(save_dir, '{:06d}.png'.format(count)), frame)
        ret, frame = cap.read()
        count += 1

def videos_to_frame(folder, ratio):
    paths = glob.glob(os.path.join(folder, '*'))
    save_dir = folder + '_' + str(ratio)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    count = 0
    img_count = 0
    for path in paths:
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        while ret:
            if count % ratio == 0:
                cv2.imwrite(os.path.join(save_dir, '{:06d}.png'.format(img_count)), frame)
                img_count += 1
            count += 1
            ret, frame = cap.read()


if __name__ == "__main__":
    # paths = glob.glob("F:/raspi-2/extracted/*")
    # for path in paths:
    #     reduce_fps(path, '7.5', 2)

    # paths = glob.glob('output-2/*')
    # paths.sort()
    # for path in paths:
    #     video_to_frame(path, 2)


    videos_to_frame('output-1', 5)
    videos_to_frame('output-2', 5)