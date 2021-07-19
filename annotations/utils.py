import glob
import os
import cv2
import glob
import shutil
from tqdm import tqdm
import json


def get_intersection_set(img_dir, ann_dir):
    ann_paths = glob.glob(os.path.join(ann_dir, '*'))
    img_paths = glob.glob(os.path.join(img_dir, '*'))
    print(f'annotation number: {len(ann_paths)}')
    print(f'images number: {len(img_paths)}')
    ann_names = [os.path.splitext(os.path.basename(path))[0]
                 for path in ann_paths]
    img_names = [os.path.splitext(os.path.basename(path))[0]
                 for path in img_paths]
    intersection = set(ann_names) & set(img_names)
    print(f'intersection number: {len(intersection)}')
    for paths in [ann_paths, img_paths]:
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            if name not in intersection:
                os.remove(path)


def rename_voc(img_dir, annotation_path, suffix):
    img_paths = glob.glob(os.path.join(img_dir, '*'))
    for path in tqdm(img_paths):
        name, ext = os.path.splitext(path)
        new_path = name + '_' + suffix + ext
        os.rename(path, new_path)
    with open(annotation_path) as f:
        data = json.load(f)
        for im in tqdm(data['images']):
            name = im['file_name']
            name, ext = os.path.splitext(name)
            new_name = name + '_' + suffix + ext
            im['file_name'] = new_name
    new_annotation_path = os.path.splitext(
        annotation_path)[0] + '_modified' + '.json'
    with open(new_annotation_path, 'w') as f:
        json.dump(data, f)


def videos_to_frame(video_dir, out_dir, ratio=1):
    """
    Args:
        folder: pathlib.Path
        out_dir: pathlib.Path
        ratio: keep 1 frame for every ratio frames
    """
    paths = video_dir.glob('*')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    count = 0
    img_count = 0
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        ret, frame = cap.read()
        while ret:
            if count % ratio == 0:
                # cv2.imwrite(os.path.join(output_folder, '{:06d}.png'.format(img_count)), frame)
                cv2.imwrite(str(out_dir / f'{img_count:06d}.png'), frame)
                img_count += 1
            count += 1
            ret, frame = cap.read()


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


if __name__ == '__main__':
    img_dir = 'E:/dataset/ofuroba/20210217-015106_0_(100)/images'
    ann_dir = 'E:/dataset/ofuroba/20210217-015106_0_(100)/annotations'
    get_intersection_set(img_dir, ann_dir)
