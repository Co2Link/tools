import glob
import os

def get_intersection_set(img_dir, ann_dir):
    ann_paths = glob.glob(os.path.join(ann_dir, '*'))
    img_paths = glob.glob(os.path.join(img_dir, '*'))
    print(f'annotation number: {len(ann_paths)}')
    print(f'images number: {len(img_paths)}')
    ann_names = [os.path.splitext(os.path.basename(path))[0] for path in ann_paths]
    img_names = [os.path.splitext(os.path.basename(path))[0] for path in img_paths]
    intersection = set(ann_names) & set(img_names)
    print(f'intersection number: {len(intersection)}')
    for paths in [ann_paths, img_paths]:
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            if name not in intersection:
                os.remove(path)





if __name__ == '__main__':
    img_dir = 'E:/dataset/ofuroba/20210217-015106_0_(100)/images'
    ann_dir = 'E:/dataset/ofuroba/20210217-015106_0_(100)/annotations'
    get_intersection_set(img_dir, ann_dir)
    
