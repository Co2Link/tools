import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from queue import Queue
import cv2
import pickle
import os
import glob
import shutil


class PtsGetter:
    def __init__(self, frame):
        """
        Args:
            frame: cvimage 
        """
        self.frame = frame
        self.MOUST_PTS = []
        self.windowsname = 'get_pts'
        self.is_finished = False

    def run(self):
        """
        Return:
            self.MOUST_PTS: list of tuple
        """
        cv2.namedWindow(self.windowsname)
        cv2.setMouseCallback(self.windowsname, self._get_mouse_points)
        while True:
            cv2.imshow(self.windowsname, self.frame)
            cv2.waitKey(1)
            if self.is_finished:
                cv2.destroyWindow(self.windowsname)
                break
        return self.MOUST_PTS

    def _get_mouse_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.frame, (x, y), 5, (255, 0, 0), 10)
            if len(self.MOUST_PTS) > 0:
                cv2.line(self.frame, (x, y),
                         self.MOUST_PTS[-1], (70, 70, 70), 2)
            self.MOUST_PTS.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.line(self.frame, self.MOUST_PTS[-1],
                     self.MOUST_PTS[0], (70, 70, 70), 2)
            pts = np.array(self.MOUST_PTS).astype('int32')
            cv2.fillPoly(self.frame, [pts], 255)
            self.is_finished = True


def analyze(video_path, draw_roi=True, mask_path=None, is_analyze=False, out_dir=None):
    # check param
    if not is_analyze:
        assert out_dir is not None
    maxsize = 5
    frame_idxs = []
    count = 1
    have_roi = draw_roi or mask_path is not None
    buf = Queue(maxsize=maxsize)
    cap = cv2.VideoCapture(video_path)
    H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mask_path is not None:
        mask = np.load(mask_path)
        if is_analyze:
            display_frame = cv2.addWeighted(frame_gray, 0.8, mask*255, 0.2, 0)
            cv2.imshow('display', display_frame)
            cv2.waitKey(0)
            cv2.destroyWindow('display')
    elif draw_roi:
        pts_getter = PtsGetter(frame_gray.copy())
        print('draw the roi')
        pts = pts_getter.run()
        mask = Image.new('L', (W, H), 0)
        ImageDraw.Draw(mask).polygon(pts, outline=1, fill=1)
        mask = np.array(mask)
        with open('mask.npy', 'wb') as f:
            np.save(f, mask)
    while ret:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = frame_gray.astype('int32')
        if buf.full():
            old_frame_gray = buf.get()
            buf.put(frame_gray)
            if have_roi:
                diff = np.abs(old_frame_gray - frame_gray)[mask == True]
            else:
                diff = np.abs(old_frame_gray - frame_gray).ravel()
            if is_analyze:
                frame_display = frame_gray.copy()
                text = '50: {:<3}, 100: {:<3}, 150: {:<3}'.format(
                    diff[diff > 50].shape[0], diff[diff > 100].shape[0], diff[diff > 150].shape[0])
                cv2.putText(frame_display, text, (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('analyze', frame_display.astype('uint8'))
                cv2.waitKey(0)
            else:
                if diff[diff > 100].shape[0] > 110:
                    frame_idxs.append(count)
                    print(count)
        else:
            buf.put(frame_gray)
        ret, frame = cap.read()
        count += 1
    cap.release()

    sections = []
    sect = [frame_idxs[0], None]
    p = sect[0]
    for i in frame_idxs[1:]:
        if i - p > 5:
            sect[1] = p
            sections.append(sect)
            sect = [i, None]
        p = i

    # remove section that less that 15 frame
    sections = [sect for sect in sections if (sect[1] - sect[0]) > 15]

    # add 5 frame at each section front and end
    for sect in sections:
        sect[0] -= 5
        sect[1] += 5

    write_frame_idx = []
    for sect in sections:
        write_frame_idx.extend(range(sect[0], sect[1]))

    video_name = os.path.basename(video_path)
    writer = cv2.VideoWriter(os.path.join(
        out_dir, video_name), cv2.VideoWriter_fourcc(*'mp4v'), 15, (W, H))
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    counter = 1
    while ret:
        if counter in write_frame_idx:
            writer.write(frame)
        ret, frame = cap.read()
        counter += 1
    cap.release()
    writer.release()


# video_path = 'H:/backup/rpi-2/2021-03-26_21.mp4'
# # video_path = 'D:/local/tools/output-1/2021-03-27_15.mp4'
# analyze(video_path, out_dir='out-test', is_analyze=True)

if __name__ == "__main__":
    video_paths = glob.glob('H:/backup/rpi-2/*')
    video_paths.sort()
    out_dir = 'output-2'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    for path in video_paths:
        print(path)
        analyze(path, mask_path='mask-2.npy', out_dir=out_dir)


