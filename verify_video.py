from pathlib import Path
from sre_constants import ANY_ALL
import cv2
import arrow
import re


class VerificationResult:
    """
    Data class for　video-verification result text file.
    An example of the text file is as following
    ```

        0:06:38
        0:06:51
        0:07:17

        表示のEnter：25
        表示のLeave：13
        観測のEnter：26
        観測のLeave：15
    ```
    """
    def __init__(self, path):
        self.observed_enter = None
        self.observed_leave = None
        self.displayed_enter = None
        self.displayed_leave = None
        self.times = []
        self.path = path
        self.datetime = arrow.Arrow.strptime(path.stem, r'%Y-%m-%d_%H')
        self._read()

    @staticmethod
    def extract_number(string):
        return int(re.findall(r'\d+', string)[0])

    def _read(self):
        with open(self.path) as f:
            print(f'Read file: {self.path.name}')
            lines = f.readlines()
        lines = map(lambda x: x.strip(), lines)
        lines = filter(lambda x: bool(x), lines)
        for line in lines:
            if '表示のEnter' in line:
                self.observed_enter = self.extract_number(line)
            elif '表示のLeave' in line:
                self.observed_leave = self.extract_number(line)
            elif '観測のEnter' in line:
                self.displayed_enter = self.extract_number(line)
            elif '観測のLeave' in line:
                self.displayed_leave = self.extract_number(line)
            else:
                self.times.append(arrow.Arrow.strptime(line, '%H:%M:%S'))


class DayResult:
    def __init__(self):
        self.observed_enter = 0
        self.observed_leave = 0
        self.displayed_enter = 0
        self.displayed_leave = 0
        self.fail_count = 0

    def add_result(self, verify_result):
        self.observed_enter += verify_result.observed_enter
        self.observed_leave += verify_result.observed_leave
        self.displayed_enter += verify_result.displayed_enter
        self.displayed_leave += verify_result.displayed_leave
        self.fail_count += len(verify_result.times)


def anlyze(verification_results):
    """
    Anlyze VerificationResults and print the summary
    Args:
        verification_results: list of VerificationResult

    """
    day_results = {}
    for r in verification_results:
        if r.datetime.date() not in day_results:
            day_results[r.datetime.date()] = DayResult()
        day_results[r.datetime.date()].add_result(r)

    print('date'.rjust(11) + 'observed_enter'.rjust(20) + 'displayed_enter'.rjust(20) +
          'observed_leave'.rjust(20) + 'displayed_leave'.rjust(20) + 'fail_count'.rjust(15))
    for k, v in day_results.items():
        print(f'{k}'.rjust(11) + f'{v.observed_enter}'.rjust(20) + f'{v.displayed_enter}'.rjust(20) +
              f'{v.observed_leave}'.rjust(20) + f'{v.displayed_leave}'.rjust(20) + f'{v.fail_count}'.rjust(15))


def time2second(t):
    return (t.hour*60+t.minute)*60+t.second


def extract_failed_clip(verify_result, out_dir_name='extracted_videos'):
    """
    Args:
        verify_result: VerifyResult()
        out_dir: pathlib.Path
    """
    if verify_result.times == 0:
        return
    # a window of 5 seconds
    before_window = 7
    after_window = 3
    video_path = verify_result.path.parent.parent / \
        'videos' / (verify_result.path.stem+'.mp4')
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir = verify_result.path.parent.parent / out_dir_name
    out_dir.mkdir(exist_ok=True)
    print(f'frame_count: {frame_count}')
    print(f'fps: {fps}')
    for t in verify_result.times:
        out_path = out_dir / (verify_result.path.stem +
                              f'_{t.strftime("%M_%S")}.mp4')
        start_frame = max(1, time2second(t.time())*fps - before_window*fps)
        end_frame = min(frame_count, time2second(
            t.time())*fps + after_window*fps)
        cap = cv2.VideoCapture(str(video_path))
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4))))
        counter = 1
        ret, frame = cap.read()
        while ret:
            if counter in list(range(start_frame, end_frame)):
                writer.write(frame)
                assert frame.shape == (700, 800, 3)
            ret, frame = cap.read()
            counter += 1
        writer.release()


dataset_root = Path('F:/archived_videos/niko-jyoshiyokuba20210921')
result_dir = dataset_root / 'results'
video_dir = dataset_root / 'videos'


results = []
for path in result_dir.glob('*'):
    results.append(VerificationResult(path))

anlyze(results)

for r in results:
    extract_failed_clip(r)
