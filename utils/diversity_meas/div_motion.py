# May 2020
# Optical flow uncertainty

from div_common import *  # contains imports


def measure_motion_magn_video(video_path, results_dir, start_stride=2, delta=4):
    '''
    Calculates the global optical flow (magnitude, angle, x, y) over time.
    The returned dictionary follows the same format as measure_cvae_uncertainty_video().
    Example output = {('ruovish.mp4', 1234): [magn, angle, x, y],
                      ('ruovish.mp4', 1236): [magn, angle, x, y], ...}.
    delta: gap between frames to compare (i.e. frame index i versus i+delta).
    NOTE: The results cover the whole video, there is no delay or stopping in advance, unlike with DPC models.
    '''

    # If available, read cached results using pickle
    results_path = os.path.join(results_dir, 'motion_results.p')
    if results_path is not None and os.path.exists(results_path):
        print('Loading calculated results from ' + results_path + '...')
        with open(results_path, 'rb') as f:
            return pickle.load(f)

    results = dict()
    frames = read_all_frames(video_path)
    n_frames = frames.shape[0]
    for i in tqdm(range(0, n_frames - delta, start_stride)):

        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
        prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(frames[i + delta], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray,
                                            flow=None, pyr_scale=0.5, levels=3, winsize=30,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        # Compute global averages
        x = flow[:, :, 0].mean()
        y = flow[:, :, 1].mean()
        magn = np.sqrt(x**2 + y**2)
        angle = np.arctan2(y, x)

        # Update results
        cur_key = (ntpath.basename(video_path), i)
        results[cur_key] = np.array([magn, angle, x, y])

    # If desired, store results using pickle
    if results_path is not None:
        print('Storing calculated results to ' + results_path + '...')
        if not(os.path.exists(Path(results_path).parent)):
            os.makedirs(Path(results_path).parent)
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

    return results
