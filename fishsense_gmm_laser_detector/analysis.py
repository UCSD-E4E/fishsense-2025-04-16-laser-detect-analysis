'''Analysis
'''
import logging
import queue
import signal
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from typing import List, Tuple

import cv2 as cv
import numpy as np
import requests

from fishsense_gmm_laser_detector.config import configure_logging, settings
from fishsense_gmm_laser_detector.gmm_laser_detector import GMMLaserDetector


class Analyzer:
    def __init__(self):
        self.__e4efs_headers = {
            'api_key': settings.e4efs.api_key
        }
        self.stop_event = Event()

        self._detector = GMMLaserDetector.initialize_from_json(settings.detector.path)
        self._log = logging.getLogger('Analyzer')
        self._output_path = Path(settings.detector.output)

    def _get_dive_list(self, session: requests.Session) -> List[str]:
        req = session.get('https://orchestrator.fishsense.e4e.ucsd.edu/api/v1/metadata/dives')
        req.raise_for_status()
        doc = req.json()
        return doc['dives']

    def _get_frame_list(self, dive: str, session: requests.Session) -> List[str]:
        req = session.get(f'https://orchestrator.fishsense.e4e.ucsd.edu/api/v1/metadata/dive/{dive}')
        req.raise_for_status()
        doc = req.json()
        return doc['frames']

    def _get_frames_loop(self,
                         dive_queue: queue.Queue[str],
                         frame_queue: queue.Queue[str],
                         session: requests.Session
                         ):
        while not self.stop_event.is_set():
            try:
                dive = dive_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                frames = self._get_frame_list(dive=dive, session=session)
            except requests.RequestException:
                dive_queue.task_done()
                continue
            for frame in frames:
                frame_queue.put(frame)
            dive_queue.task_done()

    def _get_assets(self, frame: str, jpeg_path: Path, session: requests.Session) -> Tuple[int, int, Path]:
        req = session.get(f'https://orchestrator.fishsense.e4e.ucsd.edu/api/v1/data/laser/{frame}')
        req.raise_for_status()
        laser_label = req.json()

        req = session.get(f'https://orchestrator.fishsense.e4e.ucsd.edu/api/v1/data/preprocess_jpeg/{frame}')
        req.raise_for_status()
        output_path = jpeg_path / f'{frame}.jpg'
        with open(output_path, 'wb') as handle:
            handle.write(req.content)
        return (laser_label['x'], laser_label['y'], output_path)

    def _get_assets_loop(self,
                         frame_queue: queue.Queue[str],
                         asset_queue: queue.Queue[Tuple[str, Tuple[int, int], Path]],
                         jpeg_path: Path,
                         session: requests.Session
                         ):
        while not self.stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                result = self._get_assets(frame, jpeg_path, session)
            except requests.RequestException:
                frame_queue.task_done()
                continue
            asset_queue.put((frame, result[0:2], result[2]))
            frame_queue.task_done()

    def _process_image(self,
                       img_path: Path,
                       laser_position: Tuple[int, int]
                       ) -> Tuple[float, float, int]:
        img = cv.imread(img_path)
        # denoise
        img_bgr_denoised = cv.fastNlMeansDenoisingColored(img, None, 6, 6, 4, 10)
        img_bgr = cv.bilateralFilter(img_bgr_denoised, 10, 75, 75)
        result = self._detector.find_laser(img_bgr)
        if not result:
            return (np.nan, np.nan, np.inf)
        dist = np.linalg.norm(np.array(laser_position) - np.array(result))
        return (result[0], result[1], dist)

    def _process_image_loop(self,
                       asset_queue: queue.Queue[Tuple[str, Tuple[int, int], Path]],
                       output_queue: queue.Queue[Tuple[str, Tuple[float, float], float]]
                       ):
        while not self.stop_event.is_set():
            try:
                cksum, laser_position, jpeg_path = asset_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                result = self._process_image(
                    img_path=jpeg_path,
                    laser_position=laser_position
                )
            except Exception as exc:
                self._log.exception('Image processing failed! %s', exc)
                asset_queue.task_done()
                continue
            output_queue.put((
                cksum,
                (result[0], result[1]),
                result[2]
            ))
            jpeg_path.unlink()
            asset_queue.task_done()

    def _accumulate_results_loop(self,
                                 output_queue: queue.Queue[Tuple[str, Tuple[float, float], float]]):
        with open(self._output_path, 'w', encoding='utf-8') as handle:
            handle.write(f'cksum,x,y,distance\n')

        while not self.stop_event.is_set():
            try:
                cksum, estimate, distance = output_queue.get(timeout=1)
            except queue.Empty:
                continue
            with open(self._output_path, 'a', encoding='utf-8') as handle:
                handle.write(f'{cksum},{estimate[0]},{estimate[1]},{distance}\n')
            output_queue.task_done()

    def run(self, workdir: Path, session: requests.Session):
        session.headers.update(self.__e4efs_headers)
        dive_list = self._get_dive_list(session=session)
        dive_queue = queue.Queue(maxsize=32)
        frame_queue = queue.Queue(maxsize=32)
        asset_queue = queue.Queue(maxsize=32)
        output_queue = queue.Queue(maxsize=32)

        get_frame_thread = Thread(
            target=self._get_frames_loop,
            kwargs={
                'dive_queue': dive_queue,
                'frame_queue': frame_queue,
                'session': session
            }
        )
        jpeg_path = workdir / 'jpeg'
        jpeg_path.mkdir(parents=True, exist_ok=True)
        get_asset_thread = Thread(
            target=self._get_assets_loop,
            kwargs={
                'frame_queue': frame_queue,
                'asset_queue': asset_queue,
                'jpeg_path': jpeg_path,
                'session': session
            }
        )
        process_image_threads = [
            Thread(
                target=self._process_image_loop,
                kwargs={
                    'asset_queue': asset_queue,
                    'output_queue': output_queue
                }
            )
            for _ in range(1)
        ]
        result_thread = Thread(
            target=self._accumulate_results_loop,
            kwargs={
                'output_queue': output_queue
            }
        )
        result_thread.start()
        for thread in process_image_threads:
            thread.start()
        get_asset_thread.start()
        get_frame_thread.start()
        for dive in dive_list:
            if not self.stop_event.is_set():
                dive_queue.put(dive)
        dive_queue.join()
        frame_queue.join()
        asset_queue.join()
        output_queue.join()

    def stop(self, *_, **__):
        self.stop_event.set()

        
def main():
    """Main thread
    """
    configure_logging()
    with TemporaryDirectory() as workdir, requests.Session() as session:
        Analyzer().run(
            workdir=Path(workdir),
            session=session
        )

if __name__ == '__main__':
    main()