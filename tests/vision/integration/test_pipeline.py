"""from monitoring_taxi.vision.pipeline import VisionPipeline
from tests.vision.test_pipeline import FakeDetector


def test_pipeline_run_pipeline_with_valid_video():
    fake_detector = FakeDetector()
    source = "data/videos/taxi_v0.mp4"

    metrics = VisionPipeline(
        detector=fake_detector,
        target_size=(640,  640)
    ).run(source)

    assert fake_detector.calls == metrics.frames_processed
    assert metrics.frames_processed > 0
    assert metrics.elapsed_time > 0
    assert metrics.preprocessing_time_total >= 0
    assert metrics.inference_time_total >= 0
    assert metrics.postprocessing_time_total >= 0
    assert metrics.processing_fps > 0"""