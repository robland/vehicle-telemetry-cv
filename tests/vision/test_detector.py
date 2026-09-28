from monitoring_taxi.vision.detector import PersonDetector


class FakeBoxes:
    def __init__(self, cls, conf, xyxy):
        self.cls = cls
        self.conf = conf
        self.xyxy = xyxy


class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class FakeYOLO:
    def __init__(self, results=None):
        self.called = False
        self.received_frame = None
        self.received_imgsz = None
        self.received_conf = None
        self.results = results if results is not None else []

    def predict(self, frame, imgsz, conf, verbose):
        self.called = True
        self.received_frame = frame
        self.received_imgsz = imgsz
        self.received_conf = conf

        return self.results


def test_person_detector_calls_model():
    fake_model = FakeYOLO()

    detector = PersonDetector(
        model=fake_model,
        confidence_threshold=0.5,
    )

    frame = "fake_frame"

    transform = {
        "scale": 1 / 3,
        "padding_x": 0,
        "padding_y": 140,
        "original_width": 1920,
        "original_height": 1080,
        "target_width": 640,
        "target_height": 640,
    }

    results = detector.detect(
        frame=frame
    )

    detections = detector.postprocess(results, transform)
    assert detections == []
    assert fake_model.called is True
    assert fake_model.received_frame == frame
    assert fake_model.received_imgsz == 640
    assert fake_model.received_conf == 0.5


def test_person_detector_returns_person_detection():
    fake_boxes = FakeBoxes(
        cls=[0],
        conf=[0.91],
        xyxy=[[160, 170, 320, 350]],
    )
    fake_names = ["person"]

    fake_result = FakeResult(fake_boxes, fake_names)

    fake_model = FakeYOLO(
        results=[fake_result]
    )

    detector = PersonDetector(
        model=fake_model,
        confidence_threshold=0.5,
    )

    transform = {
        "scale": 1 / 3,
        "padding_x": 0,
        "padding_y": 140,
        "original_width": 1920,
        "original_height": 1080,
        "target_width": 640,
        "target_height": 640,
    }

    results = detector.detect(
        frame="fake_frame"
    )
    detections = detector.postprocess(results, transform)

    assert len(detections) == 1

    detection = detections[0]

    assert detection.class_id == 0
    assert detection.class_name == "person"
    assert detection.confidence == 0.91

    assert detection.bbox["x_min"] == 0.25
    assert detection.bbox["y_min"] == 90 / 1080
    assert detection.bbox["x_max"] == 0.5
    assert detection.bbox["y_max"] == 630 / 1080


def test_person_detector_returns_person_without_car():
    fake_boxes = FakeBoxes(
        cls=[0, 2],
        conf=[0.91, 0.8],
        xyxy=[[160, 170, 320, 350], [190, 407, 320, 540]],
    )
    fake_names = ["person", "car"]

    fake_result = FakeResult(fake_boxes, fake_names)

    fake_model = FakeYOLO(
        results=[fake_result]
    )

    detector = PersonDetector(
        model=fake_model,
        confidence_threshold=0.5,
    )

    transform = {
        "scale": 1 / 3,
        "padding_x": 0,
        "padding_y": 140,
        "original_width": 1920,
        "original_height": 1080,
        "target_width": 640,
        "target_height": 640,
    }

    results = detector.detect(
        frame="fake_frame"
    )
    detections = detector.postprocess(results, transform)
    assert len(detections) == 1

    detection = detections[0]

    assert detection.class_id == 0
    assert detection.class_name == "person"
    assert detection.confidence == 0.91


def test_person_detector_returns_multiple_persons():
    fake_boxes = FakeBoxes(
        cls=[0, 0, 0],
        conf=[0.91, 0.87, 0.82],
        xyxy=[
            [160, 170, 320, 350],
            [330, 180, 450, 360],
            [470, 160, 590, 340],
        ],
    )

    fake_names = ["person"]

    fake_result = FakeResult(fake_boxes, fake_names)

    fake_model = FakeYOLO(
        results=[fake_result]
    )

    detector = PersonDetector(
        model=fake_model,
        confidence_threshold=0.5,
    )

    transform = {
        "scale": 1 / 3,
        "padding_x": 0,
        "padding_y": 140,
        "original_width": 1920,
        "original_height": 1080,
        "target_width": 640,
        "target_height": 640,
    }

    results = detector.detect(
        frame="fake_frame"
    )

    detections = detector.postprocess(results, transform)
    assert len(detections) == 3

    assert detections[0].confidence == 0.91
    assert detections[1].confidence == 0.87
    assert detections[2].confidence == 0.82


def test_person_detector_handles_multiple_results():
    fake_result_1 = FakeResult(
        FakeBoxes(
            cls=[0],
            conf=[0.91],
            xyxy=[[160, 170, 320, 350]],
        ),
        ["person"],
    )

    fake_result_2 = FakeResult(
        FakeBoxes(
            cls=[0],
            conf=[0.82],
            xyxy=[[330, 180, 450, 360]],
        ),
        ["person"],
    )

    fake_model = FakeYOLO(
        results=[
            fake_result_1,
            fake_result_2,
        ]
    )

    detector = PersonDetector(
        model=fake_model,
        confidence_threshold=0.5,
    )

    transform = {
        "scale": 1 / 3,
        "padding_x": 0,
        "padding_y": 140,
        "original_width": 1920,
        "original_height": 1080,
        "target_width": 640,
        "target_height": 640,
    }

    results = detector.detect(
        frame="fake_frame"
    )
    detections = detector.postprocess(results, transform)
    assert len(detections) == 2
    assert detections[0].confidence == 0.91
    assert detections[1].confidence == 0.82