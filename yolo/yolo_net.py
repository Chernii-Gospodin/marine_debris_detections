from ultralytics import YOLO

class SeaCrapNet(YOLO):
    def __init__(self, model="yolo12m_trained.pt", task="detect", verbose=True):
        super().__init__(model, task, verbose)
        self.detected: dict[str: int] = dict()

    def pipeline(self, file):
        results = self.predict(file)[0]
        self.save_detect(file, results)
        self.count_detected(results)
        

    def save_detect(self, name, results):
        results.save(name)

    def count_detected(self, results):
        classes = list(results.names.values())
        item_to_class = results.names
        self.detected = dict.fromkeys(classes, 0)
        items = results.boxes.cls
        for item in items:
            int_item = int(item)
            self.detected[item_to_class[int_item]] += 1
