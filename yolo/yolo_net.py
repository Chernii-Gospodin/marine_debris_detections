from ultralytics import YOLO

class SeaCrapNet(YOLO):
    def __init__(self, model="yolo12m_trained.pt", task="detect", verbose=True):
        super().__init__(model, task, verbose)

    def pipeline(self, file):
        results = self.predict(file)
        self.save_detect(file, results)

    def save_detect(self, name, results):
        results[0].save(name)