import torch
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class TorchFasterrcnnModel:
    def __init__(self, device=None):
        import torch
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        num_classes = 6
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.load_state_dict(torch.load("FastRCNN_resnet50_fpn_(40epoches).pth", weights_only=True))

        if device:
            self.device = device
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            
        self.model.to(self.device)
        self.labels = [None, 'can', 'foam', 'plastic', 'plastic bottle', 'unknow']


    def image2tensors(self, image):
        '''
        Convert image (raw bytes or image path) to tensor
        -------
        image: Absolute path to image | raw bytes of image
        ------
        return: torch.tensor
        '''
        img = torchvision.io.read_image(image).float()
        img /= 255.0
        return img

    
    def predict(self, targets):
        '''
        For predicting image RGB channels must take values from 0.0 to 1.0, shape = [C, H, W] 
        '''
        self.model.eval()
        if targets.dim() == 3:
            targets = [targets.to(self.device)]
        else:
            targets = [elem.to(self.device) for elem in targets]
            
        with torch.no_grad():
            preds = self.model(targets)

        for pred in preds:
            pred = {
                'boxes': pred['boxes'],
                'labels': pred['labels'],
                'scores': pred['scores']
            }
        return preds
        

    def save_image(self, image_tensor,  prediction=None, score_threshold=0.3, saved_path='predicted.jpeg'):
        import matplotlib.pyplot as plt
        import matplotlib
        
        img = image_tensor.cpu().permute(1, 2, 0).numpy()
        fig, ax = plt.subplots(1,1, figsize = (10, 10))
        ax.imshow(img)
        plt.xticks([])
        plt.yticks([])
        
        if prediction:
            for elem in prediction:
                keep = elem['scores'].cpu() > score_threshold
                boxes = elem['boxes'].cpu()[keep]
                labels = elem['labels'].cpu()[keep]
                scores = elem['scores'].cpu()[keep]
                for i in range(len(scores)):
                    box = boxes[i]
                    label = labels[i]
                    score = scores[i]
                    x1, y1, x2, y2 = box
        
                    rect = matplotlib.patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1, 
                        linewidth=1, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)

                    trash_name = self.labels[label]
                    ax.text(x1, y1, f"{trash_name}", color='white', 
                            fontsize=8, bbox=dict(facecolor='red', alpha=0.5, pad=0))
                    
                    ax.text(x1, y2, f'{score:.2f}', color='white',
                           fontsize=8, bbox=dict(facecolor='red', alpha=0.5, pad=0))
                    
        plt.savefig(saved_path)
        

        
           
