# operator.py


class OperatorBraTS:
    def __init__(self, data_wrapper, classifier, det_dict, reformer):
        """
        Operator for BraTS medical imaging pipeline handling 3D volumes and 2D slices.
        
        data: Object with .train_loader, .validation_loader, .test_loader
              Expects BraTS data format [N, 4, 155, 240, 240] (3D) or [N, 4, 240, 240] (2D)
        classifier: ClassifierUNet object with .classify() method
        det_dict: Dictionary of BraTS detectors (AEDetector3D/AEDetector2D)
        reformer: SimpleReformer3D/SimpleReformer2D object
        """
        self.data_wrapper = data_wrapper
        self.classifier = classifier
        self.det_dict = det_dict
        self.reformer = reformer

        # Load test data - handles both 3D volumes and 2D slices
        test_images, test_masks = next(iter(data_wrapper.test_loader))
        test_images = test_images.to(classifier.device)
        test_masks = test_masks.to(classifier.device)
        
        self.normal = self.operate(AttackDataBraTS(test_images, test_masks, "Normal"))
        
    def get_thrs(self, drop_rate):
        """Proper threshold calculation with medical validation data"""
        thrs = {}
        
        # Get sufficient validation samples (200+)
        val_loader = self.data_wrapper.val_loader
        val_samples = []
        for batch in val_loader:
            val_samples.append(batch[0])
            if len(val_samples) >= 200:
                break
        val_imgs = torch.cat(val_samples).to(self.classifier.device)

        print(f"Medical Validation: Using {len(val_imgs)} samples for thresholding")
        
        for name, detector in self.det_dict.items():
            marks = detector.mark(val_imgs)
            sorted_marks = np.sort(marks)
            num = int(len(marks) * drop_rate[name])
            thrs[name] = sorted_marks[-num] if num > 0 else np.inf
            
        return thrs

    def operate(self, untrusted_obj):
        device = next(self.classifier.model.parameters()).device
        X = untrusted_obj.data.to(device)
        Y_true = untrusted_obj.labels.to(device)

        with torch.no_grad():
            # Heal input
            X_prime = self.reformer.heal(X)
            
            # Get PROPER segmentation outputs (not flattened)
            Y_pred_logits = self.classifier.model(X)  # [N, C, H, W]
            Yp_pred_logits = self.classifier.model(X_prime)  # [N, C, H, W]
            
            # Convert to class predictions
            Y_pred = torch.argmax(Y_pred_logits, dim=1)  # [N, H, W]
            Yp_pred = torch.argmax(Yp_pred_logits, dim=1)  # [N, H, W]

        # Return actual segmentation maps for Dice calculation
        return list(zip(Y_pred.cpu().numpy(), Yp_pred.cpu().numpy(), Y_true.cpu().numpy()))

    def filter(self, X, thrs):
        all_pass = np.arange(X.shape[0])
        collector = {}
        D = None  # Initialize D
    
        # Handle 3D volumes for 2D detectors
        original_shape = X.shape
        if X.ndim == 5 and '2D' in next(iter(self.det_dict.values())).print():
            N, C, D, H, W = X.shape
            X = X.permute(0, 2, 1, 3, 4).reshape(N*D, C, H, W)
    
        for name, detector in self.det_dict.items():
            marks = detector.mark(X)
            idx_pass = np.argwhere(marks < thrs[name]).flatten()
            
            # Only remap if D was set (3D case)
            if D is not None:
                idx_pass = np.unique(idx_pass // D)
                
            collector[name] = len(idx_pass)
            all_pass = np.intersect1d(all_pass, idx_pass)
    
        return all_pass, collector

    def print(self):
        components = [self.reformer, self.classifier]
        return " ".join(obj.print() for obj in components)
