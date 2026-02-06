import cv2
import torch
import torch.nn as nn
import numpy as np

class GrayscaleCNN(nn.Module):
    def __init__(self):
        super(GrayscaleCNN, self).__init__()
        # 3 input channels (RGB), 1 output channel (Grayscale)
        # Kernel size 1x1 to perform a linear combination of RGB values
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, bias=False)
        
        # Initialize weights with standard luma coefficients: 0.2989, 0.5870, 0.1140
        # PyTorch Conv2d weights are (out_channels, in_channels, height, width)
        weights = torch.tensor([[[[0.2989]], [[0.5870]], [[0.1140]]]])
        self.conv.weight.data = weights

    def forward(self, x):
        return self.conv(x)

def capture_and_convert():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 's' to capture and convert, or 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.imshow('Webcam (RGB)', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Convert BGR (OpenCV default) to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Prepare image for CNN: (H, W, C) -> (C, H, W) -> (1, C, H, W)
            # Normalize to [0, 1]
            input_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
            input_tensor = input_tensor.unsqueeze(0)
            
            # Run CNN
            model = GrayscaleCNN()
            model.eval()
            with torch.no_grad():
                grayscale_tensor = model(input_tensor)
            
            # Convert back to numpy for display: (1, 1, H, W) -> (H, W)
            grayscale_img = grayscale_tensor.squeeze().cpu().numpy()
            
            # Convert to uint8 for OpenCV display [0, 255]
            grayscale_img_uint8 = (grayscale_img * 255).astype(np.uint8)
            
            # Show and save results
            cv2.imshow('Grayscale (CNN Output)', grayscale_img_uint8)
            cv2.imwrite('captured_rgb.png', frame)
            cv2.imwrite('converted_grayscale.png', grayscale_img_uint8)
            print("Images saved as 'captured_rgb.png' and 'converted_grayscale.png'")
            
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_convert()
