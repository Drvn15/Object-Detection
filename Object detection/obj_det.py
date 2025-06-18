import cv2
import numpy as np

class EnhancedObjectDetector:
    def __init__(self):
        self.template = None
        self.template_gray = None
        self.template_edges = None
        self.template_hsv_hist = None
        self.min_scale = 0.4 
        self.max_scale = 2.0
        self.scale_steps = 15 
        self.detection_threshold = 0.55 
        self.last_known_pos = None
        self.detection_history = []
        
    def detect_shape(self, contour):
        """Detect shape from contour"""
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # Count vertices to determine shape
        vertices = len(approx)
        if vertices == 3:
            return "Triangle"
        elif vertices == 4:
            # Check if it's a rectangle/square
            aspect_ratio = float(max(cv2.boundingRect(contour)[2], cv2.boundingRect(contour)[3])) / \
                          float(min(cv2.boundingRect(contour)[2], cv2.boundingRect(contour)[3]))
            if aspect_ratio <= 1.2:
                return "Square"
            return "Rectangle"
        elif vertices >= 5:
            # Check circularity
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            if 0.95 <= aspect_ratio <= 1.05:
                radius = w/2
                circularity = 4*np.pi*(area/(w*h))
                if circularity > 0.85:
                    return "Circle"
            return "Polygon"
        return "Unknown"

    def capture_template(self):
        """Capture a template from webcam ROI"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Webcam not detected! Please check your connection.")
            return False
        
        print("🔹 Press 's' to select object, then SPACE to confirm")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera error!")
                break
            
            cv2.putText(frame, "Select object & press SPACE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Template Capture", frame)
            
            key = cv2.waitKey(1)
            if key == ord('s'):
                roi = cv2.selectROI("Select Object", frame, False)
                if roi != (0, 0, 0, 0):
                    x, y, w, h = map(int, roi)
                    if w < 50 or h < 50:  
                        print("❌ Selected ROI is too small! Please select a larger area.")
                        continue
                    
                    self.template = frame[y:y+h, x:x+w]
                    
                    # Preprocess template
                    self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
                    self.template_gray = cv2.GaussianBlur(self.template_gray, (3,3), 0)
                    self.template_edges = cv2.Canny(self.template_gray, 30, 100)  
                    self.template_edges = cv2.dilate(self.template_edges, None, iterations=1)
                    
                    hsv = cv2.cvtColor(self.template, cv2.COLOR_BGR2HSV)
                    self.template_hsv_hist = cv2.calcHist([hsv], [0, 1], None, [50, 50], [0, 180, 0, 256])
                    cv2.normalize(self.template_hsv_hist, self.template_hsv_hist, 0, 1, cv2.NORM_MINMAX)
                    
                    print(f"✅ Template captured! Size: {w}x{h}")
                    cv2.imshow("Template Preview", self.template)
                    cv2.waitKey(1000)
                    break
            elif key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        return self.template is not None

    def detect_object(self):
        """Real-time object detection with improved accuracy"""
        if self.template is None:
            print("❌ No template captured! Run capture_template() first.")
            return
            
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Webcam not detected! Please check your connection.")
            return
        
        print("🔍 Detecting object... Press 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera error!")
                break
            
            # Convert to grayscale and blur
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter out small contours
                if area < 1000:
                    continue
                    
                # Get shape
                shape = self.detect_shape(contour)
                
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Draw shape outline
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                
                # Add shape label
                cv2.putText(frame, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            cv2.imshow("Shape Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = EnhancedObjectDetector()
    if detector.capture_template():  
        detector.detect_object()qimport cv2
import numpy as np

class EnhancedObjectDetector:
    def __init__(self):
        self.template = None
        self.template_gray = None
        self.template_edges = None
        self.template_hsv_hist = None
        self.min_scale = 0.4 
        self.max_scale = 2.0
        self.scale_steps = 15 
        self.detection_threshold = 0.55 
        self.last_known_pos = None
        self.detection_history = []
        
    def detect_body_parts(self, contour):
        """Detect body parts from contour"""
        area = cv2.contourArea(contour)
        if area < 1000:
            return "Unknown"
            
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        
        # Head detection
        if aspect_ratio > 0.8 and aspect_ratio < 1.2 and h < 150:
            return "Head"
            
        # Torso detection
        if aspect_ratio > 0.4 and aspect_ratio < 0.7 and h > 150:
            return "Torso"
            
        # Arm detection
        if aspect_ratio > 2.0 and h < 120:
            return "Arm"
            
        # Leg detection
        if aspect_ratio > 0.7 and aspect_ratio < 1.0 and h > 150:
            return "Leg"
            
        return "Unknown"

    def capture_template(self):
        """Capture a template from webcam ROI"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Webcam not detected! Please check your connection.")
            return False
        
        print("🔹 Press 's' to select object, then SPACE to confirm")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera error!")
                break
            
            cv2.putText(frame, "Select object & press SPACE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Template Capture", frame)
            
            key = cv2.waitKey(1)
            if key == ord('s'):
                roi = cv2.selectROI("Select Object", frame, False)
                if roi != (0, 0, 0, 0):
                    x, y, w, h = map(int, roi)
                    if w < 50 or h < 50:  
                        print("❌ Selected ROI is too small! Please select a larger area.")
                        continue
                    
                    self.template = frame[y:y+h, x:x+w]
                    
                    # Preprocess template
                    self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
                    self.template_gray = cv2.GaussianBlur(self.template_gray, (3,3), 0)
                    self.template_edges = cv2.Canny(self.template_gray, 30, 100)  
                    self.template_edges = cv2.dilate(self.template_edges, None, iterations=1)
                    
                    hsv = cv2.cvtColor(self.template, cv2.COLOR_BGR2HSV)
                    self.template_hsv_hist = cv2.calcHist([hsv], [0, 1], None, [50, 50], [0, 180, 0, 256])
                    cv2.normalize(self.template_hsv_hist, self.template_hsv_hist, 0, 1, cv2.NORM_MINMAX)
                    
                    print(f"✅ Template captured! Size: {w}x{h}")
                    cv2.imshow("Template Preview", self.template)
                    cv2.waitKey(1000)
                    break
            elif key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        return self.template is not None

    def detect_object(self):
        """Real-time object detection with improved accuracy"""
        if self.template is None:
            print("❌ No template captured! Run capture_template() first.")
            return
            
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Webcam not detected! Please check your connection.")
            return
        
        print("🔍 Detecting object... Press 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera error!")
                break
            
            # Convert to grayscale and blur
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter out small contours
                if area < 1000:
                    continue
                    
                # Get body part
                body_part = self.detect_body_parts(contour)
                
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Draw shape outline
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                
                # Add body part label
                cv2.putText(frame, body_part, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            cv2.imshow("Body Part Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = EnhancedObjectDetector()
    if detector.capture_template():  
        detector.detect_object()