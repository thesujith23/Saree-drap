# """
# Saree Draping Style Detection API
# With Advanced Multi-Person Detection (MediaPipe + Face Detection)
# Production-Ready Backend
# """

# from flask import Flask, render_template, request, jsonify
# from flask_cors import CORS
# import cv2
# import numpy as np
# from tensorflow import keras
# from PIL import Image
# import os
# import base64
# from datetime import datetime
# import mediapipe as mp

# app = Flask(__name__)
# CORS(app)

# # Configuration
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}

# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max

# # Confidence thresholds
# SAREE_CONFIDENCE_THRESHOLD = 0.60
# PERSON_DETECTION_CONFIDENCE = 0.3


# class MultiPersonDetector:
#     """Advanced multi-person detection using MediaPipe Face + Body Detection"""
    
#     def __init__(self):
#         # Initialize MediaPipe Face Detection
#         self.mp_face_detection = mp.solutions.face_detection
#         self.face_detector = self.mp_face_detection.FaceDetection(
#             model_selection=1,
#             min_detection_confidence=0.5
#         )
        
#         # Initialize MediaPipe Pose for body detection
#         self.mp_pose = mp.solutions.pose
    
#     def detect_multiple_persons(self, image):
#         """
#         Detect multiple persons using Face + Body detection
#         Returns: (person_count, confidence, details)
#         """
#         # Method 1: Face Detection
#         face_count = self._detect_faces(image)
        
#         # Method 2: Body/Pose Detection
#         body_count = self._detect_bodies(image)
        
#         # Use the maximum count from both methods
#         person_count = max(face_count, body_count)
        
#         # Calculate confidence
#         if face_count > 0 and body_count > 0:
#             confidence = 1.0  # Both methods agree
#         elif face_count > 0 or body_count > 0:
#             confidence = 0.7  # One method detected
#         else:
#             confidence = 0.0  # No detection
        
#         details = {
#             'face_detection': face_count,
#             'body_detection': body_count,
#             'method': 'face+body'
#         }
        
#         return person_count, confidence, details
    
#     def _detect_faces(self, image):
#         """Detect faces using MediaPipe Face Detection"""
#         try:
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = self.face_detector.process(image_rgb)
            
#             if results.detections:
#                 return len(results.detections)
#             return 0
#         except:
#             return 0
    
#     def _detect_bodies(self, image):
#         """
#         Detect bodies using grid-based pose detection
#         Splits wide images to detect multiple people
#         """
#         try:
#             h, w = image.shape[:2]
            
#             # If image is wide (likely group photo), check in sections
#             if w > h * 1.5:
#                 # Wide image - likely group photo
#                 sections = min(4, int(w / (h * 0.8)))  # Max 4 sections
#                 section_width = w // sections
                
#                 person_count = 0
                
#                 for i in range(sections):
#                     x_start = i * section_width
#                     x_end = (i + 1) * section_width if i < sections - 1 else w
#                     section = image[:, x_start:x_end]
                    
#                     if self._has_person_in_section(section):
#                         person_count += 1
                
#                 return person_count
#             else:
#                 # Normal aspect ratio - check single person
#                 return 1 if self._has_person_in_section(image) else 0
#         except:
#             return 0
    
#     def _has_person_in_section(self, image_section):
#         """Check if a section contains a person"""
#         try:
#             pose = self.mp_pose.Pose(
#                 static_image_mode=True,
#                 model_complexity=0,
#                 min_detection_confidence=0.3
#             )
            
#             image_rgb = cv2.cvtColor(image_section, cv2.COLOR_BGR2RGB)
#             results = pose.process(image_rgb)
#             pose.close()
            
#             if results.pose_landmarks:
#                 # Count visible landmarks
#                 visible = sum(1 for lm in results.pose_landmarks.landmark if lm.visibility > 0.5)
#                 return visible > 10  # Need at least 10 visible landmarks
            
#             return False
#         except:
#             return False
    
#     def close(self):
#         """Clean up resources"""
#         try:
#             self.face_detector.close()
#         except:
#             pass


# class SareeDrapingDetector:
#     """Saree Detector with Advanced Multi-Person Detection"""
    
#     def __init__(self):
#         """Initialize all models"""
#         try:
#             print("🔄 Loading models...")
            
#             # Load saree detection models
#             self.saree_model = keras.models.load_model('models/saree_detection_model.h5')
#             self.draping_model = keras.models.load_model('models/draping_style_model_final.h5')
            
#             # Load draping classes
#             with open('models/draping_classes.txt', 'r') as f:
#                 self.draping_classes = [line.strip() for line in f.readlines()]
            
#             # Initialize person detector
#             self.person_detector = MultiPersonDetector()
            
#             # Initialize MediaPipe Pose for single person validation
#             self.mp_pose = mp.solutions.pose
#             self.pose = self.mp_pose.Pose(
#                 static_image_mode=True,
#                 model_complexity=1,
#                 enable_segmentation=False,
#                 min_detection_confidence=PERSON_DETECTION_CONFIDENCE
#             )
            
#             # Style descriptions
#             self.style_descriptions = {
#                 'normal_drape_style': 'The Nivi drape is one of the most popular and modern saree draping styles, originating from Andhra Pradesh. It is characterized by pleats tucked into the waistband of the petticoat, with the decorated end, known as the pallu, draped over the left shoulder.',
#                 'coorg_drape_style': 'The Coorg drape originates from Karnataka and features pleats at the back with the pallu coming over the shoulder from behind. This unique style is both traditional and elegant.',
#                 'uttar_drape_style': 'The Uttara Karnataka drape features a trouser-like appearance with the pallu pinned at the back. It is practical and distinctive to the region.',
#                 'uttara_karnataka_drape': 'The Uttara Karnataka drape features a trouser-like appearance with the pallu pinned at the back. It is practical and distinctive to the region.'
#             }
            
#             print("✅ All models loaded successfully!")
#             print(f"✅ Multi-person detection: ENABLED")
#             print(f"✅ Draping styles: {', '.join(self.draping_classes)}")
            
#         except Exception as e:
#             print(f"❌ Error loading models: {e}")
#             raise
    
#     def validate_image(self, image):
#         """Validate image quality and content"""
#         try:
#             h, w = image.shape[:2]
            
#             # Check dimensions
#             if h < 100 or w < 100:
#                 return False, "Image too small (minimum 100x100 pixels required)"
            
#             # Check if image is not blank
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             variance = np.var(gray)
            
#             if variance < 50:
#                 return False, "Image appears to be blank or uniform"
            
#             return True, None
            
#         except Exception as e:
#             return False, f"Image validation failed: {str(e)}"
    
#     def preprocess_image(self, image):
#         """Preprocess image for classification"""
#         try:
#             # Resize to 224x224
#             img_resized = cv2.resize(image, (224, 224))
            
#             # Convert BGR to RGB
#             img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
#             # Normalize to [0, 1]
#             img_array = img_rgb.astype(np.float32) / 255.0
            
#             # Add batch dimension
#             img_array = np.expand_dims(img_array, axis=0)
            
#             return img_array, None
            
#         except Exception as e:
#             return None, f"Preprocessing failed: {str(e)}"
    
#     def detect_and_classify(self, image_path):
#         """
#         Complete detection pipeline with multi-person detection
#         """
#         result = {
#             'timestamp': datetime.now().strftime("%Y-%m-%d %I:%M %p"),
#             'status': 'processing'
#         }
        
#         try:
#             # Step 1: Read image
#             image = cv2.imread(image_path)
#             if image is None:
#                 result['status'] = 'error'
#                 result['error_type'] = 'invalid_image'
#                 result['message'] = '❌ Could not read the image file'
#                 return result
            
#             # Step 2: Validate image
#             is_valid, error_msg = self.validate_image(image)
#             if not is_valid:
#                 result['status'] = 'error'
#                 result['error_type'] = 'invalid_content'
#                 result['message'] = f'❌ {error_msg}'
#                 return result
            
#             # Step 3: MULTI-PERSON DETECTION
#             person_count, detection_conf, detection_details = self.person_detector.detect_multiple_persons(image)
            
#             result['person_count'] = int(person_count)
#             result['detection_confidence'] = float(detection_conf * 100)
#             result['detection_details'] = detection_details
            
#             # Step 4: Validate person count
#             if person_count == 0:
#                 result['status'] = 'error'
#                 result['error_type'] = 'no_person'
#                 result['message'] = '❌ No Person Detected'
#                 result['suggestion'] = 'Please upload a clear image with a person visible. Make sure the person is clearly visible in good lighting.'
#                 return result
            
#             if person_count > 1:
#                 result['status'] = 'error'
#                 result['error_type'] = 'multiple_persons'
#                 result['message'] = f'⚠️ Multiple Persons Detected ({person_count} people)'
#                 result['suggestion'] = 'Please upload an image with ONLY ONE PERSON. Try:\n• Cropping the image to show only one person\n• Taking a new photo with just one person\n• Using a different image'
#                 result['warning'] = f'Group photo detected with {person_count} people. This system works best with single-person images.'
#                 return result
            
#             # Step 5: Single person detected - Additional validation with pose
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             pose_results = self.pose.process(image_rgb)
            
#             if pose_results.pose_landmarks:
#                 visibilities = [lm.visibility for lm in pose_results.pose_landmarks.landmark]
#                 person_confidence = sum(visibilities) / len(visibilities) * 100
#                 result['person_confidence'] = float(person_confidence)
#             else:
#                 result['person_confidence'] = float(detection_conf * 100)
            
#             # Step 6: Preprocess for classification
#             img_array, prep_error = self.preprocess_image(image)
#             if prep_error:
#                 result['status'] = 'error'
#                 result['error_type'] = 'preprocessing_failed'
#                 result['message'] = f'❌ {prep_error}'
#                 return result
            
#             # Step 7: Saree detection
#             saree_prediction = self.saree_model.predict(img_array, verbose=0)
#             saree_score = float(saree_prediction[0][0])
            
#             is_saree = saree_score > SAREE_CONFIDENCE_THRESHOLD
#             saree_confidence = saree_score if is_saree else (1 - saree_score)
            
#             result['is_saree'] = bool(is_saree)
#             result['saree_detected'] = bool(is_saree)
#             result['saree_confidence'] = float(saree_confidence * 100)
            
#             # Step 8: If saree detected, classify draping style
#             if is_saree:
#                 draping_prediction = self.draping_model.predict(img_array, verbose=0)
#                 predicted_idx = np.argmax(draping_prediction[0])
#                 draping_style = self.draping_classes[predicted_idx]
#                 draping_confidence = float(draping_prediction[0][predicted_idx])
                
#                 result['status'] = 'success'
#                 result['draping_style'] = self._format_style_name(draping_style)
#                 result['draping_confidence'] = float(draping_confidence * 100)
#                 result['description'] = self._get_description(draping_style)
                
#                 # All predictions
#                 result['all_predictions'] = {
#                     self._format_style_name(self.draping_classes[i]): 
#                     float(draping_prediction[0][i] * 100)
#                     for i in range(len(self.draping_classes))
#                 }
                
#             else:
#                 result['status'] = 'no_saree'
#                 result['message'] = '❌ No Saree Detected - The person is not wearing a saree'
            
#             return result
            
#         except Exception as e:
#             print(f"❌ Error in detection: {e}")
#             result['status'] = 'error'
#             result['error_type'] = 'processing_exception'
#             result['message'] = f'❌ Processing error: {str(e)}'
#             return result
    
#     def _format_style_name(self, style_name):
#         """Format style name for display"""
#         mappings = {
#             'normal_drape_style': 'Nivi Style',
#             'coorg_drape_style': 'Coorg Style',
#             'uttar_drape_style': 'Uttara Karnataka Style',
#             'uttara_karnataka_drape': 'Uttara Karnataka Style'
#         }
#         return mappings.get(style_name.lower(), style_name.replace('_', ' ').title())
    
#     def _get_description(self, style_name):
#         """Get style description"""
#         return self.style_descriptions.get(
#             style_name.lower(), 
#             'Traditional Indian saree draping style.'
#         )
    
#     def cleanup(self):
#         """Clean up resources"""
#         try:
#             self.pose.close()
#             self.person_detector.close()
#         except:
#             pass


# # Initialize detector globally
# try:
#     detector = SareeDrapingDetector()
#     print("✅ Detector initialized successfully")
# except Exception as e:
#     print(f"❌ Failed to initialize detector: {e}")
#     detector = None


# def allowed_file(filename):
#     """Check if file extension is allowed"""
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# @app.route('/')
# def home():
#     """Serve the main page"""
#     return render_template('index.html')


# @app.route('/api/detect', methods=['POST'])
# def detect_saree():
#     """Main detection endpoint"""
#     try:
#         # Check if detector is ready
#         if detector is None:
#             return jsonify({
#                 'status': 'error',
#                 'error_type': 'server_error',
#                 'message': '❌ Server not ready - Models not loaded'
#             }), 500
        
#         # Check for file
#         if 'image' not in request.files:
#             return jsonify({
#                 'status': 'error',
#                 'error_type': 'no_file',
#                 'message': '❌ No image file provided'
#             }), 400
        
#         file = request.files['image']
        
#         if file.filename == '':
#             return jsonify({
#                 'status': 'error',
#                 'error_type': 'empty_filename',
#                 'message': '❌ No file selected'
#             }), 400
        
#         # Validate file extension
#         if not allowed_file(file.filename):
#             ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'unknown'
#             return jsonify({
#                 'status': 'error',
#                 'error_type': 'invalid_format',
#                 'message': f'❌ Invalid file format (.{ext})',
#                 'suggestion': 'Please upload JPG, PNG, GIF, BMP, or WEBP'
#             }), 400
        
#         # Check file size
#         file.seek(0, os.SEEK_END)
#         file_size = file.tell()
#         file.seek(0)
        
#         if file_size == 0:
#             return jsonify({
#                 'status': 'error',
#                 'error_type': 'empty_file',
#                 'message': '❌ Empty file uploaded'
#             }), 400
        
#         if file_size > app.config['MAX_CONTENT_LENGTH']:
#             size_mb = file_size / (1024 * 1024)
#             return jsonify({
#                 'status': 'error',
#                 'error_type': 'file_too_large',
#                 'message': f'❌ File too large ({size_mb:.1f}MB). Max: 5MB'
#             }), 413
        
#         # Save file temporarily
#         try:
#             filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)
            
#         except Exception as e:
#             return jsonify({
#                 'status': 'error',
#                 'error_type': 'save_failed',
#                 'message': f'❌ Failed to save file: {str(e)}'
#             }), 500
        
#         # Perform detection
#         try:
#             result = detector.detect_and_classify(filepath)
            
#             # Clean up file
#             try:
#                 os.remove(filepath)
#             except:
#                 pass
            
#             # Return appropriate status code
#             if result['status'] == 'error':
#                 return jsonify(result), 400
            
#             return jsonify(result), 200
            
#         except Exception as e:
#             # Clean up on error
#             try:
#                 os.remove(filepath)
#             except:
#                 pass
            
#             return jsonify({
#                 'status': 'error',
#                 'error_type': 'detection_failed',
#                 'message': f'❌ Detection failed: {str(e)}'
#             }), 500
    
#     except Exception as e:
#         return jsonify({
#             'status': 'error',
#             'error_type': 'unexpected_error',
#             'message': f'❌ Unexpected error: {str(e)}'
#         }), 500


# @app.route('/api/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     models_loaded = detector is not None
    
#     return jsonify({
#         'status': 'healthy' if models_loaded else 'unhealthy',
#         'models_loaded': models_loaded,
#         'multi_person_detection': True,
#         'detection_methods': ['MediaPipe Face', 'OpenCV Haar', 'Pose Grid'],
#         'timestamp': datetime.now().strftime("%Y-%m-%d %I:%M %p"),
#         'version': '3.0'
#     }), 200 if models_loaded else 503


# @app.route('/api/styles', methods=['GET'])
# def get_styles():
#     """Get supported draping styles"""
#     if detector is None:
#         return jsonify({
#             'status': 'error',
#             'message': 'Server not ready'
#         }), 500
    
#     styles = [detector._format_style_name(cls) for cls in detector.draping_classes]
    
#     return jsonify({
#         'status': 'success',
#         'styles': styles,
#         'count': len(styles)
#     }), 200


# @app.errorhandler(413)
# def request_entity_too_large(error):
#     """Handle file too large"""
#     return jsonify({
#         'status': 'error',
#         'error_type': 'file_too_large',
#         'message': '❌ File too large - Maximum 5MB allowed'
#     }), 413


# @app.errorhandler(500)
# def internal_error(error):
#     """Handle server errors"""
#     return jsonify({
#         'status': 'error',
#         'error_type': 'server_error',
#         'message': '❌ Internal server error'
#     }), 500


# @app.errorhandler(404)
# def not_found(error):
#     """Handle not found"""
#     return jsonify({
#         'status': 'error',
#         'error_type': 'not_found',
#         'message': '❌ Endpoint not found'
#     }), 404


# if __name__ == '__main__':
#     print("\n" + "="*70)
#     print("🎨 SAREE DRAPING STYLE DETECTION API v3.0")
#     print("   Advanced Multi-Person Detection System")
#     print("="*70)
#     print(f"🌐 Server: http://localhost:5000")
#     print(f"🏥 Health: http://localhost:5000/api/health")
#     print(f"🔍 Detect: POST http://localhost:5000/api/detect")
#     print(f"📋 Styles: GET http://localhost:5000/api/styles")
#     print("="*70)
#     print("\n✨ Advanced Features:")
#     print("  ✓ Multi-person detection (3 methods)")
#     print("  ✓ Group photo detection & warning")
#     print("  ✓ Face detection (MediaPipe + OpenCV)")
#     print("  ✓ Body detection with grid analysis")
#     print("  ✓ Saree vs non-saree classification")
#     print("  ✓ Multi-style draping identification")
#     print("="*70 + "\n")
    
#     try:
#         app.run(debug=True, host='0.0.0.0', port=5000)
#     finally:
#         if detector:
#             detector.cleanup()

"""
Saree Draping Style Detection API
With Advanced Multi-Person Detection (MediaPipe + Face Detection)
Production-Ready Backend
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow import keras
from PIL import Image
import os
import base64
from datetime import datetime
import mediapipe as mp

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max

# Confidence thresholds
SAREE_CONFIDENCE_THRESHOLD = 0.60
PERSON_DETECTION_CONFIDENCE = 0.3


class MultiPersonDetector:
    """Advanced multi-person detection using MediaPipe Face + Body Detection"""
    
    def __init__(self):
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        
        # Initialize MediaPipe Pose for body detection
        self.mp_pose = mp.solutions.pose
    
    def detect_multiple_persons(self, image):
        """
        Detect multiple persons using Face + Body detection
        Returns: (person_count, confidence, details)
        """
        # Method 1: Face Detection
        face_count = self._detect_faces(image)
        
        # Method 2: Body/Pose Detection
        body_count = self._detect_bodies(image)
        
        # Use the maximum count from both methods
        person_count = max(face_count, body_count)
        
        # Calculate confidence
        if face_count > 0 and body_count > 0:
            confidence = 1.0  # Both methods agree
        elif face_count > 0 or body_count > 0:
            confidence = 0.7  # One method detected
        else:
            confidence = 0.0  # No detection
        
        details = {
            'face_detection': face_count,
            'body_detection': body_count,
            'method': 'face+body'
        }
        
        return person_count, confidence, details
    
    def _detect_faces(self, image):
        """Detect faces using MediaPipe Face Detection"""
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(image_rgb)
            
            if results.detections:
                return len(results.detections)
            return 0
        except:
            return 0
    
    def _detect_bodies(self, image):
        """
        Detect bodies using grid-based pose detection
        Splits wide images to detect multiple people
        """
        try:
            h, w = image.shape[:2]
            
            # If image is wide (likely group photo), check in sections
            if w > h * 1.5:
                # Wide image - likely group photo
                sections = min(4, int(w / (h * 0.8)))  # Max 4 sections
                section_width = w // sections
                
                person_count = 0
                
                for i in range(sections):
                    x_start = i * section_width
                    x_end = (i + 1) * section_width if i < sections - 1 else w
                    section = image[:, x_start:x_end]
                    
                    if self._has_person_in_section(section):
                        person_count += 1
                
                return person_count
            else:
                # Normal aspect ratio - check single person
                return 1 if self._has_person_in_section(image) else 0
        except:
            return 0
    
    def _has_person_in_section(self, image_section):
        """Check if a section contains a person"""
        try:
            pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=0,
                min_detection_confidence=0.3
            )
            
            image_rgb = cv2.cvtColor(image_section, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            pose.close()
            
            if results.pose_landmarks:
                # Count visible landmarks
                visible = sum(1 for lm in results.pose_landmarks.landmark if lm.visibility > 0.5)
                return visible > 10  # Need at least 10 visible landmarks
            
            return False
        except:
            return False
    
    def close(self):
        """Clean up resources"""
        try:
            self.face_detector.close()
        except:
            pass


class SareeDrapingDetector:
    """Saree Detector with Advanced Multi-Person Detection"""
    
    def __init__(self):
        """Initialize all models"""
        try:
            print("🔄 Loading models...")
            
            # Load saree detection models
            self.saree_model = keras.models.load_model('models/saree_detection_model.h5')
            self.draping_model = keras.models.load_model('models/draping_style_model_final.h5')
            
            # Load draping classes
            with open('models/draping_classes.txt', 'r') as f:
                self.draping_classes = [line.strip() for line in f.readlines()]
            
            # Initialize person detector
            self.person_detector = MultiPersonDetector()
            
            # Initialize MediaPipe Pose for single person validation
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=PERSON_DETECTION_CONFIDENCE
            )
            
            # Style descriptions
            self.style_descriptions = {
                'normal_drape_style': 'The Nivi drape is one of the most popular and modern saree draping styles, originating from Andhra Pradesh. It is characterized by pleats tucked into the waistband of the petticoat, with the decorated end, known as the pallu, draped over the left shoulder.',
                'coorg_drape_style': 'The Coorg drape originates from Karnataka and features pleats at the back with the pallu coming over the shoulder from behind. This unique style is both traditional and elegant.',
                'uttar_drape_style': 'The Uttara Karnataka drape features a trouser-like appearance with the pallu pinned at the back. It is practical and distinctive to the region.',
                'uttara_karnataka_drape': 'The Uttara Karnataka drape features a trouser-like appearance with the pallu pinned at the back. It is practical and distinctive to the region.'
            }
            
            print("✅ All models loaded successfully!")
            print(f"✅ Multi-person detection: ENABLED")
            print(f"✅ Draping styles: {', '.join(self.draping_classes)}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def validate_image(self, image):
        """Validate image quality and content"""
        try:
            h, w = image.shape[:2]
            
            # Check dimensions
            if h < 100 or w < 100:
                return False, "Image too small (minimum 100x100 pixels required)"
            
            # Check if image is not blank
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            variance = np.var(gray)
            
            if variance < 50:
                return False, "Image appears to be blank or uniform"
            
            return True, None
            
        except Exception as e:
            return False, f"Image validation failed: {str(e)}"
    
    def preprocess_image(self, image):
        """Preprocess image for classification"""
        try:
            # Resize to 224x224
            img_resized = cv2.resize(image, (224, 224))
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            img_array = img_rgb.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array, None
            
        except Exception as e:
            return None, f"Preprocessing failed: {str(e)}"
    
    def detect_and_classify(self, image_path):
        """
        Complete detection pipeline with multi-person detection
        """
        result = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %I:%M %p"),
            'status': 'processing'
        }
        
        try:
            # Step 1: Read image
            image = cv2.imread(image_path)
            if image is None:
                result['status'] = 'error'
                result['error_type'] = 'invalid_image'
                result['message'] = '❌ Could not read the image file'
                return result
            
            # Step 2: Validate image
            is_valid, error_msg = self.validate_image(image)
            if not is_valid:
                result['status'] = 'error'
                result['error_type'] = 'invalid_content'
                result['message'] = f'❌ {error_msg}'
                return result
            
            # Step 3: MULTI-PERSON DETECTION
            person_count, detection_conf, detection_details = self.person_detector.detect_multiple_persons(image)
            
            result['person_count'] = int(person_count)
            result['detection_confidence'] = float(detection_conf * 100)
            result['detection_details'] = detection_details
            
            # Step 4: Validate person count
            if person_count == 0:
                result['status'] = 'error'
                result['error_type'] = 'no_person'
                result['message'] = '❌ No Person Detected'
                result['suggestion'] = 'Please upload a clear image with a person visible. Make sure the person is clearly visible in good lighting.'
                return result
            
            if person_count > 1:
                result['status'] = 'error'
                result['error_type'] = 'multiple_persons'
                result['message'] = f'⚠️ Multiple Persons Detected'
                result['suggestion'] = 'Please upload an image with only ONE person'
                result['warning'] = f'More than 1 person found in this image. Only single person images are supported.'
                # Don't include exact count in response
                result['person_count'] = 'multiple'
                return result
            
            # Step 5: Single person detected - Additional validation with pose
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(image_rgb)
            
            if pose_results.pose_landmarks:
                visibilities = [lm.visibility for lm in pose_results.pose_landmarks.landmark]
                person_confidence = sum(visibilities) / len(visibilities) * 100
                result['person_confidence'] = float(person_confidence)
            else:
                result['person_confidence'] = float(detection_conf * 100)
            
            # Step 6: Preprocess for classification
            img_array, prep_error = self.preprocess_image(image)
            if prep_error:
                result['status'] = 'error'
                result['error_type'] = 'preprocessing_failed'
                result['message'] = f'❌ {prep_error}'
                return result
            
            # Step 7: Saree detection
            saree_prediction = self.saree_model.predict(img_array, verbose=0)
            saree_score = float(saree_prediction[0][0])
            
            is_saree = saree_score > SAREE_CONFIDENCE_THRESHOLD
            saree_confidence = saree_score if is_saree else (1 - saree_score)
            
            result['is_saree'] = bool(is_saree)
            result['saree_detected'] = bool(is_saree)
            result['saree_confidence'] = float(saree_confidence * 100)
            
            # Step 8: If saree detected, classify draping style
            if is_saree:
                draping_prediction = self.draping_model.predict(img_array, verbose=0)
                predicted_idx = np.argmax(draping_prediction[0])
                draping_style = self.draping_classes[predicted_idx]
                draping_confidence = float(draping_prediction[0][predicted_idx])
                
                result['status'] = 'success'
                result['draping_style'] = self._format_style_name(draping_style)
                result['draping_confidence'] = float(draping_confidence * 100)
                result['description'] = self._get_description(draping_style)
                
                # All predictions
                result['all_predictions'] = {
                    self._format_style_name(self.draping_classes[i]): 
                    float(draping_prediction[0][i] * 100)
                    for i in range(len(self.draping_classes))
                }
                
            else:
                result['status'] = 'no_saree'
                result['message'] = '❌ No Saree Detected - The person is not wearing a saree'
            
            return result
            
        except Exception as e:
            print(f"❌ Error in detection: {e}")
            result['status'] = 'error'
            result['error_type'] = 'processing_exception'
            result['message'] = f'❌ Processing error: {str(e)}'
            return result
    
    def _format_style_name(self, style_name):
        """Format style name for display"""
        mappings = {
            'normal_drape_style': 'Nivi Style',
            'coorg_drape_style': 'Coorg Style',
            'uttar_drape_style': 'Uttara Karnataka Style',
            'uttara_karnataka_drape': 'Uttara Karnataka Style'
        }
        return mappings.get(style_name.lower(), style_name.replace('_', ' ').title())
    
    def _get_description(self, style_name):
        """Get style description"""
        return self.style_descriptions.get(
            style_name.lower(), 
            'Traditional Indian saree draping style.'
        )
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.pose.close()
            self.person_detector.close()
        except:
            pass


# Initialize detector globally
try:
    detector = SareeDrapingDetector()
    print("✅ Detector initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize detector: {e}")
    detector = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def detect_saree():
    """Main detection endpoint"""
    try:
        # Check if detector is ready
        if detector is None:
            return jsonify({
                'status': 'error',
                'error_type': 'server_error',
                'message': '❌ Server not ready - Models not loaded'
            }), 500
        
        # Check for file
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'error_type': 'no_file',
                'message': '❌ No image file provided'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'error_type': 'empty_filename',
                'message': '❌ No file selected'
            }), 400
        
        # Validate file extension
        if not allowed_file(file.filename):
            ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'unknown'
            return jsonify({
                'status': 'error',
                'error_type': 'invalid_format',
                'message': f'❌ Invalid file format (.{ext})',
                'suggestion': 'Please upload JPG, PNG, GIF, BMP, or WEBP'
            }), 400
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size == 0:
            return jsonify({
                'status': 'error',
                'error_type': 'empty_file',
                'message': '❌ Empty file uploaded'
            }), 400
        
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            size_mb = file_size / (1024 * 1024)
            return jsonify({
                'status': 'error',
                'error_type': 'file_too_large',
                'message': f'❌ File too large ({size_mb:.1f}MB). Max: 5MB'
            }), 413
        
        # Save file temporarily
        try:
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error_type': 'save_failed',
                'message': f'❌ Failed to save file: {str(e)}'
            }), 500
        
        # Perform detection
        try:
            result = detector.detect_and_classify(filepath)
            
            # Clean up file
            try:
                os.remove(filepath)
            except:
                pass
            
            # Return appropriate status code
            if result['status'] == 'error':
                return jsonify(result), 400
            
            return jsonify(result), 200
            
        except Exception as e:
            # Clean up on error
            try:
                os.remove(filepath)
            except:
                pass
            
            return jsonify({
                'status': 'error',
                'error_type': 'detection_failed',
                'message': f'❌ Detection failed: {str(e)}'
            }), 500
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error_type': 'unexpected_error',
            'message': f'❌ Unexpected error: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    models_loaded = detector is not None
    
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'multi_person_detection': True,
        'detection_methods': ['MediaPipe Face', 'MediaPipe Body/Pose'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        'version': '3.0'
    }), 200 if models_loaded else 503


@app.route('/api/styles', methods=['GET'])
def get_styles():
    """Get supported draping styles"""
    if detector is None:
        return jsonify({
            'status': 'error',
            'message': 'Server not ready'
        }), 500
    
    styles = [detector._format_style_name(cls) for cls in detector.draping_classes]
    
    return jsonify({
        'status': 'success',
        'styles': styles,
        'count': len(styles)
    }), 200


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large"""
    return jsonify({
        'status': 'error',
        'error_type': 'file_too_large',
        'message': '❌ File too large - Maximum 5MB allowed'
    }), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle server errors"""
    return jsonify({
        'status': 'error',
        'error_type': 'server_error',
        'message': '❌ Internal server error'
    }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle not found"""
    return jsonify({
        'status': 'error',
        'error_type': 'not_found',
        'message': '❌ Endpoint not found'
    }), 404


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🎨 SAREE DRAPING STYLE DETECTION API v3.0")
    print("   Advanced Multi-Person Detection System")
    print("="*70)
    print(f"🌐 Server: http://localhost:5000")
    print(f"🏥 Health: http://localhost:5000/api/health")
    print(f"🔍 Detect: POST http://localhost:5000/api/detect")
    print(f"📋 Styles: GET http://localhost:5000/api/styles")
    print("="*70)
    print("\n✨ Advanced Features:")
    print("  ✓ Multi-person detection (Face + Body)")
    print("  ✓ Group photo detection & warning")
    print("  ✓ MediaPipe Face detection")
    print("  ✓ MediaPipe Body/Pose detection")
    print("  ✓ Saree vs non-saree classification")
    print("  ✓ Multi-style draping identification")
    print("="*70 + "\n")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    finally:
        if detector:
            detector.cleanup()