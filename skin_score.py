import cv2
import numpy as np
from typing import Dict, List, Tuple
import logging
import os
from datetime import datetime
import json
from sklearn.cluster import KMeans

class SkinAnalyzer:
    def __init__(self):
        self.logger = self._setup_logger()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Define analysis weights
        self.weights = {
            'texture': 0.25,
            'spots': 0.25,
            'brightness': 0.15,
            'evenness': 0.20,
            'hydration': 0.15
        }
        
        # Define makeup detection thresholds
        self.makeup_thresholds = {
            'lip_color_variance': 30,
            'eye_makeup_intensity': 50,
            'foundation_smoothness': 20,
            'color_saturation': 60
        }
    
    def _setup_logger(self):
        """Initialize logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def detect_makeup(self, image: np.ndarray) -> Tuple[bool, Dict]:
        """
        Detect if the person is wearing makeup that would significantly affect skin analysis.
        Adjusted for selfie conditions.
        Returns: (is_wearing_heavy_makeup, detection_details)
        """
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        detection_details = {}
        h, w = image.shape[:2]
        
        # More lenient foundation detection for selfies
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        variance = cv2.Laplacian(blurred, cv2.CV_64F).var()
        foundation_detected = variance < self.makeup_thresholds['foundation_smoothness'] * 0.7
        detection_details['foundation'] = foundation_detected
        
        # Improved lip detection considering selfie lighting
        lip_region = image[int(2*h/3):, int(w/3):int(2*w/3)]
        lip_hsv = cv2.cvtColor(lip_region, cv2.COLOR_BGR2HSV)
        
        # Calculate lip color statistics with adjusted thresholds
        lip_saturation = np.mean(lip_hsv[:,:,1])
        lip_value = np.mean(lip_hsv[:,:,2])
        
        # More lenient lipstick detection for selfies
        lipstick_detected = (
            lip_saturation > self.makeup_thresholds['lip_color_variance'] * 1.2 and
            lip_value > 80
        )
        detection_details['lipstick'] = lipstick_detected
        
        # Analyze cheek areas with consideration for selfie angles
        # Use relative positioning for cheek detection
        left_cheek = image[int(h*0.3):int(h*0.7), int(w*0.1):int(w*0.3)]
        right_cheek = image[int(h*0.3):int(h*0.7), int(w*0.7):int(w*0.9)]
        
        left_hsv = cv2.cvtColor(left_cheek, cv2.COLOR_BGR2HSV)
        right_hsv = cv2.cvtColor(right_cheek, cv2.COLOR_BGR2HSV)
        
        # Average saturation with compensation for uneven lighting
        left_saturation = np.mean(left_hsv[:,:,1])
        right_saturation = np.mean(right_hsv[:,:,1])
        
        # Use the lower saturation value to avoid false positives from lighting
        avg_saturation = min(left_saturation, right_saturation)
        high_saturation = avg_saturation > self.makeup_thresholds['color_saturation'] * 1.3
        detection_details['high_saturation'] = high_saturation
        
        # Texture analysis with selfie considerations
        cheek_ycrcb_left = cv2.cvtColor(left_cheek, cv2.COLOR_BGR2YCrCb)
        cheek_ycrcb_right = cv2.cvtColor(right_cheek, cv2.COLOR_BGR2YCrCb)
        
        # Use minimum texture variation to avoid lighting interference
        left_texture = min(np.std(cheek_ycrcb_left[:,:,1]), np.std(cheek_ycrcb_left[:,:,2]))
        right_texture = min(np.std(cheek_ycrcb_right[:,:,1]), np.std(cheek_ycrcb_right[:,:,2]))
        
        texture_uniformity = min(left_texture, right_texture)
        uniform_texture = texture_uniformity < self.makeup_thresholds['foundation_smoothness'] * 2.5
        detection_details['uniform_texture'] = uniform_texture
        
        # Calculate makeup score with adjusted weights for selfies
        makeup_score = 0
        if foundation_detected:
            makeup_score += 0.25
        if high_saturation:
            makeup_score += 0.15
        if uniform_texture:
            makeup_score += 0.15
        if lipstick_detected:
            makeup_score += 0.1
        
        # More lenient threshold for heavy makeup detection
        is_wearing_heavy_makeup = makeup_score > 0.8
        
        # Add confidence level to detection details
        detection_details['makeup_confidence'] = makeup_score * 100
        detection_details['light_makeup_warning'] = 0.4 < makeup_score <= 0.8
        
        return is_wearing_heavy_makeup, detection_details
    
    def check_lighting_conditions(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        Check if the selfie was taken in appropriate lighting conditions,
        with more lenient thresholds for typical selfie scenarios
        Returns: (is_good_lighting, message)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:,:,2]
        
        # Calculate metrics for different regions
        h, w = v_channel.shape
        center_region = v_channel[h//4:3*h//4, w//4:3*w//4]
        
        # Analyze center face region separately
        center_brightness = np.mean(center_region)
        center_std = np.std(center_region)
        
        # Overall image metrics
        avg_brightness = np.mean(v_channel)
        brightness_std = np.std(v_channel)
        
        # More lenient thresholds for selfies
        too_dark = avg_brightness < 60  # Reduced threshold
        too_bright = avg_brightness > 240  # Increased threshold
        
        # Calculate percentage of well-lit pixels with more lenient range
        well_lit_pixels = np.sum((v_channel >= 50) & (v_channel <= 240)) / v_channel.size
        mostly_well_lit = well_lit_pixels > 0.6  # Reduced to 60%
        
        # Check center face region specifically
        center_too_dark = center_brightness < 50
        center_too_bright = center_brightness > 240
        
        # More lenient uneven lighting check
        uneven_lighting = brightness_std > 80 and center_std > 50
        
        # Handling common selfie scenarios
        if too_dark and center_too_dark:
            return False, "The image is too dark. Please take the photo in better lighting or move closer to a light source."
        elif too_bright and center_too_bright:
            return False, "The image is too bright. Try moving away from direct light or turning off flash."
        elif uneven_lighting and not mostly_well_lit:
            # Check if it's just normal selfie shadowing
            if center_brightness >= 50 and center_brightness <= 240:
                return True, "Acceptable lighting for analysis, though more even lighting would improve results."
            return False, "The lighting is very uneven. Try facing towards a window or light source."
        elif mostly_well_lit or (center_brightness >= 50 and center_brightness <= 240):
            if brightness_std > 60:
                return True, "Lighting is acceptable for analysis, though more even lighting would give better results."
            return True, "Lighting conditions are good for analysis."
        
        return True, "Lighting conditions are acceptable but could be improved for better results."


    def extract_face(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract face regions safely"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            raise ValueError("No face detected")
        
        x, y, w, h = faces[0]
        face = image[y:y+h, x:x+w]
        
        standard_size = (300, 300)
        face_resized = cv2.resize(face, standard_size)
        
        h, w = standard_size
        regions = {
            'forehead': face_resized[0:int(h*0.33), :],
            'cheeks': face_resized[int(h*0.33):int(h*0.66), :],
            'chin': face_resized[int(h*0.66):h, :],
            'full_face': face_resized
        }
        
        return regions

    def analyze_texture(self, image: np.ndarray) -> float:
        """Analyze skin texture"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        local_var = cv2.Laplacian(blurred, cv2.CV_64F).var()
        texture_score = max(0, min(100, 100 - (local_var / 100)))
        return texture_score

    def analyze_spots(self, image: np.ndarray) -> Dict[str, float]:
        """
        Enhanced analysis of dark spots, blemishes, and skin discoloration
        with balanced scoring and fixed output formatting
        """
        # Convert image to different color spaces for better analysis
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Initialize spots dictionary
        spots = {}
        
        # Get individual channels
        l_channel = lab[:,:,0]
        a_channel = lab[:,:,1]
        b_channel = lab[:,:,2]
        
        # Common kernel for morphological operations
        kernel = np.ones((3,3), np.uint8)
        
        def analyze_dark_spots():
            # Use local contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_l = clahe.apply(l_channel)
            
            # Calculate mean and std of L channel
            mean_l = np.mean(enhanced_l)
            std_l = np.std(enhanced_l)
            
            # Create adaptive threshold
            dark_threshold = mean_l - (1.0 * std_l)
            dark_mask = (enhanced_l < dark_threshold).astype(np.uint8) * 255
            
            # Clean up noise
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
            
            # Calculate density with adjusted scaling
            density = np.sum(dark_mask) / dark_mask.size
            score = max(20, min(100, 100 - (density * 300)))  # Adjusted base score
            
            return score, dark_mask
        
        def analyze_red_spots():
            # Normalize a* channel
            a_norm = cv2.normalize(a_channel, None, 0, 255, cv2.NORM_MINMAX)
            mean_a = np.mean(a_norm)
            std_a = np.std(a_norm)
            
            # Adjusted threshold for redness
            thresh_value = mean_a + (1.0 * std_a)
            red_mask = (a_norm > thresh_value).astype(np.uint8) * 255
            
            # Clean up mask
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            
            # Calculate score with minimum baseline
            density = np.sum(red_mask) / red_mask.size
            score = max(20, min(100, 100 - (density * 250)))  # Adjusted scaling
            
            return score, red_mask
        
        def analyze_brown_spots():
            # Enhanced brown spot detection using HSV
            lower_brown = np.array([10, 20, 50])
            upper_brown = np.array([25, 255, 255])
            
            brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
            
            # Clean up mask
            brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, kernel)
            
            # Calculate score with minimum baseline
            density = np.sum(brown_mask) / brown_mask.size
            score = max(20, min(100, 100 - (density * 200)))  # Adjusted scaling
            
            return score, brown_mask
        
        # Get scores and masks
        dark_score, dark_mask = analyze_dark_spots()
        red_score, red_mask = analyze_red_spots()
        brown_score, brown_mask = analyze_brown_spots()
        
        # Store individual scores
        spots['dark_spots'] = float(dark_score)  # Ensure float values
        spots['red_spots'] = float(red_score)
        spots['brown_spots'] = float(brown_score)
        
        # Analyze combined spots
        combined_mask = cv2.bitwise_or(
            dark_mask,
            cv2.bitwise_or(red_mask, brown_mask)
        )
        
        # Connected components analysis
        num_labels, labels = cv2.connectedComponents(combined_mask)
        spots['spot_count'] = float(num_labels - 1)  # Convert to float
        
        # Analyze distribution pattern with improved scoring
        if spots['spot_count'] > 0:
            spot_sizes = [np.sum(labels == i) for i in range(1, num_labels)]
            avg_spot_size = np.mean(spot_sizes) if spot_sizes else 0
            
            # Normalized factors
            count_factor = min(1.0, spots['spot_count'] / 150)  # Adjusted normalization
            size_factor = min(1.0, avg_spot_size / 100)
            
            # Calculate distribution score
            distribution_score = 100 - ((count_factor * 0.5 + size_factor * 0.5) * 80)  # Adjusted max penalty
            spots['pattern_distribution'] = float(max(20, min(100, distribution_score)))
        else:
            spots['pattern_distribution'] = 100.0
        
        # Calculate severity with adjusted weights and minimum score
        severity = (
            (dark_score * 0.3) +
            (red_score * 0.3) +
            (brown_score * 0.2) +
            (spots['pattern_distribution'] * 0.2)
        )
        spots['severity'] = float(max(20, min(100, severity)))
        
        # Calculate overall score with balanced weights
        overall = (
            (dark_score * 0.25) +
            (red_score * 0.25) +
            (brown_score * 0.25) +
            (spots['pattern_distribution'] * 0.25)
        )
        spots['overall'] = float(max(20, min(100, overall)))
        
        # Updated classification with float values
        spots['classification'] = {
            'minor': float(spots['spot_count']) if spots['severity'] > 70 else 0.0,
            'moderate': float(spots['spot_count']) if 40 <= spots['severity'] <= 70 else 0.0,
            'significant': float(spots['spot_count']) if spots['severity'] < 40 else 0.0
        }
        
        return spots

    def analyze_brightness(self, image: np.ndarray) -> float:
        """Analyze skin brightness"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]
        brightness = np.mean(l_channel)
        return max(0, min(100, (brightness / 255) * 100))

    def analyze_evenness(self, image: np.ndarray) -> float:
        """Analyze skin tone evenness"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]
        evenness = np.std(l_channel)
        return max(0, min(100, 100 - (evenness / 2)))

    def analyze_hydration(self, image: np.ndarray) -> float:
        """Analyze skin hydration"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]
        highlights = np.percentile(l_channel, 90)
        shadow = np.percentile(l_channel, 10)
        hydration_score = (highlights - shadow) / 255 * 100
        return max(0, min(100, 100 - hydration_score))

    def calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        score = 0
        for metric, weight in self.weights.items():
            if isinstance(metrics[metric], dict) and 'overall' in metrics[metric]:
                value = metrics[metric]['overall']
            else:
                value = metrics[metric]
            score += value * weight
        return max(10, min(100, score))

    def generate_recommendations(self, metrics: Dict) -> Dict:
        """Generate personalized recommendations"""
        recommendations = {
            'concerns': [],
            'ingredients': [],
            'products': [],
            'routine': {
                'morning': [],
                'evening': [],
                'weekly': []
            }
        }
        
        # Texture recommendations
        if metrics['texture'] < 70:
            recommendations['concerns'].append("Uneven texture")
            recommendations['ingredients'].extend([
                "Glycolic Acid",
                "Salicylic Acid",
                "Lactic Acid"
            ])
            recommendations['products'].extend([
                "Gentle Exfoliator",
                "Chemical Peel"
            ])
            recommendations['routine']['weekly'].append(
                "Use gentle exfoliator 2-3 times per week"
            )
        
        # Spots recommendations
        if metrics['spots']['overall'] < 70:
            recommendations['concerns'].append("Hyperpigmentation")
            recommendations['ingredients'].extend([
                "Vitamin C",
                "Niacinamide",
                "Alpha Arbutin",
                "Kojic Acid"
            ])
            recommendations['products'].extend([
                "Brightening Serum",
                "Dark Spot Treatment"
            ])
            recommendations['routine']['morning'].append(
                "Apply vitamin C serum"
            )
        
        # Hydration recommendations
        if metrics['hydration'] < 70:
            recommendations['concerns'].append("Dehydration")
            recommendations['ingredients'].extend([
                "Hyaluronic Acid",
                "Glycerin",
                "Ceramides"
            ])
            recommendations['products'].extend([
                "Hydrating Toner",
                "Moisturizing Cream"
            ])
            recommendations['routine']['morning'].append(
                "Use hydrating toner"
            )
            recommendations['routine']['evening'].append(
                "Apply rich moisturizer"
            )
        
        return recommendations

    def analyze_image(self, image_path: str) -> Dict:
        """Complete skin analysis with improved selfie handling"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Extract face regions with improved selfie handling
            regions = self.extract_face(image)
            face = regions['full_face']
            
            # Check lighting conditions with selfie considerations
            is_good_lighting, lighting_message = self.check_lighting_conditions(face)
            
            # Continue analysis even with suboptimal lighting
            metrics = {
                'texture': self.analyze_texture(face),
                'spots': self.analyze_spots(face),
                'brightness': self.analyze_brightness(face),
                'evenness': self.analyze_evenness(face),
                'hydration': self.analyze_hydration(face),
                'lighting_quality': {
                    'is_optimal': is_good_lighting,
                    'message': lighting_message
                },
                'regional_analysis': {
                    'forehead': {
                        'texture': self.analyze_texture(regions['forehead']),
                        'spots': self.analyze_spots(regions['forehead'])
                    },
                    'cheeks': {
                        'texture': self.analyze_texture(regions['cheeks']),
                        'spots': self.analyze_spots(regions['cheeks'])
                    },
                    'chin': {
                        'texture': self.analyze_texture(regions['chin']),
                        'spots': self.analyze_spots(regions['chin'])
                    }
                }
            }
            
            # Check for makeup
            is_wearing_heavy_makeup, makeup_details = self.detect_makeup(face)
            
            # Add warnings and confidence levels
            metrics['warnings'] = []
            if not is_good_lighting:
                metrics['warnings'].append({
                    'type': 'lighting',
                    'message': lighting_message,
                    'severity': 'moderate'
                })
            
            if is_wearing_heavy_makeup:
                metrics['warnings'].append({
                    'type': 'makeup',
                    'message': "Heavy makeup detected - results may be less accurate",
                    'severity': 'high'
                })
            elif makeup_details.get('light_makeup_warning'):
                metrics['warnings'].append({
                    'type': 'makeup',
                    'message': "Light makeup detected - results may be slightly affected",
                    'severity': 'low'
                })
            
            metrics['overall_score'] = self.calculate_overall_score(metrics)
            metrics['recommendations'] = self.generate_recommendations(metrics)
            metrics['makeup_details'] = makeup_details
            metrics['confidence_score'] = self._calculate_confidence_score(metrics)
            
            return metrics
                
        except Exception as e:
            self.logger.error(f"Error analyzing image: {str(e)}")
            return {'error': 'analysis_failed', 'message': str(e)}
        
    def _calculate_confidence_score(self, metrics: Dict) -> float:
        """Calculate confidence score for the analysis based on various factors"""
        confidence = 100.0
        
        # Reduce confidence based on lighting
        if not metrics['lighting_quality']['is_optimal']:
            confidence *= 0.8
        
        # Reduce confidence based on makeup
        if 'makeup_details' in metrics:
            makeup_score = metrics['makeup_details'].get('makeup_confidence', 0)
            if makeup_score > 80:
                confidence *= 0.6
            elif makeup_score > 40:
                confidence *= 0.8
        
        # Adjust for extreme brightness/darkness
        if metrics['brightness'] < 30 or metrics['brightness'] > 90:
            confidence *= 0.9
        
        return max(0, min(100, confidence))
    def generate_skin_profile(self, metrics: Dict) -> Dict:
        """
        Generate a standardized skin profile for matching purposes
        Returns a dictionary of normalized skin attributes
        """
        profile = {
            'skin_attributes': {
                'texture_quality': {
                    'score': metrics['texture'],
                    'weight': 0.25,
                    'category': self._categorize_score(metrics['texture'])
                },
                'hydration_level': {
                    'score': metrics['hydration'],
                    'weight': 0.15,
                    'category': self._categorize_score(metrics['hydration'])
                },
                'brightness': {
                    'score': metrics['brightness'],
                    'weight': 0.15,
                    'category': self._categorize_score(metrics['brightness'])
                },
                'evenness': {
                    'score': metrics['evenness'],
                    'weight': 0.20,
                    'category': self._categorize_score(metrics['evenness'])
                },
                'spot_profile': {
                    'score': metrics['spots']['overall'],
                    'weight': 0.25,
                    'category': self._categorize_score(metrics['spots']['overall']),
                    'details': {
                        'dark_spots': metrics['spots']['dark_spots'],
                        'red_spots': metrics['spots']['red_spots'],
                        'brown_spots': metrics['spots']['brown_spots'],
                        'distribution': metrics['spots']['pattern_distribution']
                    }
                }
            },
            'regional_variations': {
                'forehead': {
                    'texture': metrics['regional_analysis']['forehead']['texture'],
                    'spots': metrics['regional_analysis']['forehead']['spots']['overall']
                },
                'cheeks': {
                    'texture': metrics['regional_analysis']['cheeks']['texture'],
                    'spots': metrics['regional_analysis']['cheeks']['spots']['overall']
                },
                'chin': {
                    'texture': metrics['regional_analysis']['chin']['texture'],
                    'spots': metrics['regional_analysis']['chin']['spots']['overall']
                }
            },
            'overall_metrics': {
                'score': metrics['overall_score'],
                'category': self._categorize_score(metrics['overall_score']),
                'primary_concerns': self._identify_primary_concerns(metrics)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return profile

    def _categorize_score(self, score: float) -> str:
        """Categorize numerical scores into descriptive categories"""
        if score >= 80:
            return "excellent"
        elif score >= 65:
            return "good"
        elif score >= 50:
            return "moderate"
        elif score >= 35:
            return "fair"
        else:
            return "needs_attention"

    def _identify_primary_concerns(self, metrics: Dict) -> List[str]:
        """Identify primary skin concerns based on metrics"""
        concerns = []
        thresholds = {
            'hydration': {'score': metrics['hydration'], 'threshold': 70, 'concern': 'dehydration'},
            'brightness': {'score': metrics['brightness'], 'threshold': 70, 'concern': 'dullness'},
            'evenness': {'score': metrics['evenness'], 'threshold': 70, 'concern': 'uneven_tone'},
            'spots': {'score': metrics['spots']['overall'], 'threshold': 70, 'concern': 'spots'},
            'texture': {'score': metrics['texture'], 'threshold': 70, 'concern': 'texture'}
        }
        
        for metric, data in thresholds.items():
            if data['score'] < data['threshold']:
                concerns.append(data['concern'])
        
        return concerns
    def save_skin_profile(self, profile: Dict, filename: str = None) -> str:
        """Save skin profile to a JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"skin_profile_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(profile, f, indent=4)
            return filename
        except Exception as e:
            self.logger.error(f"Error saving skin profile: {str(e)}")
            raise



def main():
    # Initialize analyzer
    analyzer = SkinAnalyzer()
    
    # Example image path
    image_path = "/home/sourav/Desktop/SkinCare/WhatsApp Image 2024-11-04 at 20.35.57.jpeg"
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    # Analyze image
    results = analyzer.analyze_image(image_path)
    
    if 'error' in results:
        print(f"\nError: {results['message']}")
        if 'makeup_details' in results:
            print("\nMakeup Detection Details:")
            for detail, value in results['makeup_details'].items():
                print(f"- {detail}: {value}")
        return
    
    # Generate skin profile
    skin_profile = analyzer.generate_skin_profile(results)
    
    # Save profile
    profile_file = analyzer.save_skin_profile(skin_profile)
    
    # Print simplified results
    print("\nSkin Profile Generated:")
    print(f"Overall Score: {skin_profile['overall_metrics']['score']:.1f}/100")
    print(f"Category: {skin_profile['overall_metrics']['category']}")
    
    print("\nKey Attributes:")
    for attr, data in skin_profile['skin_attributes'].items():
        if isinstance(data, dict) and 'score' in data:
            print(f"- {attr.replace('_', ' ').title()}: {data['score']:.1f} ({data['category']})")
    
    print("\nPrimary Concerns:")
    for concern in skin_profile['overall_metrics']['primary_concerns']:
        print(f"- {concern.replace('_', ' ').title()}")
    
    print(f"\nProfile saved to: {profile_file}")

if __name__ == "__main__":
    main()
