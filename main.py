import pandas as pd
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import time
from typing import Dict, List
import logging

class CosIngExtractor:
    def __init__(self):
        self.base_url = "https://ec.europa.eu/growth/tools-databases/cosing"
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='cosing_extraction.log'
        )
        return logging.getLogger(__name__)
    
    def extract_ingredient_data(self, ingredient_name: str) -> Dict:
        """
        Extract detailed information about a specific ingredient
        """
        try:
            # Simulate search request
            search_url = f"{self.base_url}/ingredient/search"
            params = {
                'name': ingredient_name,
                'type': 'exact'
            }
            
            ingredient_data = {
                'name': ingredient_name,
                'functions': [],
                'restrictions': [],
                'regulatory_status': None,
                'inn_name': None,
                'ph_europe_name': None,
                'cas_number': None,
                'ec_number': None,
                'chemical_description': None
            }
            
            # Parse response
            # Note: This is a simplified version. In practice, you'd need to handle
            # the actual API endpoints and authentication if required
            
            return ingredient_data
            
        except Exception as e:
            self.logger.error(f"Error extracting ingredient data: {str(e)}")
            return {}

class IngredientAnalyzer:
    def __init__(self):
        self.ingredient_categories = {
            'emollients': ['glycerin', 'squalane', 'ceramides'],
            'humectants': ['hyaluronic acid', 'glycerin', 'urea'],
            'occlusives': ['petrolatum', 'dimethicone', 'shea butter'],
            'antioxidants': ['vitamin c', 'vitamin e', 'niacinamide'],
            'exfoliants': ['glycolic acid', 'salicylic acid', 'lactic acid'],
            'preservatives': ['phenoxyethanol', 'parabens', 'benzyl alcohol']
        }
        
        self.safety_ratings = {
            'low_risk': 1,
            'moderate_risk': 2,
            'high_risk': 3
        }
    
    def analyze_ingredient_list(self, ingredients: List[str]) -> Dict:
        """
        Analyze a list of ingredients for various properties
        """
        analysis = {
            'categories': self._categorize_ingredients(ingredients),
            'potential_concerns': self._identify_concerns(ingredients),
            'main_functions': self._identify_functions(ingredients),
            'safety_profile': self._assess_safety(ingredients)
        }
        return analysis
    
    def _categorize_ingredients(self, ingredients: List[str]) -> Dict:
        """
        Categorize ingredients by their primary functions
        """
        categories = {}
        for category, category_ingredients in self.ingredient_categories.items():
            matching_ingredients = [
                ing for ing in ingredients
                if any(cat_ing in ing.lower() for cat_ing in category_ingredients)
            ]
            if matching_ingredients:
                categories[category] = matching_ingredients
        return categories
    
    def _identify_concerns(self, ingredients: List[str]) -> List[str]:
        """
        Identify potential concerns with ingredients
        """
        concerns = []
        problematic_ingredients = {
            'fragrance': 'May cause sensitivity',
            'alcohol denat': 'Can be drying',
            'essential oils': 'Potential irritant',
            'methylisothiazolinone': 'Known allergen',
            'sodium lauryl sulfate': 'Can be harsh'
        }
        
        for ing in ingredients:
            for prob_ing, concern in problematic_ingredients.items():
                if prob_ing in ing.lower():
                    concerns.append(f"{ing}: {concern}")
        
        return concerns
    
    def _identify_functions(self, ingredients: List[str]) -> Dict:
        """
        Identify the main functions of the ingredient list
        """
        functions = {}
        ingredient_functions = {
            'moisturizing': ['glycerin', 'hyaluronic acid', 'ceramides'],
            'exfoliating': ['glycolic acid', 'salicylic acid', 'lactic acid'],
            'brightening': ['vitamin c', 'kojic acid', 'niacinamide'],
            'protecting': ['zinc oxide', 'titanium dioxide', 'vitamin e']
        }
        
        for function, func_ingredients in ingredient_functions.items():
            matching = [
                ing for ing in ingredients
                if any(func_ing in ing.lower() for func_ing in func_ingredients)
            ]
            if matching:
                functions[function] = matching
                
        return functions
    
    def _assess_safety(self, ingredients: List[str]) -> Dict:
        """
        Assess overall safety profile of ingredient list
        """
        safety_assessment = {
            'overall_rating': 'Low Risk',
            'notes': [],
            'allergen_count': 0,
            'irritant_count': 0
        }
        
        # Common allergens and irritants
        allergens = ['methylisothiazolinone', 'fragrance', 'formaldehyde']
        irritants = ['alcohol denat', 'sodium lauryl sulfate', 'menthol']
        
        for ing in ingredients:
            ing_lower = ing.lower()
            
            # Check allergens
            if any(allergen in ing_lower for allergen in allergens):
                safety_assessment['allergen_count'] += 1
                safety_assessment['notes'].append(f"Contains allergen: {ing}")
            
            # Check irritants
            if any(irritant in ing_lower for irritant in irritants):
                safety_assessment['irritant_count'] += 1
                safety_assessment['notes'].append(f"Contains irritant: {ing}")
        
        # Update overall rating based on counts
        total_concerns = safety_assessment['allergen_count'] + safety_assessment['irritant_count']
        if total_concerns > 3:
            safety_assessment['overall_rating'] = 'High Risk'
        elif total_concerns > 1:
            safety_assessment['overall_rating'] = 'Moderate Risk'
            
        return safety_assessment

def main():
    # Initialize extractors and analyzers
    cosing_extractor = CosIngExtractor()
    ingredient_analyzer = IngredientAnalyzer()
    
    # Example ingredient list
    ingredients = [
        "Water",
        "Glycerin",
        "Niacinamide",
        "Hyaluronic Acid",
        "Phenoxyethanol",
        "Fragrance"
    ]
    
    # Get detailed data for each ingredient
    ingredient_details = {}
    for ingredient in ingredients:
        details = cosing_extractor.extract_ingredient_data(ingredient)
        ingredient_details[ingredient] = details
        time.sleep(1)  # Be respectful with requests
    
    # Analyze the ingredient list
    analysis = ingredient_analyzer.analyze_ingredient_list(ingredients)
    
    # Print results
    print("Ingredient Analysis:")
    print("\nCategories:", analysis['categories'])
    print("\nPotential Concerns:", analysis['potential_concerns'])
    print("\nMain Functions:", analysis['main_functions'])
    print("\nSafety Profile:", analysis['safety_profile'])

if __name__ == "__main__":
    main()