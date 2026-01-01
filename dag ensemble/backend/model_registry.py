import os
import sys
import importlib.util
import inspect
from typing import Dict, List, Any 
import pandas as pd

class ModelRegistry:
    def __init__(self, models_dir: str):
        """
        models_dir: Path to the directory containing model python scripts (backend/models).
        """
        # We assume models_dir is the 'models' folder, not 'pretrained_models'.
        # Since main.py passes PRETRAINED_MODELS_DIR (which was wrong logic), 
        # we need to fix main.py or adjust here.
        # Ideally, main.py should pass the MODELS_CODE_DIR.
        # But for now, we can deduce it relative to this file?
        # Or better, let's look for 'models' directory relative to backend.
        
        self.backend_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.backend_dir, 'models')
        
        self.models: Dict[str, Any] = {}
        self.reload_models()

    def reload_models(self):
        """Scans backend/models directory and loads model classes."""
        self.models = {}
        
        if not os.path.exists(self.models_dir):
            print(f"Models directory not found: {self.models_dir}")
            return

        print(f"Scanning models in {self.models_dir}...")
        for filename in os.listdir(self.models_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                model_name = filename[:-3] # remove .py
                file_path = os.path.join(self.models_dir, filename)
                
                try:
                    # Dynamic Import
                    spec = importlib.util.spec_from_file_location(model_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[model_name] = module # optional but helps with pickles sometimes
                    spec.loader.exec_module(module)
                    
                    # Find the Model Class
                    # Heuristic: Look for a class that has 'predict' and 'fit' methods,
                    # OR matches the UpperCamelCase version of the filename?
                    # Let's simple check for a class that doesn't start with underscore
                    # and matches roughly the name or is the "main" class.
                    
                    found_class = None
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if obj.__module__ == model_name: # Defined in this file
                             # Check for required methods
                             if hasattr(obj, 'predict') and hasattr(obj, 'fit'):
                                 found_class = obj
                                 break
                    
                    if found_class:
                        print(f"Found model class {found_class.__name__} for {model_name}")
                        instance = found_class()
                        # Attempt to load pretrained state
                        # The class should have a load() method
                        if hasattr(instance, 'load'):
                             instance.load()
                        
                        self.models[model_name] = instance
                        print(f"Registered model: {model_name}")
                    else:
                        print(f"No valid model class found in {filename}")

                except Exception as e:
                    print(f"Failed to load model script {filename}: {e}")

    def get_model_names(self) -> List[str]:
        return list(self.models.keys())
    
    def get_model_metadata(self, model_name: str) -> Dict:
        if model_name in self.models:
             m = self.models[model_name]
             return {
                 "features": getattr(m, "features", []),
                 "target": getattr(m, "target", "unknown")
             }
        return {}

    def predict(self, model_name: str, features: Any) -> Any:
        if model_name not in self.models:
            return None
        
        # Delegate to the class instance
        # This instance handles feature engineering internally now!
        return self.models[model_name].predict(features)

    def train(self, model_name: str, data: Any) -> bool:
        """
        Trains the specified model with the provided data.
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found for training.")
            return False
            
        try:
            model_instance = self.models[model_name]
            if hasattr(model_instance, 'fit'):
                print(f"Registry: Training {model_name}...")
                model_instance.fit(data)
                
                # [NEW] Auto-Reload to ensure fresh state from disk
                if hasattr(model_instance, 'load'):
                    print(f"Registry: Reloading {model_name} from disk...")
                    model_instance.load()
                
                return True
            else:
                print(f"Model {model_name} does not have a 'fit' method.")
                return False
        except Exception as e:
            print(f"Error training model {model_name}: {e}")
            return False
