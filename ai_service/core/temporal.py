import numpy as np

class TemporalAnalyzer:
    def analyze(self, frame_results):
        """
        Step 3: Temporal Risk Calculation
        temporal_risk = standard_deviation(hf_frame_scores)
        """
        if not frame_results:
            return 0.0, []

        hf_scores = [r['hf_score'] for r in frame_results]
        
        # Calculate Standard Deviation
        if len(hf_scores) < 2:
            return 0.0, []
            
        temporal_risk = float(np.std(hf_scores))
        
        signals = []
        if temporal_risk > 0.15:
            signals.append("high_temporal_fluctuation")
            
        return temporal_risk, signals
