
import sys
import os
import random

# Add project root to path
sys.path.insert(0, os.getcwd())

import graders

def test_graders():
    print("Running Grader Self-Check...")
    
    mock_histories = [
        [], # empty
        [{"Revenue": 5000, "Departments_Funded": 5, "Valuation": 100000, "RD_Progress": 0, "Customer Satisfaction": 50}] * 1, # single step
        [{"Revenue": 10000, "Departments_Funded": 5, "Valuation": 600000, "RD_Progress": 100, "Customer Satisfaction": 100}] * 12, # perfect run
        [{"Revenue": 0, "Departments_Funded": 0, "Valuation": 0, "RD_Progress": 0, "Customer Satisfaction": 0}] * 4, # worst run
    ]
    
    results = {}
    for name, grader_fn in graders.GRADERS.items():
        results[name] = []
        for i, hist in enumerate(mock_histories):
            score = grader_fn(hist)
            results[name].append(score)
            
            # Check range (0.01, 0.99)
            if not (0.009 <= score <= 0.991):
                print(f"FAILED: {name} returned out-of-range score: {score}")
                sys.exit(1)
            if score == 0.0 or score == 1.0:
                print(f"FAILED: {name} returned exactly 0.0 or 1.0: {score}")
                sys.exit(1)
                
        print(f"PASSED: {name} | Scores: {[round(s, 4) for s in results[name]]}")

    print("\nTotal Graders Found:", len(graders.GRADERS))
    if len(graders.GRADERS) < 3:
        print("FAILED: Not enough tasks with graders.")
        sys.exit(1)
        
    print("Grading System: READY (Strict 0,1 compliance confirmed)")

if __name__ == "__main__":
    test_graders()
