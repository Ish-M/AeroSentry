import math
import random
import time

# Configuration
TOTAL_ENGINES_TO_TEST = 100
CRITICAL_RUL_THRESHOLD = 20

def calculate_nasa_score(d):
    """Asymmetric NASA scoring function."""
    if d < 0:
        return math.exp(-d / 10.0) - 1
    else:
        return math.exp(d / 13.0) - 1

def run_fast_benchmark():
    print(f"⚡ Starting high-speed batch benchmark for {TOTAL_ENGINES_TO_TEST} engines...")
    
    start_time = time.time()
    
    # Metrics tracking
    total_events_processed = 0
    total_nasa_score = 0.0
    engine_histories = {}
    critical_alerts_triggered = 0

    # 1. GENERATE & PROCESS DATA INSTANTLY IN MEMORY
    for i in range(1, TOTAL_ENGINES_TO_TEST + 1):
        engine_id = f"Engine_{i:03d}"
        
        # Give each engine a random initial lifespan (cycles)
        initial_rul = random.randint(120, 250)
        
        engine_histories[engine_id] = {"latest_rul": 0, "total_penalty": 0.0, "alerts": 0}
        
        # Simulate the engine run down to 0 remaining cycles
        for true_rul in range(initial_rul, -1, -1):
            total_events_processed += 1
            
            # Simulate model estimation noise (-5 to +7 cycles)
            model_noise = random.randint(-5, 7)
            estimated_rul = max(0, true_rul + model_noise)
            
            # Calculate NASA error penalty
            d = estimated_rul - true_rul
            penalty_score = calculate_nasa_score(d)
            
            # Accumulate metrics
            total_nasa_score += penalty_score
            engine_histories[engine_id]["total_penalty"] += penalty_score
            engine_histories[engine_id]["latest_rul"] = estimated_rul
            
            # Check for failure detection
            if estimated_rul <= CRITICAL_RUL_THRESHOLD:
                engine_histories[engine_id]["alerts"] += 1
                critical_alerts_triggered += 1

    end_time = time.time()
    duration = end_time - start_time

    # 2. GENERATE SUMMARY REPORT
    print("\n" + "="*50)
    print("⚡ HIGH-SPEED BENCHMARK COMPLETE")
    print("="*50)
    print(f"Execution Time              : {duration:.4f} seconds")
    print(f"Total Engine Logs Processed : {total_events_processed:,}")
    print(f"Total Assets Evaluated      : {len(engine_histories)}")
    print(f"Cumulative NASA Penalty     : {total_nasa_score:.4f}")
    print(f"Critical Alerts Triggered   : {critical_alerts_triggered:,}")
    print("-"*50)
    
    # Show a snippet of the top 5 worst performing engines to keep terminal clean
    print("Top 5 Engines with Highest Penalty Scores:")
    sorted_engines = sorted(engine_histories.items(), key=lambda x: x[1]["total_penalty"], reverse=True)
    for eng, stats in sorted_engines[:5]:
        print(f" ⚙️ {eng} -> Total Penalty: {stats['total_penalty']:.2f} | Alerts Raised: {stats['alerts']}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_fast_benchmark()