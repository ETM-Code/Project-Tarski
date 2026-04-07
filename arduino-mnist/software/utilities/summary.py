#!/bin/env python3
import csv
import statistics

def main():
    prediction_times = []
    correct_count = 0
    total_count = 0
    
    with open('results.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue

            # Collect prediction times
            prediction_times.append(float(row[0]))
            # Count correct classifications
            if row[-1].strip().lower() == 'true':
                correct_count += 1
            total_count += 1
    
    if prediction_times:
        avg_time = statistics.mean(prediction_times)
        print(f"Average prediction time: {avg_time:.2f} ms")
    else:
        print("No data in results.csv")
        return
    
    incorrect_count = total_count - correct_count
    percentage = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"Total correct: {correct_count}")
    print(f"Total incorrect: {incorrect_count}")
    print(f"Percentage correct: {percentage:.2f}%")

if __name__ == "__main__":
    main()
