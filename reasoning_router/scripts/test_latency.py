import time

from reasoning_router import select_strategy
# from agent_action_guard.train_nn import dataset

if not dataset:
	raise ValueError("Dataset is empty. Please ensure the dataset is loaded correctly.")


def main():
	start_time = time.time()
	for row in dataset:
		_ = select_strategy(row['prompt'])
		
	end_time = time.time()
	print(f"Time taken: {end_time - start_time} seconds")
	print(f"Processed {len(dataset)} rows.")

	avg_time_per_action = (end_time - start_time) / len(dataset)
	avg_time_ms = avg_time_per_action * 1000.0
	print(f"Average time per action: {avg_time_ms:.4f} ms")


if __name__ == "__main__":
	main()
