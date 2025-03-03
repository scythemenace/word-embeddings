from datasets import load_dataset

dataset = load_dataset("wikipedia", "20220301.simple")

print(dataset["train"][0])
