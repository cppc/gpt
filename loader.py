from data_loader.dataset import create_dataloader_v1

f = open("the-verdict.txt", "r", encoding="utf-8")
raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

data_iter = iter(dataloader)
batch = next(data_iter)
print(batch)
batch = next(data_iter)
print(batch)

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("Targets:\n", targets)
